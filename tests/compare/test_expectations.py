#!/usr/bin/env python3
"""Unit tests of the expected-results registry; no renderer or corpus needed.

Run with `python3 -I tests/compare/test_expectations.py`.
"""

import datetime as dt
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import expectations as ex

LIMITS = {"rmse": 2.0, "tile": 12.0}
TODAY = dt.date(2026, 1, 2)
SEEDED = "unclassified (seeded 2026-01-02)"


def scored(page: int, rmse: float, tile: float, file: str = "a.pdf") -> ex.Observation:
    """Return a scored page of `file`."""
    return ex.Observation(file, page, ex.SCORED, {"rmse": rmse, "tile": tile})


def differs(page: int, file: str = "a.pdf", **limits: float) -> ex.Entry:
    """Return a `differs` entry for a page of `file`."""
    return ex.Entry(file, page, ex.DIFFERS, "known", limits)


class Classify(unittest.TestCase):
    """The four verdicts."""

    def verdict(self, obs: ex.Observation, entry: ex.Entry | None) -> ex.Verdict:
        """Return the verdict on `obs` under `entry` and the default limits."""
        return ex.classify(obs, entry, LIMITS).verdict

    def test_pass_without_entry(self) -> None:
        """A page within the limits and unregistered passes."""
        self.assertEqual(self.verdict(scored(1, 1.0, 5.0), None), ex.Verdict.PASS)

    def test_document_ok_passes(self) -> None:
        """A document that rendered passes."""
        obs = ex.Observation("a.pdf", None, ex.OK)
        self.assertEqual(self.verdict(obs, None), ex.Verdict.PASS)

    def test_new_failure(self) -> None:
        """Over either limit without an entry is unexpected."""
        self.assertEqual(self.verdict(scored(1, 2.5, 5.0), None), ex.Verdict.FAIL)
        self.assertEqual(self.verdict(scored(1, 1.0, 12.5), None), ex.Verdict.FAIL)
        self.assertEqual(self.verdict(ex.Observation("a.pdf", 1, "size"), None), ex.Verdict.FAIL)

    def test_limit_is_inclusive(self) -> None:
        """A metric equal to its limit passes, as compare.sh's `>` test has it."""
        self.assertEqual(self.verdict(scored(1, 2.0, 12.0), None), ex.Verdict.PASS)
        self.assertEqual(self.verdict(scored(1, 3.0, 5.0), differs(1, rmse=3.0)), ex.Verdict.XFAIL)

    def test_expected_failure_within_registered_limit(self) -> None:
        """A failure within the entry's limit is expected."""
        result = ex.classify(scored(1, 1.0, 20.0), differs(1, tile=21.0), LIMITS)
        self.assertEqual(result.verdict, ex.Verdict.XFAIL)
        self.assertIn("known", result.why)

    def test_worse_than_registered(self) -> None:
        """Past the entry's limit is unexpected, and the reason names the metric."""
        result = ex.classify(scored(1, 1.0, 22.0), differs(1, tile=21.0), LIMITS)
        self.assertEqual(result.verdict, ex.Verdict.FAIL)
        self.assertIn("tile 22 > 21", result.why)

    def test_unlisted_metric_keeps_suite_limit(self) -> None:
        """An entry raising only the tile limit leaves RMSE at the suite's limit."""
        entry = differs(1, tile=21.0)
        self.assertEqual(self.verdict(scored(1, 2.5, 20.0), entry), ex.Verdict.FAIL)

    def test_entry_without_limits_accepts_any_value(self) -> None:
        """A `differs` entry listing no limit accepts any difference."""
        self.assertEqual(self.verdict(scored(1, 90.0, 200.0), differs(1)), ex.Verdict.XFAIL)

    def test_changed_outcome_is_unexpected(self) -> None:
        """A failure of another kind than registered is unexpected."""
        obs = ex.Observation("a.pdf", 1, "missing")
        self.assertEqual(self.verdict(obs, differs(1)), ex.Verdict.FAIL)
        doc = ex.Observation("a.pdf", None, "panic")
        entry = ex.Entry("a.pdf", None, "strict", "known")
        self.assertEqual(self.verdict(doc, entry), ex.Verdict.FAIL)

    def test_surprise_pass(self) -> None:
        """A registered test that passes is a surprise pass."""
        self.assertEqual(self.verdict(scored(1, 1.0, 5.0), differs(1)), ex.Verdict.XPASS)
        doc = ex.Observation("a.pdf", None, ex.OK)
        entry = ex.Entry("a.pdf", None, "error", "known")
        self.assertEqual(self.verdict(doc, entry), ex.Verdict.XPASS)


class Outcome(unittest.TestCase):
    """The run's overall verdict and exit status."""

    def results(self, *verdicts: ex.Verdict) -> list[ex.Result]:
        """Results with the given verdicts."""
        obs = ex.Observation("a.pdf", None, ex.OK)
        return [ex.Result(obs, v, "", None) for v in verdicts]

    def test_overall_precedence(self) -> None:
        """FAIL outranks XPASS, which outranks XFAIL, which outranks PASS."""
        v = ex.Verdict
        self.assertEqual(ex.overall(self.results()), v.PASS)
        self.assertEqual(ex.overall(self.results(v.PASS, v.XFAIL)), v.XFAIL)
        self.assertEqual(ex.overall(self.results(v.XFAIL, v.XPASS)), v.XPASS)
        self.assertEqual(ex.overall(self.results(v.XPASS, v.FAIL, v.XFAIL)), v.FAIL)

    def test_fatal(self) -> None:
        """Unexpected failures are fatal; surprise passes only under strict."""
        v = ex.Verdict
        self.assertFalse(ex.is_fatal(self.results(v.PASS, v.XFAIL), strict=True))
        self.assertTrue(ex.is_fatal(self.results(v.FAIL), strict=False))
        self.assertFalse(ex.is_fatal(self.results(v.XPASS), strict=False))
        self.assertTrue(ex.is_fatal(self.results(v.XPASS), strict=True))

    def test_update_without_registry_still_fails(self) -> None:
        """--update spares a failing run only when a registry recorded it."""
        failed = self.results(ex.Verdict.FAIL)
        registry = pathlib.Path("e.json")
        self.assertFalse(ex.run_is_fatal(failed, registry, update=True, strict=True))
        self.assertTrue(ex.run_is_fatal(failed, None, update=True, strict=False))
        self.assertTrue(ex.run_is_fatal(failed, registry, update=False, strict=False))


class Update(unittest.TestCase):
    """Rewriting the registry from a run."""

    def test_seeds_new_failures_and_drops_surprise_passes(self) -> None:
        """FAIL seeds an entry, XPASS removes one, XFAIL and unobserved entries stay."""
        entries: dict[ex.TestId, ex.Entry] = {
            ("a.pdf", 1): differs(1, tile=30.0),  # stays failing: kept
            ("a.pdf", 2): differs(2),  # passes now: dropped
            ("b.pdf", 1): differs(1, "b.pdf"),  # not observed: kept
        }
        observations = [
            scored(1, 1.0, 20.0),
            scored(2, 1.0, 5.0),
            scored(3, 3.333, 15.0),
            ex.Observation("a.pdf", None, "strict"),
        ]
        results = ex.classify_all(observations, entries, LIMITS)
        self.assertEqual(
            [r.verdict for r in results],
            [ex.Verdict.XFAIL, ex.Verdict.XPASS, ex.Verdict.FAIL, ex.Verdict.FAIL],
        )
        out = ex.updated(entries, results, LIMITS, TODAY)
        self.assertEqual(out[("a.pdf", 1)], entries[("a.pdf", 1)])
        self.assertNotIn(("a.pdf", 2), out)
        self.assertEqual(out[("b.pdf", 1)], entries[("b.pdf", 1)])
        self.assertEqual(
            out[("a.pdf", 3)],
            ex.Entry("a.pdf", 3, ex.DIFFERS, SEEDED, {"rmse": 3.40, "tile": 15.3}),
        )
        self.assertEqual(out[("a.pdf", None)], ex.Entry("a.pdf", None, "strict", SEEDED))

    def test_seeds_only_the_metrics_over_the_limit(self) -> None:
        """A page failing on tile alone registers a tile limit alone."""
        entry = ex.seeded_entry(scored(1, 0.5, 13.0), LIMITS, TODAY)
        self.assertEqual(entry.limits, {"tile": 13.26})

    def test_worse_failure_is_reseeded(self) -> None:
        """A failure past its limits gets a fresh entry with an unreviewed reason."""
        entries: dict[ex.TestId, ex.Entry] = {("a.pdf", 1): differs(1, tile=14.0)}
        results = ex.classify_all([scored(1, 1.0, 20.0)], entries, LIMITS)
        out = ex.updated(entries, results, LIMITS, TODAY)
        self.assertEqual(out[("a.pdf", 1)].limits, {"tile": 20.4})
        self.assertEqual(out[("a.pdf", 1)].reason, SEEDED)

    def test_seeded_limit_classifies_the_seeding_run_as_expected(self) -> None:
        """Re-running against a freshly seeded registry yields no FAIL or XPASS."""
        observations = [scored(1, 7.77, 99.99), scored(2, 2.01, 12.01)]
        first = ex.classify_all(observations, {}, LIMITS)
        seeded = ex.updated({}, first, LIMITS, TODAY)
        again = ex.classify_all(observations, seeded, LIMITS)
        self.assertEqual({r.verdict for r in again}, {ex.Verdict.XFAIL})


class FileFormat(unittest.TestCase):
    """Parsing and writing the registry."""

    def test_round_trip(self) -> None:
        """Parsing what `dump` wrote is the identity, and empty suites are dropped."""
        suites = {
            "s": {
                ("a.pdf", None): ex.Entry("a.pdf", None, "error", "broken xref"),
                ("a.pdf", 2): differs(2, rmse=3.5, tile=20.0),
            },
            "empty": {},
        }
        text = ex.dump(suites)
        self.assertEqual(ex.parse(text), {"s": suites["s"]})
        self.assertEqual(ex.dump(ex.parse(text)), text)

    def test_rejects_malformed_entries(self) -> None:
        """Missing reasons, bad pages, passing outcomes, stray and non-finite limits are errors."""
        bad = [
            '{"file": "a.pdf", "outcome": "error"}',
            '{"file": "a.pdf", "outcome": "error", "reason": " "}',
            '{"file": "a.pdf", "page": 0, "outcome": "size", "reason": "r"}',
            '{"file": "a.pdf", "outcome": "ok", "reason": "r"}',
            '{"file": "a.pdf", "outcome": "size", "tile": 3, "reason": "r"}',
            '{"file": "a.pdf", "outcome": "differs", "tile": -1, "reason": "r"}',
            '{"file": "a.pdf", "outcome": "differs", "tile": NaN, "reason": "r"}',
            '{"file": "a.pdf", "outcome": "differs", "rmse": Infinity, "reason": "r"}',
            '{"file": "a.pdf", "outcome": "error", "reason": "r", "note": "x"}',
        ]
        for entry in bad:
            with self.subTest(entry=entry), self.assertRaises(ex.RegistryError):
                ex.parse(f'{{"format": 1, "suites": {{"s": [{entry}]}}}}')

    def test_rejects_duplicates_and_other_formats(self) -> None:
        """A test listed twice, or another format number, is an error."""
        entry = '{"file": "a.pdf", "outcome": "error", "reason": "r"}'
        with self.assertRaises(ex.RegistryError):
            ex.parse(f'{{"format": 1, "suites": {{"s": [{entry}, {entry}]}}}}')
        with self.assertRaises(ex.RegistryError):
            ex.parse('{"format": 2, "suites": {}}')

    def test_save_keeps_permissions(self) -> None:
        """An existing registry keeps its mode across a save."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "e.json"
            path.write_text("{}", encoding="utf-8")
            path.chmod(0o640)
            ex.save(path, {"s": {("a.pdf", None): ex.Entry("a.pdf", None, "error", "r")}})
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertIn(("a.pdf", None), ex.load(path)["s"])

    def test_missing_file_is_empty(self) -> None:
        """A registry that does not exist has no entries."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ex.load(pathlib.Path(tmp) / "none.json"), {})


class CompareCommand(unittest.TestCase):
    """The `compare` command compare.sh runs."""

    def run_compare(
        self, registry: pathlib.Path, rows: str, *extra: str, registry_arg: str | None = None
    ) -> list[str]:
        """Run the command on `rows`; return its output lines.

        The rows file is written beside `registry`, which `--registry` names
        unless `registry_arg` replaces it.
        """
        rows_file = registry.parent / "rows.tsv"
        rows_file.write_text(rows, encoding="utf-8")
        cmd = [
            sys.executable,
            "-I",
            str(pathlib.Path(ex.__file__)),
            "compare",
            "--registry",
            str(registry) if registry_arg is None else registry_arg,
            "--reference",
            "pdftoppm",
            "--dpi",
            "150",
            "--rmse-limit",
            "2.0",
            "--tile-limit",
            "12",
            "--file",
            "/nowhere/x.pdf",
            *extra,
            str(rows_file),
        ]
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        return r.stdout.splitlines()

    def test_seed_then_classify(self) -> None:
        """--update seeds the failures, and the next run classifies them XFAIL."""
        rows = (
            "page-1\t1\tscored\t1.0\t5.0\tok\n"
            "page-2\t2\tscored\t3.0\t20.0\tbad\n"
            "page-3\t3\tsize\t\t\tref=1x1 rrocket=9x9\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            registry = pathlib.Path(tmp) / "e.json"
            first = self.run_compare(registry, rows, "--update", "--strict")
            self.assertEqual(first[-1], "STATUS FAIL 0 2 0 0")
            suite = "compare.sh pdftoppm 150dpi rmse<=2 tile<=12"
            entries = ex.load(registry)[suite]
            self.assertEqual(set(entries), {("x.pdf", 2), ("x.pdf", 3)})
            again = self.run_compare(registry, rows)
            self.assertEqual(again[-1], "STATUS XFAIL 0 0 2 0")
            fixed = rows.replace("size\t\t\t", "scored\t0.5\t1.0\t")
            surprise = self.run_compare(registry, fixed, "--strict")
            self.assertEqual(surprise[-1], "STATUS XPASS 1 0 1 1")
            self.assertTrue(surprise[1].startswith("XPASS    page-3"))

    def test_update_without_registry_is_fatal(self) -> None:
        """--update with the registry disabled records nothing and keeps the FAIL fatal."""
        rows = "page-1\t1\tscored\t3.0\t20.0\tbad\n"
        with tempfile.TemporaryDirectory() as tmp:
            registry = pathlib.Path(tmp) / "e.json"
            out = self.run_compare(registry, rows, "--update", registry_arg="")
            self.assertEqual(out[-1], "STATUS FAIL 1 1 0 0")
            self.assertFalse(registry.exists())


# Stub renderers: each writes a white 2 x 2 binary PPM as page 1 of the
# output prefix, its last argument.  The rrocket stub exits with
# $STUB_STATUS (default 0).
STUB_PDFTOPPM = """#!/bin/sh
for a; do prefix=$a; done
printf 'P6\\n2 2\\n255\\n' > "$prefix-1.ppm"
head -c 12 /dev/zero | tr '\\0' '\\377' >> "$prefix-1.ppm"
"""
STUB_RROCKET = STUB_PDFTOPPM + 'exit "${STUB_STATUS:-0}"\n'


@unittest.skipUnless(
    all(shutil.which(c) for c in ("compare", "identify", "convert", "bc")),
    "compare.sh needs ImageMagick and bc",
)
class CompareScript(unittest.TestCase):
    """compare.sh end to end, with stub renderers drawing identical pages."""

    SUITE = "compare.sh pdftoppm 150dpi rmse<=2 tile<=12"

    def setUp(self) -> None:
        """Create the stub renderers, a PDF argument and an empty registry path."""
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        bin_dir = self.dir / "bin"
        bin_dir.mkdir()
        for name, text in (("pdftoppm", STUB_PDFTOPPM), ("rrocket", STUB_RROCKET)):
            stub = bin_dir / name
            stub.write_text(text, encoding="utf-8")
            stub.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "RROCKET_BIN": str(bin_dir / "rrocket"),
        }
        self.pdf = self.dir / "x.pdf"
        self.pdf.write_bytes(b"%PDF-1.4\n")
        self.registry = self.dir / "e.json"

    def run_script(self, *args: str, status: int = 0) -> subprocess.CompletedProcess[str]:
        """Run compare.sh on the PDF with `args`, the rrocket stub exiting `status`."""
        script = pathlib.Path(ex.__file__).with_name("compare.sh")
        cmd = [str(script), "-E", str(self.registry), *args, str(self.pdf)]
        env = {**self.env, "STUB_STATUS": str(status)}
        return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)  # noqa: S603

    def test_document_that_renders_is_observed(self) -> None:
        """A document entry seeded from a failure becomes XPASS once rrocket succeeds."""
        seed = self.run_script("-U", status=1)
        self.assertEqual(seed.returncode, 0, seed.stderr)
        entries = ex.load(self.registry)[self.SUITE]
        self.assertEqual(entries[("x.pdf", None)].outcome, "error")
        fixed = self.run_script("-S")
        self.assertEqual(fixed.returncode, 1, fixed.stdout)
        self.assertIn("XPASS    document", fixed.stdout)
        self.run_script("-U")
        self.assertNotIn(self.SUITE, ex.load(self.registry))


if __name__ == "__main__":
    unittest.main()
