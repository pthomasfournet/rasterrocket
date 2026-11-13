#!/usr/bin/env python3
"""Expected-results registry for the reference-comparison harnesses.

    python3 -I tests/compare/expectations.py compare --reference REF --dpi DPI
        --rmse-limit T --tile-limit K --file PDF [--registry PATH]
        [--update] [--strict] [--verbose] ROWS

The registry (tests/compare/expectations.json by default) lists the known
failing outcomes of compare.sh and pdfium_corpus.py.  It holds one list of
entries per suite: a harness and the settings that decide what passes
(reference renderer, resolution, RMSE and tile-delta limits), so entries
recorded under one setting never classify a run under another.  An entry
names a test, a document (`file`) or one of its pages (`file` and `page`,
1-based), and records:

- `outcome`: the failure expected.  For a page: `differs` (scored over a
  limit), `size`, `missing` or `unreadable`; for a document, the harness's
  class (`error`, `strict`, `panic`, ...).
- `reason`: why the failure is accepted; mandatory.  Seeded entries read
  `unclassified (seeded YYYY-MM-DD)` until someone reviews them.
- `rmse`, `tile` (optional, `differs` only): the largest value accepted
  for that metric, in place of the suite's limit.  A metric the entry does
  not list keeps the suite's limit; an entry listing neither accepts any
  value of both.

Every observed test is classified:

    PASS    passes, no entry
    XFAIL   fails as registered, within the registered limits
    FAIL    fails without an entry, with another outcome, or past the
            registered limits (unexpected)
    XPASS   passes although registered (the entry is to be removed or tightened)

A run fails on any FAIL, and on any XPASS under --strict.  With --update,
the registry is rewritten from the run: a new or changed failure gets a
seeded entry (limits 2 % above the observed values, rounded up to 0.01),
an XPASS entry is removed, and an XFAIL entry and every entry for a test
the run did not observe are kept as they are.

The `compare` command serves compare.sh: ROWS is a tab-separated file of
`label page outcome rmse tile detail` rows, `page` being `-` for the
document.  It prints a line per non-passing test (every test with
--verbose), then `STATUS <word> <fatal> <fail> <xfail> <xpass>`, where
<word> is PASS, XFAIL, XPASS or FAIL and <fatal> is 1 when the run fails.
"""

import argparse
import collections
import dataclasses
import datetime as dt
import enum
import json
import math
import os
import pathlib
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_REGISTRY = HERE / "expectations.json"
FORMAT = 1
METRICS = ("rmse", "tile")
SEED_HEADROOM = 1.02
PASSING = "pass"
DIFFERS = "differs"
# Observed outcomes that pass: a scored page within the limits, a document
# rendered without a failure.
SCORED = "scored"
OK = "ok"
ENTRY_KEYS = frozenset({"file", "page", "outcome", "reason", *METRICS})

TestId = tuple[str, int | None]


class RegistryError(Exception):
    """The registry file is malformed."""


class Verdict(enum.StrEnum):
    """The classification of one observed test."""

    PASS = "PASS"  # noqa: S105 -- a verdict, not a credential
    XFAIL = "XFAIL"
    FAIL = "FAIL"
    XPASS = "XPASS"


@dataclasses.dataclass(frozen=True)
class Observation:
    """One test outcome from a harness run.

    `outcome` is `scored` for a page compared pixel by pixel, with its
    `metrics`; `ok` for a document that rendered; any other value names a
    failure.
    """

    file: str
    page: int | None
    outcome: str
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    @property
    def test(self) -> TestId:
        """The test this observation is of."""
        return (self.file, self.page)


@dataclasses.dataclass(frozen=True)
class Entry:
    """A registered expected failure."""

    file: str
    page: int | None
    outcome: str
    reason: str
    limits: Mapping[str, float] = dataclasses.field(default_factory=dict)

    @property
    def test(self) -> TestId:
        """The test this entry is for."""
        return (self.file, self.page)


@dataclasses.dataclass(frozen=True)
class Result:
    """An observation, its verdict, the entry it was judged by and why."""

    observation: Observation
    verdict: Verdict
    why: str
    entry: Entry | None


def suite_name(harness: str, *, reference: str, dpi: int, rmse: float, tile: float) -> str:
    """Return the suite key of a harness run under the given settings."""
    return f"{harness} {reference} {dpi}dpi rmse<={rmse:g} tile<={tile:g}"


def test_label(test: TestId) -> str:
    """Return `file` or `file p<page>`."""
    file, page = test
    return file if page is None else f"{file} p{page}"


def effective_outcome(obs: Observation, limits: Mapping[str, float]) -> str:
    """Return `pass`, `differs` (a scored page over a limit) or the failure named."""
    if obs.outcome == SCORED:
        over = any(obs.metrics[m] > limits[m] for m in obs.metrics)
        return DIFFERS if over else PASSING
    if obs.outcome == OK:
        return PASSING
    return obs.outcome


def classify(obs: Observation, entry: Entry | None, limits: Mapping[str, float]) -> Result:
    """Return the verdict on `obs` given its registry entry and the suite limits."""
    outcome = effective_outcome(obs, limits)
    if outcome == PASSING:
        if entry is None:
            return Result(obs, Verdict.PASS, "", None)
        return Result(
            obs,
            Verdict.XPASS,
            f"registered as {entry.outcome}, now passes: remove or tighten the entry",
            entry,
        )
    if entry is None:
        return Result(obs, Verdict.FAIL, f"new failure ({outcome})", None)
    if entry.outcome != outcome:
        return Result(obs, Verdict.FAIL, f"registered as {entry.outcome}, now {outcome}", entry)
    if outcome == DIFFERS and entry.limits:
        allowed = {m: entry.limits.get(m, limits[m]) for m in obs.metrics}
        worse = [
            f"{m} {obs.metrics[m]:g} > {allowed[m]:g}"
            for m in sorted(obs.metrics)
            if obs.metrics[m] > allowed[m]
        ]
        if worse:
            return Result(obs, Verdict.FAIL, "worse than registered: " + ", ".join(worse), entry)
    return Result(obs, Verdict.XFAIL, f"expected: {entry.reason}", entry)


def classify_all(
    observations: Iterable[Observation],
    entries: Mapping[TestId, Entry],
    limits: Mapping[str, float],
) -> list[Result]:
    """Classify every observation against `entries`."""
    return [classify(obs, entries.get(obs.test), limits) for obs in observations]


def seed_limit(value: float) -> float:
    """Return the registered limit for an observed metric: 2 % headroom, up to 0.01."""
    # Rounding first keeps float error (15 * 1.02 * 100 = 1530.0000000000002)
    # from adding 0.01.
    return math.ceil(round(value * SEED_HEADROOM * 100, 6)) / 100


def seeded_entry(obs: Observation, limits: Mapping[str, float], today: dt.date) -> Entry:
    """Return the entry registering the current failure of `obs`."""
    outcome = effective_outcome(obs, limits)
    entry_limits = {}
    if outcome == DIFFERS:
        entry_limits = {m: seed_limit(v) for m, v in sorted(obs.metrics.items()) if v > limits[m]}
    reason = f"unclassified (seeded {today.isoformat()})"
    return Entry(obs.file, obs.page, outcome, reason, entry_limits)


def updated(
    entries: Mapping[TestId, Entry],
    results: Iterable[Result],
    limits: Mapping[str, float],
    today: dt.date,
) -> dict[TestId, Entry]:
    """Return `entries` rewritten to register the failures in `results`.

    A FAIL gets a seeded entry, an XPASS loses its entry, and every other
    entry, including those of tests absent from `results`, is kept.
    """
    out = dict(entries)
    for r in results:
        if r.verdict is Verdict.XPASS:
            del out[r.observation.test]
        elif r.verdict is Verdict.FAIL:
            out[r.observation.test] = seeded_entry(r.observation, limits, today)
    return out


def tally(results: Iterable[Result]) -> collections.Counter[Verdict]:
    """Return the number of results of each verdict."""
    return collections.Counter(r.verdict for r in results)


def overall(results: Sequence[Result]) -> Verdict:
    """Return FAIL if any test failed unexpectedly, else XPASS, XFAIL or PASS likewise."""
    counts = tally(results)
    for verdict in (Verdict.FAIL, Verdict.XPASS, Verdict.XFAIL):
        if counts[verdict]:
            return verdict
    return Verdict.PASS


def is_fatal(results: Sequence[Result], *, strict: bool) -> bool:
    """Return whether the run fails: any FAIL, or any XPASS under `strict`."""
    counts = tally(results)
    return counts[Verdict.FAIL] > 0 or (strict and counts[Verdict.XPASS] > 0)


# ── file format ──────────────────────────────────────────────────────────────


def _entry_from_json(suite: str, raw: object) -> Entry:
    """Return the entry `raw` describes; `RegistryError` when it is malformed."""
    where = f"suite {suite!r}"
    if not isinstance(raw, dict):
        msg = f"{where}: an entry is not an object: {raw!r}"
        raise RegistryError(msg)
    unknown = set(raw) - ENTRY_KEYS
    if unknown:
        msg = f"{where}: unknown keys {sorted(unknown)} in {raw!r}"
        raise RegistryError(msg)
    file, page, outcome, reason = (raw.get(k) for k in ("file", "page", "outcome", "reason"))
    if not isinstance(file, str) or not file:
        msg = f"{where}: entry without a file: {raw!r}"
        raise RegistryError(msg)
    where = f"{where}, {file}"
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        msg = f"{where}: page is not a positive integer: {page!r}"
        raise RegistryError(msg)
    if not isinstance(outcome, str) or outcome in {"", PASSING, SCORED, OK}:
        msg = f"{where}: outcome is not a failure: {outcome!r}"
        raise RegistryError(msg)
    if not isinstance(reason, str) or not reason.strip():
        msg = f"{where}: entry without a reason"
        raise RegistryError(msg)
    return Entry(file, page, outcome, reason, _limits_from_json(where, outcome, raw))


def _limits_from_json(where: str, outcome: str, raw: Mapping[str, object]) -> dict[str, float]:
    """Return the metric limits of entry `raw`; `RegistryError` when one is malformed."""
    limits = {}
    for m in METRICS:
        if m not in raw:
            continue
        v = raw[m]
        if outcome != DIFFERS:
            msg = f"{where}: {m} limit on a {outcome} entry"
            raise RegistryError(msg)
        if not isinstance(v, int | float) or isinstance(v, bool) or v < 0:
            msg = f"{where}: {m} is not a non-negative number: {v!r}"
            raise RegistryError(msg)
        limits[m] = float(v)
    return limits


def parse(text: str) -> dict[str, dict[TestId, Entry]]:
    """Return `{suite: {test: entry}}` from registry JSON; `RegistryError` if malformed."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        msg = f"not JSON: {e}"
        raise RegistryError(msg) from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        msg = f'expected an object with "format": {FORMAT}'
        raise RegistryError(msg)
    suites = doc.get("suites", {})
    if not isinstance(suites, dict) or set(doc) - {"format", "suites"}:
        msg = 'expected only "format" and a "suites" object'
        raise RegistryError(msg)
    out: dict[str, dict[TestId, Entry]] = {}
    for suite, raw_entries in suites.items():
        if not isinstance(raw_entries, list):
            msg = f"suite {suite!r} is not a list"
            raise RegistryError(msg)
        entries: dict[TestId, Entry] = {}
        for raw in raw_entries:
            entry = _entry_from_json(suite, raw)
            if entry.test in entries:
                msg = f"suite {suite!r}: {test_label(entry.test)} is listed twice"
                raise RegistryError(msg)
            entries[entry.test] = entry
        out[suite] = entries
    return out


def _sort_key(test: TestId) -> tuple[str, int]:
    """Order a document's own entry before its pages, pages by number."""
    file, page = test
    return (file, 0 if page is None else page)


def _entry_json(e: Entry) -> str:
    """Return the one-line JSON object of entry `e`."""
    row: dict[str, object] = {"file": e.file}
    if e.page is not None:
        row["page"] = e.page
    row["outcome"] = e.outcome
    row.update(sorted(e.limits.items()))
    row["reason"] = e.reason
    return json.dumps(row, ensure_ascii=False)


def dump(suites: Mapping[str, Mapping[TestId, Entry]]) -> str:
    """Return the registry JSON of `suites`, one entry a line; empty suites are dropped."""
    blocks = []
    for suite in sorted(suites):
        entries = suites[suite]
        if not entries:
            continue
        lines = ",\n".join(
            f"      {_entry_json(entries[t])}" for t in sorted(entries, key=_sort_key)
        )
        blocks.append(f"    {json.dumps(suite, ensure_ascii=False)}: [\n{lines}\n    ]")
    body = "{\n" + ",\n".join(blocks) + "\n  }" if blocks else "{}"
    return f'{{\n  "format": {FORMAT},\n  "suites": {body}\n}}\n'


def load(path: pathlib.Path) -> dict[str, dict[TestId, Entry]]:
    """Return the registry at `path`, empty when the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        return parse(text)
    except RegistryError as e:
        msg = f"{path}: {e}"
        raise RegistryError(msg) from e


def save(path: pathlib.Path, suites: Mapping[str, Mapping[TestId, Entry]]) -> None:
    """Write `suites` to `path`, replacing it atomically.

    The file keeps its permissions; a new one gets 0666 less the umask
    (`mkstemp` itself creates 0600).
    """
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dump(suites))
        pathlib.Path(tmp).replace(path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def run_suite(
    registry: pathlib.Path | None,
    suite: str,
    observations: Sequence[Observation],
    limits: Mapping[str, float],
    *,
    update: bool,
) -> list[Result]:
    """Classify `observations` against `suite` in `registry`; rewrite it with `update`.

    A `None` registry has no entries and is never written.  Seeded entries
    carry today's local date.
    """
    suites = {} if registry is None else load(registry)
    entries = suites.get(suite, {})
    results = classify_all(observations, entries, limits)
    if update and registry is not None:
        today = dt.date.today()  # noqa: DTZ011 -- a calendar date for a human reader
        suites[suite] = updated(entries, results, limits, today)
        save(registry, suites)
    return results


def registry_path(text: str) -> pathlib.Path | None:
    """Parse a registry path argument; the empty string disables the registry."""
    return pathlib.Path(text) if text else None


def file_key(path: str) -> str:
    """Return the registry name of a PDF: repository-relative, else its basename.

    The path is made absolute without resolving symbolic links, so a
    fixture linked into the repository keeps its repository path.
    """
    absolute = pathlib.Path(os.path.abspath(path))  # noqa: PTH100 -- keeps symbolic links
    try:
        return absolute.relative_to(REPO).as_posix()
    except ValueError:
        return absolute.name


# ── compare.sh helper ────────────────────────────────────────────────────────


def _read_rows(path: pathlib.Path, file: str) -> list[tuple[str, str, Observation]]:
    """Return `(label, detail, observation)` for each row compare.sh wrote."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        label, page, outcome, rmse, tile, detail = line.split("\t")
        metrics = {"rmse": float(rmse), "tile": float(tile)} if outcome == SCORED else {}
        obs = Observation(file, None if page == "-" else int(page), outcome, metrics)
        rows.append((label, detail, obs))
    return rows


def compare_main(args: argparse.Namespace) -> int:
    """Classify one compare.sh document and print its report and STATUS line."""
    limits = {"rmse": args.rmse_limit, "tile": args.tile_limit}
    suite = suite_name(
        "compare.sh",
        reference=args.reference,
        dpi=args.dpi,
        rmse=args.rmse_limit,
        tile=args.tile_limit,
    )
    rows = _read_rows(args.rows, file_key(args.file))
    results = run_suite(
        registry_path(args.registry),
        suite,
        [obs for _, _, obs in rows],
        limits,
        update=args.update,
    )
    for (label, detail, _), r in zip(rows, results, strict=True):
        if r.verdict is Verdict.PASS and not args.verbose:
            continue
        line = f"{r.verdict:<8} {label:<12}  {detail}"
        if r.why:
            line += f"  [{r.why}]"
        print(line)
    counts = tally(results)
    fatal = int(is_fatal(results, strict=args.strict) and not args.update)
    print(
        f"STATUS {overall(results)} {fatal} {counts[Verdict.FAIL]} "
        f"{counts[Verdict.XFAIL]} {counts[Verdict.XPASS]}"
    )
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Return the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="classify the pages of one compare.sh run")
    c.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="'' for none")
    c.add_argument("--reference", required=True)
    c.add_argument("--dpi", type=int, required=True)
    c.add_argument("--rmse-limit", type=float, required=True)
    c.add_argument("--tile-limit", type=float, required=True)
    c.add_argument("--file", required=True, help="the PDF, as compare.sh was given it")
    c.add_argument("--update", action="store_true")
    c.add_argument("--strict", action="store_true")
    c.add_argument("--verbose", action="store_true")
    c.add_argument("rows", type=pathlib.Path)
    return ap.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    """Run the command named on the command line."""
    args = parse_args(argv)
    try:
        return compare_main(args)
    except RegistryError as e:
        print(f"expectations: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
