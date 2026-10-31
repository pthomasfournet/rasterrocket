#!/usr/bin/env python3
"""Score rrocket on pdfium's test corpus against pdfium's expected renderings.

    python3 -I tests/compare/pdfium_corpus.py --pdfium PDFIUM_SRC --out DIR
        [--rrocket BIN] [--jobs N] [--timeout S] [--pdftoppm] [--match TEXT]

The corpus is PDFIUM_SRC/testing/resources (BSD-3-Clause, see
PDFIUM_SRC/LICENSE).  It is read in place; nothing from it is copied into
this repository.  Every generated file goes under DIR:

- DIR/expanded/: the `.in` templates without a sibling `.pdf`, expanded by
  pdfium's own testing/tools/fixup_pdf_template.py;
- DIR/results.tsv: one row per document;
- DIR/pages.tsv: one row per scored page.

Each document is rendered by rrocket at 72 dpi (1 px per point, the scale
of pdfium's expected images) with `--strict --backend cpu`, under a
timeout, and classified:

    ok          exit 0
    strict      exit 1 under --strict, exit 0 without it: some image or
                content stream was skipped (the non-strict render is scored)
    error       exit 1 without --strict either (the first stderr line says why)
    panic       exit 101
    abort       killed by SIGABRT (a panic under panic = "abort", or an
                allocation failure)
    crash       killed by another signal
    hang        no exit within the timeout
    no-pages    exit 0 and no page written

A strict failure whose non-strict rerun panics, aborts, crashes or hangs
takes the rerun's class.

Pages with a plain `<stem>_expected.pdf.<N>.png` next to the source (no
platform or backend suffix) are scored against it with score_pair.sh: RMSE,
tile delta, or a size mismatch.  With --pdftoppm, pages of documents
without an expected image are scored against `pdftoppm -cropbox -r 72`.

Requires: rrocket (release build), ImageMagick, python3; pdftoppm for
--pdftoppm.
"""

import argparse
import collections
import concurrent.futures
import dataclasses
import functools
import os
import pathlib
import re
import shutil
import signal
import sys
import tempfile
from collections.abc import Iterable, Sequence
from typing import NamedTuple

# The shared helpers live in scripts/; `python3 -I` puts no script directory on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from runtool import (
    executable,
    last_line,
    positive_int,
    require,
    run,
    run_limited,
)

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(__file__).resolve().parents[2]
SCORE = HERE / "score_pair.sh"
EXPECTED = re.compile(r"^(?P<stem>.+)_expected\.pdf\.(?P<page>\d+)\.png$")
PAGE_FILE = re.compile(r"-(\d+)$")
STRICT_EXIT = 1
PANIC_EXIT = 101
FAILURES = ("panic", "abort", "crash", "hang")


@dataclasses.dataclass(frozen=True)
class Config:
    """The settings every document run shares."""

    resources: pathlib.Path
    rrocket: pathlib.Path
    pdftoppm: str | None
    timeout: int
    work: pathlib.Path


class Doc(NamedTuple):
    """A PDF to render and the corpus file it came from (itself or its template)."""

    pdf: pathlib.Path
    source: pathlib.Path


class DocResult(NamedTuple):
    """One row of results.tsv."""

    document: str
    cls: str
    exit: str
    pages: int
    reference: str
    first_stderr_line: str


class PageRow(NamedTuple):
    """One row of pages.tsv: a `scored`, `size`, `missing` or `unreadable` page."""

    document: str
    page: int
    reference: str
    result: str
    rmse_or_ref_size: str
    tile_or_new_size: str


def expand_templates(resources: pathlib.Path, out: pathlib.Path) -> list[Doc]:
    """Return the PDFs to run, expanding every `.in` without a sibling `.pdf`."""
    docs = []
    templates: dict[pathlib.Path, list[pathlib.Path]] = {}
    for path in sorted(resources.rglob("*")):
        if path.suffix == ".pdf":
            docs.append(Doc(path, path))
        elif path.suffix == ".in" and not path.with_suffix(".pdf").exists():
            templates.setdefault(path.parent, []).append(path)
    fixup = resources.parent / "tools" / "fixup_pdf_template.py"
    if templates and not fixup.is_file():
        sys.exit(f"{len(templates)} template directories but no {fixup}")
    for directory, ins in templates.items():
        dest = out / "expanded" / directory.relative_to(resources)
        dest.mkdir(parents=True, exist_ok=True)
        for src in ins:
            # One template per run, so a template the tool cannot expand
            # does not stop the others.
            r = run([sys.executable, "-I", str(fixup), f"--output-dir={dest}", str(src)])
            pdf = dest / (src.stem + ".pdf")
            if r.returncode == 0 and pdf.exists():
                docs.append(Doc(pdf, src))
            else:
                print(f"expand failed: {src}: {last_line(r.stderr)}", file=sys.stderr)
    return docs


def expected_images(source: pathlib.Path) -> dict[int, pathlib.Path]:
    """Return `{page_number: png}` of the plain expected images of `source`.

    pdfium numbers its expected images from 0; the result numbers pages from 1.
    """
    found = {}
    for png in source.parent.glob(f"{source.stem}_expected.pdf.*.png"):
        m = EXPECTED.match(png.name)
        if m and m["stem"] == source.stem:
            found[int(m["page"]) + 1] = png
    return found


def numbered_pages(directory: pathlib.Path, prefix: str) -> dict[int, pathlib.Path]:
    """Return `{page_number: path}` of the `<prefix>-<n>.ppm` files in `directory`."""
    pages = {}
    for f in directory.glob(f"{prefix}-*.ppm"):
        m = PAGE_FILE.search(f.stem)
        if m:
            pages[int(m.group(1))] = f
    return pages


def run_rrocket(
    cfg: Config, pdf: pathlib.Path, prefix: pathlib.Path, *, strict: bool
) -> tuple[int | None, str]:
    """Return the exit code (`None` on timeout) and stderr of one render."""
    args = [str(cfg.rrocket), "--no-ram", "--backend", "cpu", "-r", "72"]
    if strict:
        args.append("--strict")
    r = run_limited([*args, str(pdf), str(prefix)], cfg.timeout)
    return (None, "") if r is None else (r.returncode, r.stderr)


def classify(code: int | None) -> str:
    """Return the class of a single run's exit code; exit 1 is `error`."""
    if code is None:
        return "hang"
    if code == 0:
        return "ok"
    if code == PANIC_EXIT:
        return "panic"
    if code == -signal.SIGABRT:
        return "abort"
    if code < 0:
        return "crash"
    return "error"


def score(ref: pathlib.Path, new: pathlib.Path) -> tuple[str, str, str]:
    """Return `(result, rmse_or_ref_size, tile_or_new_size)` from score_pair.sh."""
    r = run([str(SCORE), str(ref), str(new)])
    if r.returncode == 0:
        match r.stdout.split():
            case ["SIZE", ref_w, ref_h, new_w, new_h]:
                return ("size", f"{ref_w}x{ref_h}", f"{new_w}x{new_h}")
            case ["SCORE", rmse, tile]:
                return ("scored", rmse, tile)
            case _:
                pass
    return ("unreadable", "", "")


def render(cfg: Config, pdf: pathlib.Path, tmp: pathlib.Path) -> tuple[str, int | None, str]:
    """Render `pdf` into `tmp`; return its class, strict exit code and first diagnostic.

    A strict failure is re-rendered without `--strict`; the class is then
    `strict` when that succeeds, else the relaxed run's own class.
    """
    code, err = run_rrocket(cfg, pdf, tmp / "p", strict=True)
    cls = classify(code)
    if code == STRICT_EXIT:
        for f in tmp.glob("p-*.ppm"):
            f.unlink()
        relaxed, _ = run_rrocket(cfg, pdf, tmp / "p", strict=False)
        cls = "strict" if relaxed == 0 else classify(relaxed)
    # rrocket's own verdict, else its first warning.
    lines = [line for line in err.splitlines() if line.strip()]
    verdicts = [line for line in lines if line.startswith("rrocket:")]
    first_err = (verdicts or lines or [""])[0]
    return cls, code, first_err


def references(
    cfg: Config, doc: Doc, tmp: pathlib.Path, *, rendered: bool
) -> tuple[str, dict[int, pathlib.Path]]:
    """Return the reference kind and `{page_number: image}` for `doc`.

    pdfium's own expected images win; otherwise pdftoppm renders the
    references when enabled and rrocket wrote pages.
    """
    refs = expected_images(doc.source)
    if refs:
        return "pdfium", refs
    if cfg.pdftoppm is None or not rendered:
        return "", {}
    r = run_limited(
        [cfg.pdftoppm, "-cropbox", "-r", "72", str(doc.pdf), str(tmp / "ref")], cfg.timeout
    )
    if r is None:
        print(f"pdftoppm timed out: {doc.pdf}", file=sys.stderr)
        return "", {}
    if r.returncode != 0:
        return "", {}
    return "pdftoppm", numbered_pages(tmp, "ref")


def page_rows(
    name: str,
    ref_kind: str,
    refs: dict[int, pathlib.Path],
    pages: dict[int, pathlib.Path],
) -> list[PageRow]:
    """Score every reference page of `name` against the page rrocket wrote."""
    rows = []
    for page, ref in sorted(refs.items()):
        new = pages.get(page)
        if new is None:
            rows.append(PageRow(name, page, ref_kind, "missing", "", ""))
        else:
            rows.append(PageRow(name, page, ref_kind, *score(ref, new)))
    return rows


def process(doc: Doc, cfg: Config) -> tuple[DocResult, list[PageRow]]:
    """Render, classify and score one document."""
    name = str(doc.source.relative_to(cfg.resources))
    with tempfile.TemporaryDirectory(dir=cfg.work) as tmp_name:
        tmp = pathlib.Path(tmp_name)
        cls, code, first_err = render(cfg, doc.pdf, tmp)
        pages = numbered_pages(tmp, "p")
        if cls == "ok" and not pages:
            cls = "no-pages"
        ref_kind, refs = references(cfg, doc, tmp, rendered=bool(pages))
        rows = page_rows(name, ref_kind, refs, pages)
    exit_code = "" if code is None else str(code)
    return DocResult(name, cls, exit_code, len(pages), ref_kind, first_err), rows


def write_tsv(
    path: pathlib.Path, fields: Sequence[str], rows: Iterable[tuple[object, ...]]
) -> None:
    """Write `rows` under a `fields` header as tab-separated values."""
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        f.writelines("\t".join(str(v) for v in r) + "\n" for r in sorted(rows))


def summarise(results: list[DocResult], pages: list[PageRow]) -> None:
    """Print the class counts, the failures and the largest tile deltas."""
    counts = collections.Counter(r.cls for r in results)
    print(f"{len(results)} documents: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in sorted(results):
        if r.cls in FAILURES:
            print(f"  {r.cls:6} {r.document}  {r.first_stderr_line}")
    by_result = collections.Counter(p.result for p in pages)
    print(
        f"{len(pages)} reference pages: {by_result['scored']} scored, {by_result['size']} size "
        f"mismatches, {by_result['missing']} missing from rrocket's output, "
        f"{by_result['unreadable']} unreadable"
    )
    scored = [p for p in pages if p.result == "scored"]
    print("largest tile deltas:")
    for p in sorted(scored, key=lambda p: -float(p.tile_or_new_size))[:15]:
        print(
            f"  {float(p.tile_or_new_size):7.2f}  rmse {float(p.rmse_or_ref_size):6.2f}  "
            f"{p.reference:8}  {p.document} p{p.page}"
        )


def parse_args() -> argparse.Namespace:
    """Return the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdfium", type=pathlib.Path, required=True, help="pdfium source checkout")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
    ap.add_argument(
        "--rrocket",
        type=executable,
        default=str(REPO / "target/release/rrocket"),
        help="rrocket binary (cargo build --release -p rasterrocket-cli)",
    )
    ap.add_argument("--jobs", type=positive_int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=positive_int, default=60, help="seconds per rrocket run")
    ap.add_argument(
        "--pdftoppm",
        action="store_true",
        help="score documents without an expected image against pdftoppm",
    )
    ap.add_argument("--match", default="", help="only documents whose path contains TEXT")
    return ap.parse_args()


def main() -> None:
    """Run the corpus and write the tables under --out."""
    args = parse_args()
    resources = args.pdfium / "testing" / "resources"
    if not resources.is_dir():
        sys.exit(f"not a pdfium checkout: {args.pdfium}")
    require("compare")  # ImageMagick, for score_pair.sh
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        resources=resources,
        rrocket=args.rrocket,
        pdftoppm=require("pdftoppm") if args.pdftoppm else None,
        timeout=args.timeout,
        work=args.out / "work",
    )
    cfg.work.mkdir(exist_ok=True)

    docs = [d for d in expand_templates(resources, args.out) if args.match in str(d.source)]
    results: list[DocResult] = []
    pages: list[PageRow] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            for result, rows in pool.map(functools.partial(process, cfg=cfg), docs):
                results.append(result)
                pages.extend(rows)
    finally:
        shutil.rmtree(cfg.work, ignore_errors=True)

    write_tsv(
        args.out / "results.tsv",
        ("document", "class", "exit", "pages", "reference", "first_stderr_line"),
        results,
    )
    write_tsv(args.out / "pages.tsv", PageRow._fields, pages)
    summarise(results, pages)
    print(f"tables: {args.out / 'results.tsv'}, {args.out / 'pages.tsv'}")


if __name__ == "__main__":
    main()
