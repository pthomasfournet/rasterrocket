#!/usr/bin/env python3
"""Run rrocket over pdfium's test corpus and score it against pdfium's own
expected renderings.

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

Pages with a plain `<stem>_expected.pdf.<N>.png` next to the source (no
platform or backend suffix) are scored against it with score_pair.sh: RMSE,
tile delta, or a size mismatch.  With --pdftoppm, pages of documents
without an expected image are scored against `pdftoppm -cropbox -r 72`.

Requires: rrocket (release build), ImageMagick, python3; pdftoppm for
--pdftoppm.
"""

import argparse
import concurrent.futures
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCORE = HERE / "score_pair.sh"
EXPECTED = re.compile(r"^(?P<stem>.+)_expected\.pdf\.(?P<page>\d+)\.png$")


def expand_templates(resources, out):
    """Expand every `.in` without a sibling `.pdf`; returns the PDFs to run,
    each paired with its source path (for finding expected images)."""
    docs = []
    templates = {}
    for path in sorted(resources.rglob("*")):
        if path.suffix == ".pdf":
            docs.append((path, path))
        elif path.suffix == ".in" and not path.with_suffix(".pdf").exists():
            templates.setdefault(path.parent, []).append(path)
    fixup = resources.parent / "tools" / "fixup_pdf_template.py"
    for directory, ins in templates.items():
        dest = out / "expanded" / directory.relative_to(resources)
        dest.mkdir(parents=True, exist_ok=True)
        for src in ins:
            # One template per run, so a template the tool cannot expand
            # does not stop the others.
            r = subprocess.run(
                [sys.executable, "-I", str(fixup), f"--output-dir={dest}", str(src)],
                check=False,
                capture_output=True,
            )
            pdf = dest / (src.stem + ".pdf")
            if r.returncode == 0 and pdf.exists():
                docs.append((pdf, src))
            else:
                print(f"expand failed: {src}", file=sys.stderr)
    return docs


def expected_images(source):
    """`{page_index: png}` of the plain expected images of `source`."""
    found = {}
    for png in source.parent.glob(f"{source.stem}_expected.pdf.*.png"):
        m = EXPECTED.match(png.name)
        if m and m["stem"] == source.stem:
            found[int(m["page"])] = png
    return found


def run_rrocket(rrocket, pdf, prefix, timeout, strict):
    args = [str(rrocket), "--no-ram", "--backend", "cpu", "-r", "72"]
    if strict:
        args.append("--strict")
    try:
        r = subprocess.run(
            [*args, str(pdf), str(prefix)],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, ""
    return r.returncode, r.stderr.decode("utf-8", "replace")


def classify(code, strict_ok):
    if code is None:
        return "hang"
    if code == 0:
        return "ok"
    if code == 101:
        return "panic"
    if code == -signal.SIGABRT:
        return "abort"
    if code < 0:
        return "crash"
    return "strict" if strict_ok else "error"


def rendered_pages(directory):
    """`{page_number: path}` of the pages rrocket wrote as `p-<n>.ppm`."""
    pages = {}
    for f in directory.glob("p-*.ppm"):
        pages[int(f.stem.split("-")[-1])] = f
    return pages


def score(ref, new):
    r = subprocess.run(
        [str(SCORE), str(ref), str(new)], check=False, capture_output=True, text=True
    )
    if r.returncode != 0:
        return ("unreadable", "", "")
    kind, *vals = r.stdout.split()
    if kind == "SIZE":
        return ("size", f"{vals[0]}x{vals[1]}", f"{vals[2]}x{vals[3]}")
    return ("scored", vals[0], vals[1])


def process(doc, args, work):
    pdf, source = doc
    name = str(source.relative_to(args.resources))
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        tmp = pathlib.Path(tmp)
        code, err = run_rrocket(args.rrocket, pdf, tmp / "p", args.timeout, True)
        strict_ok = False
        if code == 1:
            for f in tmp.glob("p-*.ppm"):
                f.unlink()
            relaxed, _ = run_rrocket(args.rrocket, pdf, tmp / "p", args.timeout, False)
            strict_ok = relaxed == 0
        cls = classify(code, strict_ok)
        pages = rendered_pages(tmp)
        if cls == "ok" and not pages:
            cls = "no-pages"
        # rrocket's own verdict, else its first warning.
        lines = [l for l in err.splitlines() if l.strip()]
        first_err = next(
            (l for l in lines if l.startswith("rrocket:")), lines[0] if lines else ""
        )

        refs = {i + 1: png for i, png in expected_images(source).items()}
        ref_kind = "pdfium" if refs else ""
        if not refs and args.pdftoppm and pages:
            r = subprocess.run(
                ["pdftoppm", "-cropbox", "-r", "72", str(pdf), str(tmp / "ref")],
                check=False,
                capture_output=True,
                timeout=args.timeout,
            )
            if r.returncode == 0:
                refs = {int(f.stem.split("-")[-1]): f for f in tmp.glob("ref-*.ppm")}
                ref_kind = "pdftoppm"

        rows = []
        for page, ref in sorted(refs.items()):
            new = pages.get(page)
            if new is None:
                rows.append((name, page, ref_kind, "missing", "", ""))
            else:
                rows.append((name, page, ref_kind, *score(ref, new)))
    return (
        name,
        cls,
        "" if code is None else str(code),
        len(pages),
        ref_kind,
        first_err,
    ), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pdfium", type=pathlib.Path, required=True, help="pdfium source checkout"
    )
    ap.add_argument("--out", type=pathlib.Path, required=True, help="output directory")
    ap.add_argument(
        "--rrocket", type=pathlib.Path, default=REPO / "target/release/rrocket"
    )
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=int, default=60, help="seconds per rrocket run")
    ap.add_argument(
        "--pdftoppm",
        action="store_true",
        help="score documents without an expected image against pdftoppm",
    )
    ap.add_argument(
        "--match", default="", help="only documents whose path contains TEXT"
    )
    args = ap.parse_args()

    args.resources = args.pdfium / "testing" / "resources"
    if not args.resources.is_dir():
        sys.exit(f"not a pdfium checkout: {args.pdfium}")
    if not args.rrocket.is_file():
        sys.exit(
            f"rrocket not found at {args.rrocket}; cargo build --release -p rasterrocket-cli"
        )
    if args.pdftoppm and shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found")
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)

    docs = [
        d for d in expand_templates(args.resources, args.out) if args.match in str(d[1])
    ]
    results, pages = [], []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for result, rows in pool.map(lambda d: process(d, args, work), docs):
            results.append(result)
            pages.extend(rows)
    shutil.rmtree(work, ignore_errors=True)

    with open(args.out / "results.tsv", "w") as f:
        f.write("document\tclass\texit\tpages\treference\tfirst_stderr_line\n")
        f.writelines("\t".join(str(v) for v in r) + "\n" for r in sorted(results))
    with open(args.out / "pages.tsv", "w") as f:
        f.write(
            "document\tpage\treference\tresult\trmse_or_ref_size\ttile_or_new_size\n"
        )
        f.writelines("\t".join(str(v) for v in r) + "\n" for r in sorted(pages))

    counts = {}
    for r in results:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print(
        f"{len(results)} documents: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    for r in sorted(results):
        if r[1] in ("panic", "abort", "crash", "hang"):
            print(f"  {r[1]:6} {r[0]}  {r[5]}")
    scored = [p for p in pages if p[3] == "scored"]
    sizes = [p for p in pages if p[3] == "size"]
    missing = [p for p in pages if p[3] == "missing"]
    print(
        f"{len(pages)} reference pages: {len(scored)} scored, {len(sizes)} size "
        f"mismatches, {len(missing)} missing from rrocket's output"
    )
    print("largest tile deltas:")
    for p in sorted(scored, key=lambda p: -float(p[5]))[:15]:
        print(
            f"  {float(p[5]):7.2f}  rmse {float(p[4]):6.2f}  {p[2]:8}  {p[0]} p{p[1]}"
        )
    print(f"tables: {args.out / 'results.tsv'}, {args.out / 'pages.tsv'}")


if __name__ == "__main__":
    main()
