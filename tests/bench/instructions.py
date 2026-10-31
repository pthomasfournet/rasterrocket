#!/usr/bin/env python3
"""Compare the instruction counts of two rrocket builds, document by document.

    python3 -I tests/bench/instructions.py BASE_BIN HEAD_BIN
        [--pdf FILE ...] [--corpus-dir DIR] [--dpi N] [--pages N]
        [--tool perf|callgrind] [--fail-above PCT] [--markdown FILE]

Wall-clock time on shared or virtual machines varies by several percent
between runs; the number of user-space instructions a single-threaded
render retires barely does (perf: within 0.3 % on a 27 M-instruction page,
0.003 % at 400 M; callgrind: within 0.01 %).  Each document is rendered once per
build with `--threads 1 --backend cpu --no-ram`, and the table reports the
change per document and in total.

Documents: every --pdf, every *.pdf under --corpus-dir, or, when neither is
given, a synthetic set generated from files committed to this repository
(no corpus needed): hinted TrueType and box-glyph text, vector paths with
dashes and shadings, a transparency group, DCT, JPX and Flate images with a
soft mask, and CMYK fills.

--tool perf counts `instructions:u` with perf stat (needs
perf_event_paranoid <= 2 and hardware counters); callgrind runs under
valgrind and needs a build without AVX-512 code paths in use (build with
RUSTFLAGS="-C target-cpu=generic").  The default is perf when it works,
else callgrind.

Exit status: 0, or 1 when the total or any document regresses by more than
--fail-above percent (default: never fail).
"""

import argparse
import pathlib
import re
import sys
import tempfile
import zlib
from collections.abc import Callable, Sequence
from typing import NamedTuple

# The shared helpers live in scripts/; `python3 -I` puts no script directory on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from pdfgen import one_page, stream
from runtool import describe_exit, executable, find, last_line, positive_int, require, run

ROOT = pathlib.Path(__file__).resolve().parents[2]
PERF_EVENT = "instructions:u"


def font_file(path: pathlib.Path) -> bytes:
    """Return a FontFile2 stream object holding the TrueType program at `path`."""
    data = path.read_bytes()
    return stream(b"/Length1 %d" % len(data), data)


def truetype_font(name: bytes, descriptor: int, widths: bytes) -> bytes:
    """Return a simple TrueType font dictionary over codes 32..126."""
    return (
        b"<< /Type /Font /Subtype /TrueType /BaseFont /%s /FirstChar 32 /LastChar 126 "
        b"/Widths %s /FontDescriptor %d 0 R >>" % (name, widths, descriptor)
    )


def font_descriptor(name: bytes, font_file_ref: int) -> bytes:
    """Return a font descriptor for `name` embedding the program at `font_file_ref`."""
    return (
        b"<< /Type /FontDescriptor /FontName /%s /Flags 32 /FontBBox [0 -200 1000 800] "
        b"/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 "
        b"/FontFile2 %d 0 R >>" % (name, font_file_ref)
    )


def text_page() -> bytes:
    """Return 60 lines of hinted TrueType at sizes 6-35 pt and box-glyph text."""
    lines = []
    for i in range(60):
        size = 6 + i % 30
        font, text = (
            (b"/F1", b"AaBb AbBa BaAb " * 4)
            if i % 2 == 0
            else (b"/F2", b"The quick brown fox 0123456789")
        )
        lines.append(b"BT %s %d Tf 20 %d Td (%s) Tj ET" % (font, size, 780 - i * 12, text))
    widths = b"[" + b" ".join([b"1000"] * 95) + b"]"
    ttf = ROOT / "crates/font_test_data/test_data/ttf"
    return one_page(
        b"\n".join(lines),
        b"/Font << /F1 5 0 R /F2 6 0 R >>",
        [
            truetype_font(b"Tinos", 7, widths),
            truetype_font(b"Ahem", 8, widths),
            font_descriptor(b"Tinos", 9),
            font_descriptor(b"Ahem", 10),
            font_file(ttf / "tinos_subset.ttf"),
            font_file(ttf / "ahem.ttf"),
        ],
    )


def vector_page() -> bytes:
    """Return curves, dashed strokes, shadings and a blended transparency group."""
    ops = []
    for i in range(400):
        x, y = 20 + (i * 37) % 560, 20 + (i * 53) % 740
        rgb = ((i % 7) / 7, (i % 5) / 5, (i % 3) / 3)
        curve = (x, y, x + 40, y + 60, x + 80, y - 30, x + 20, y + 10)
        ops.append(b"%.3f %.3f %.3f rg %d %d m %d %d %d %d %d %d c h f" % (*rgb, *curve))
    ops.append(b"[6 3] 0 d 2 w 0 0 1 RG")
    ops += [b"%d 20 m %d 770 l S" % (10 + i * 6, 600 - i * 6) for i in range(100)]
    ops.append(b"q 50 50 200 200 re W n /Sh1 sh Q q 350 400 200 200 re W n /Sh2 sh Q")
    ops.append(b"q /GS1 gs /Fm1 Do Q")
    return one_page(
        b"\n".join(ops),
        b"/Shading << /Sh1 5 0 R /Sh2 6 0 R >> /ExtGState << /GS1 7 0 R >> "
        b"/XObject << /Fm1 8 0 R >>",
        [
            (
                b"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [50 50 250 250] "
                b"/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>"
            ),
            (
                b"<< /ShadingType 3 /ColorSpace /DeviceRGB /Coords [450 500 0 450 500 100] "
                b"/Function << /FunctionType 2 /Domain [0 1] /C0 [1 1 0] /C1 [0 1 0] /N 2 >> "
                b"/Extend [true true] >>"
            ),
            b"<< /Type /ExtGState /BM /Multiply /ca 0.6 /CA 0.6 >>",
            stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] "
                b"/Group << /S /Transparency /I true /K false >>",
                b"0.2 0.8 0.4 rg 100 100 400 600 re f 0.9 0.1 0.1 rg 200 200 300 300 re f",
            ),
        ],
    )


def image_page() -> bytes:
    """Return DCT and JPX images drawn twice each and a soft-masked Flate image.

    The Flate image is 512 x 512 RGB.
    """
    jpeg = (ROOT / "tests/fixtures/jpeg/colour_128x128_420.jpg").read_bytes()
    jp2 = (ROOT / "crates/pdf_interp/src/resources/image/testdata/rgba8.jp2").read_bytes()
    rgb = bytes((x ^ y) & 0xFF for y in range(512) for x in range(512 * 3))
    alpha = bytes((x + y) // 4 & 0xFF for y in range(512) for x in range(512))
    return one_page(
        b"q 280 0 0 280 20 480 cm /Im1 Do Q q 140 0 0 140 320 480 cm /Im1 Do Q "
        b"q 200 0 0 200 20 240 cm /Im2 Do Q q 100 0 0 100 320 240 cm /Im2 Do Q "
        b"q 220 0 0 220 340 20 cm /Im3 Do Q",
        b"/XObject << /Im1 5 0 R /Im2 6 0 R /Im3 7 0 R >>",
        [
            stream(
                b"/Type /XObject /Subtype /Image /Width 128 /Height 128 /ColorSpace /DeviceRGB "
                b"/BitsPerComponent 8 /Filter /DCTDecode",
                jpeg,
            ),
            stream(b"/Type /XObject /Subtype /Image /Width 8 /Height 8 /Filter /JPXDecode", jp2),
            stream(
                b"/Type /XObject /Subtype /Image /Width 512 /Height 512 /ColorSpace /DeviceRGB "
                b"/BitsPerComponent 8 /Filter /FlateDecode /SMask 8 0 R",
                zlib.compress(rgb),
            ),
            stream(
                b"/Type /XObject /Subtype /Image /Width 512 /Height 512 /ColorSpace /DeviceGray "
                b"/BitsPerComponent 8 /Filter /FlateDecode",
                zlib.compress(alpha),
            ),
        ],
    )


def cmyk_page() -> bytes:
    """Return a grid of DeviceCMYK fills."""
    return one_page(
        b"\n".join(
            b"%.2f %.2f %.2f %.2f k %d %d 30 30 re f"
            % (c / 19, r / 25, (c + r) % 7 / 7, 0.1, 10 + c * 30, 10 + r * 30)
            for r in range(26)
            for c in range(20)
        )
    )


SYNTHETIC: dict[str, Callable[[], bytes]] = {
    "text": text_page,
    "vector": vector_page,
    "image": image_page,
    "cmyk": cmyk_page,
}


def synthetic(directory: pathlib.Path) -> list[pathlib.Path]:
    """Write the synthetic documents into `directory` and return their paths."""
    paths = []
    for name, build in SYNTHETIC.items():
        path = directory / f"synthetic-{name}.pdf"
        path.write_bytes(build())
        paths.append(path)
    return paths


def perf_works(perf: str) -> bool:
    """Return whether `perf` can count user-space instructions here."""
    r = run([perf, "stat", "-x,", "-e", PERF_EVENT, "true"])
    return r.returncode == 0 and re.match(r"^\d+,", last_line(r.stderr)) is not None


class InstructionCounter(NamedTuple):
    """An instruction counter: the wrapper command and the pattern of its total."""

    name: str
    prefix: list[str]
    total: re.Pattern[str]


def perf_counter(perf: str) -> InstructionCounter:
    """Return the perf counter run through the binary at `perf`."""
    return InstructionCounter(
        "perf",
        [perf, "stat", "-x,", "-e", PERF_EVENT],
        re.compile(r"^(\d+),[^,]*,instructions", re.MULTILINE),
    )


def callgrind_counter(valgrind: str) -> InstructionCounter:
    """Return the callgrind counter run through the binary at `valgrind`."""
    return InstructionCounter(
        "callgrind",
        [valgrind, "--tool=callgrind", "--callgrind-out-file=/dev/null"],
        re.compile(r"Collected : (\d+)"),
    )


def choose_counter(requested: str | None) -> InstructionCounter:
    """Return the requested counter, else perf when it works, else callgrind."""
    if requested == "perf":
        return perf_counter(require("perf"))
    if requested == "callgrind":
        return callgrind_counter(require("valgrind"))
    perf = find("perf")
    if perf is not None and perf_works(perf):
        return perf_counter(perf)
    valgrind = find("valgrind")
    if valgrind is None:
        sys.exit("neither a working perf nor valgrind is available")
    return callgrind_counter(valgrind)


def count(
    counter: InstructionCounter,
    binary: pathlib.Path,
    pdf: pathlib.Path,
    render: Sequence[str],
    out: pathlib.Path,
) -> int | None:
    """Return the instructions one render of `pdf` retires, or `None` when it fails.

    A failure is reported on stderr with the last line the render printed.
    """
    cmd = [str(binary), "--no-ram", "--threads", "1", "--backend", "cpu", *render]
    r = run([*counter.prefix, *cmd, str(pdf), str(out / "p")])
    for f in out.iterdir():
        f.unlink()
    m = counter.total.search(r.stderr)
    if r.returncode == 0 and m is not None:
        return int(m.group(1))
    # The counter appends its own report; the diagnostic is the line before it.
    render_err = "\n".join(line for line in r.stderr.splitlines() if not counter.total.search(line))
    print(
        f"{binary} on {pdf.name}: {describe_exit(r.returncode)}: "
        f"{last_line(render_err, 'no output')}",
        file=sys.stderr,
    )
    return None


def change(base: int | None, head: int | None) -> float | None:
    """Return the change from `base` to `head` in percent; `None` when undefined."""
    if not base or head is None:
        return None
    return (head - base) * 100.0 / base


def percent(p: float | None) -> str:
    """Format a change for the table."""
    return "-" if p is None else f"{p:+.2f} %"


def table(
    rows: list[tuple[str, int | None, int | None]], heading: str
) -> tuple[list[str], float | None]:
    """Return the Markdown table lines and the total change."""
    ok = [(b, h) for _, b, h in rows if b and h is not None]
    total_base, total_head = sum(b for b, _ in ok), sum(h for _, h in ok)
    total = change(total_base, total_head)
    lines = [heading, "", "| document | base | head | change |", "|---|---:|---:|---:|"]
    lines += [
        f"| {name} | {'failed' if b is None else b} | {'failed' if h is None else h} | "
        f"{percent(change(b, h))} |"
        for name, b, h in rows
    ]
    lines.append(f"| **total** | {total_base} | {total_head} | {percent(total)} |")
    return lines, total


def parse_args() -> argparse.Namespace:
    """Return the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", type=executable, help="baseline rrocket binary")
    ap.add_argument("head", type=executable, help="rrocket binary under test")
    ap.add_argument("--pdf", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--corpus-dir", type=pathlib.Path)
    ap.add_argument("--dpi", type=positive_int, default=72)
    ap.add_argument("--pages", type=positive_int, default=5, help="first N pages of each document")
    ap.add_argument("--tool", choices=("perf", "callgrind"))
    ap.add_argument("--fail-above", type=float, help="regression limit, percent")
    ap.add_argument("--markdown", type=pathlib.Path, help="also write the table as Markdown")
    args = ap.parse_args()
    if args.corpus_dir and not args.corpus_dir.is_dir():
        ap.error(f"not a directory: {args.corpus_dir}")
    missing = [str(p) for p in args.pdf if not p.is_file()]
    if missing:
        ap.error(f"no such PDF: {', '.join(missing)}")
    return args


def main() -> None:
    """Count both builds over every document and print the comparison."""
    args = parse_args()
    counter = choose_counter(args.tool)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = pathlib.Path(tmp_name)
        docs = list(args.pdf)
        if args.corpus_dir:
            docs += sorted(args.corpus_dir.rglob("*.pdf"))
        if not docs:
            docs = synthetic(tmp)
        out = tmp / "out"
        out.mkdir()
        render = ["-r", str(args.dpi), "-l", str(args.pages)]
        rows = [
            (
                pdf.name,
                count(counter, args.base, pdf, render, out),
                count(counter, args.head, pdf, render, out),
            )
            for pdf in docs
        ]

    lines, total = table(
        rows,
        f"Instructions ({counter.name}, {args.dpi} dpi, first {args.pages} pages, 1 thread)",
    )
    print("\n".join(lines))
    if args.markdown:
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.fail_above is not None:
        failed = any(b is None or h is None for _, b, h in rows)
        changes = [change(b, h) for _, b, h in rows] + [total]
        if failed or any(p is not None and p > args.fail_above for p in changes):
            sys.exit(1)


if __name__ == "__main__":
    main()
