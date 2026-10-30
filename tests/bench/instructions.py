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
import shutil
import subprocess
import sys
import tempfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pdfgen import one_page, stream


def font_file(path):
    return stream(b"/Length1 %d" % path.stat().st_size, path.read_bytes())


def text_page():
    """60 lines of hinted TrueType at sizes 6-35 pt and box-glyph text."""
    lines = []
    for i in range(60):
        size = 6 + i % 30
        font = b"/F1" if i % 2 == 0 else b"/F2"
        text = (
            b"AaBb AbBa BaAb " * 4
            if font == b"/F1"
            else b"The quick brown fox 0123456789"
        )
        lines.append(
            b"BT %s %d Tf 20 %d Td (%s) Tj ET" % (font, size, 780 - i * 12, text)
        )
    widths = b"[" + b" ".join(b"1000" for _ in range(95)) + b"]"
    return one_page(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        [
            stream(b"", b"\n".join(lines)),
            b"<< /Type /Font /Subtype /TrueType /BaseFont /Tinos /FirstChar 32 "
            b"/LastChar 126 /Widths " + widths + b" /FontDescriptor 7 0 R >>",
            b"<< /Type /Font /Subtype /TrueType /BaseFont /Ahem /FirstChar 32 "
            b"/LastChar 126 /Widths " + widths + b" /FontDescriptor 8 0 R >>",
            (
                b"<< /Type /FontDescriptor /FontName /Tinos /Flags 32 /FontBBox [0 -200 1000 800] "
                b"/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 "
                b"/FontFile2 9 0 R >>"
            ),
            (
                b"<< /Type /FontDescriptor /FontName /Ahem /Flags 32 /FontBBox [0 -200 1000 800] "
                b"/ItalicAngle 0 /Ascent 800 /Descent -200 /CapHeight 700 /StemV 80 "
                b"/FontFile2 10 0 R >>"
            ),
            font_file(ROOT / "crates/font_test_data/test_data/ttf/tinos_subset.ttf"),
            font_file(ROOT / "crates/font_test_data/test_data/ttf/ahem.ttf"),
        ],
    )


def vector_page():
    """Curves, dashed strokes, axial and radial shadings, and a
    transparency group with a blend mode."""
    ops = []
    for i in range(400):
        x, y = 20 + (i * 37) % 560, 20 + (i * 53) % 740
        ops.append(
            b"%.3f %.3f %.3f rg %d %d m %d %d %d %d %d %d c h f"
            % (
                (i % 7) / 7,
                (i % 5) / 5,
                (i % 3) / 3,
                x,
                y,
                x + 40,
                y + 60,
                x + 80,
                y - 30,
                x + 20,
                y + 10,
            )
        )
    ops.append(b"[6 3] 0 d 2 w 0 0 1 RG")
    for i in range(100):
        ops.append(b"%d 20 m %d 770 l S" % (10 + i * 6, 600 - i * 6))
    ops.append(b"q 50 50 200 200 re W n /Sh1 sh Q q 350 400 200 200 re W n /Sh2 sh Q")
    ops.append(b"q /GS1 gs /Fm1 Do Q")
    return one_page(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Shading << /Sh1 5 0 R /Sh2 6 0 R >> /ExtGState << /GS1 7 0 R >> "
        b"/XObject << /Fm1 8 0 R >> >> >>",
        [
            stream(b"", b"\n".join(ops)),
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


def image_page():
    """DCT and JPX images drawn twice each, and a 512 x 512 Flate RGB image
    with a soft mask."""
    jpeg = (ROOT / "tests/fixtures/jpeg/colour_128x128_420.jpg").read_bytes()
    jp2 = (
        ROOT / "crates/pdf_interp/src/resources/image/testdata/rgba8.jp2"
    ).read_bytes()
    rgb = bytes((x ^ y) & 0xFF for y in range(512) for x in range(512 * 3))
    alpha = bytes((x + y) // 4 & 0xFF for y in range(512) for x in range(512))
    return one_page(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /XObject << /Im1 5 0 R /Im2 6 0 R /Im3 7 0 R >> >> >>",
        [
            stream(
                b"",
                b"q 280 0 0 280 20 480 cm /Im1 Do Q q 140 0 0 140 320 480 cm /Im1 Do Q "
                b"q 200 0 0 200 20 240 cm /Im2 Do Q q 100 0 0 100 320 240 cm /Im2 Do Q "
                b"q 220 0 0 220 340 20 cm /Im3 Do Q",
            ),
            stream(
                b"/Type /XObject /Subtype /Image /Width 128 /Height 128 /ColorSpace /DeviceRGB "
                b"/BitsPerComponent 8 /Filter /DCTDecode",
                jpeg,
            ),
            stream(
                b"/Type /XObject /Subtype /Image /Width 8 /Height 8 /Filter /JPXDecode",
                jp2,
            ),
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


def cmyk_page():
    """A grid of DeviceCMYK fills."""
    ops = [
        b"%.2f %.2f %.2f %.2f k %d %d 30 30 re f"
        % (c / 19, r / 25, (c + r) % 7 / 7, 0.1, 10 + c * 30, 10 + r * 30)
        for r in range(26)
        for c in range(20)
    ]
    return one_page(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        [stream(b"", b"\n".join(ops))],
    )


def synthetic(directory):
    paths = []
    for name, build in (
        ("text", text_page),
        ("vector", vector_page),
        ("image", image_page),
        ("cmyk", cmyk_page),
    ):
        path = directory / f"synthetic-{name}.pdf"
        path.write_bytes(build())
        paths.append(path)
    return paths


def perf_works():
    if shutil.which("perf") is None:
        return False
    r = subprocess.run(
        ["perf", "stat", "-x,", "-e", "instructions:u", "true"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        r.returncode == 0
        and re.match(r"^\d+,", r.stderr.strip().splitlines()[-1] or "") is not None
    )


def count(tool, binary, pdf, args, out):
    cmd = [
        str(binary),
        "--no-ram",
        "--threads",
        "1",
        "--backend",
        "cpu",
        *args,
        str(pdf),
        str(out / "p"),
    ]
    if tool == "perf":
        r = subprocess.run(
            ["perf", "stat", "-x,", "-e", "instructions:u", *cmd],
            check=False,
            capture_output=True,
            text=True,
        )
        m = re.search(r"^(\d+),[^,]*,instructions", r.stderr, re.MULTILINE)
    else:
        r = subprocess.run(
            ["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null", *cmd],
            check=False,
            capture_output=True,
            text=True,
        )
        m = re.search(r"Collected : (\d+)", r.stderr)
    for f in out.iterdir():
        f.unlink()
    if r.returncode != 0 or m is None:
        return None
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", type=pathlib.Path, help="baseline rrocket binary")
    ap.add_argument("head", type=pathlib.Path, help="rrocket binary under test")
    ap.add_argument("--pdf", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--corpus-dir", type=pathlib.Path)
    ap.add_argument("--dpi", type=int, default=72)
    ap.add_argument(
        "--pages", type=int, default=5, help="first N pages of each document"
    )
    ap.add_argument("--tool", choices=("perf", "callgrind"))
    ap.add_argument("--fail-above", type=float, help="regression limit, percent")
    ap.add_argument(
        "--markdown", type=pathlib.Path, help="also write the table as Markdown"
    )
    args = ap.parse_args()

    tool = args.tool or ("perf" if perf_works() else "callgrind")
    if tool == "callgrind" and shutil.which("valgrind") is None:
        sys.exit("neither a working perf nor valgrind is available")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        docs = list(args.pdf)
        if args.corpus_dir:
            docs += sorted(args.corpus_dir.rglob("*.pdf"))
        if not docs:
            docs = synthetic(tmp)
        out = tmp / "out"
        out.mkdir()
        render = ["-r", str(args.dpi), "-l", str(args.pages)]

        rows = []
        for pdf in docs:
            base = count(tool, args.base, pdf, render, out)
            head = count(tool, args.head, pdf, render, out)
            rows.append((pdf.name, base, head))

    def pct(b, h):
        return (h - b) * 100.0 / b if b and h is not None else None

    ok = [(b, h) for _, b, h in rows if b and h is not None]
    total = (sum(b for b, _ in ok), sum(h for _, h in ok))
    lines = [
        f"Instructions ({tool}, {args.dpi} dpi, first {args.pages} pages, 1 thread)",
        "",
        "| document | base | head | change |",
        "|---|---:|---:|---:|",
    ]
    for name, b, h in rows:
        p = pct(b, h)
        lines.append(
            f"| {name} | {b if b is not None else 'failed'} | {h if h is not None else 'failed'} | "
            f"{f'{p:+.2f} %' if p is not None else '-'} |"
        )
    tp = pct(*total)
    lines.append(
        f"| **total** | {total[0]} | {total[1]} | {f'{tp:+.2f} %' if tp is not None else '-'} |"
    )
    print("\n".join(lines))
    if args.markdown:
        args.markdown.write_text("\n".join(lines) + "\n")

    if args.fail_above is not None:
        worst = [p for p in (pct(b, h) for _, b, h in rows) if p is not None] + (
            [tp] if tp is not None else []
        )
        failed = any(b is None or h is None for _, b, h in rows)
        if failed or any(p > args.fail_above for p in worst):
            sys.exit(1)


if __name__ == "__main__":
    main()
