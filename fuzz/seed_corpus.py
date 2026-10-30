#!/usr/bin/env python3
"""Build seed corpora for the fuzz targets.

    python3 -I fuzz/seed_corpus.py OUT_DIR [--pdfium PDFIUM_SRC] [--extra-pdfs DIR]

Writes OUT_DIR/<target>/ for every target in fuzz/fuzz_targets, from:

- files committed to this repository (JPEG / JPEG 2000 test images, test
  fonts, hand-written content streams, CMaps and functions);
- the thirteen encrypted variants of the encryption integration suite,
  produced with qpdf (skipped with a note when qpdf is absent);
- with --pdfium, the PDFs under PDFIUM_SRC/testing/resources (BSD-3-Clause,
  read in place, never copied into the repository);
- with --extra-pdfs, every PDF under DIR (e.g. a local document archive).

Each seed carries the selector bytes its target's layout expects (see the
header comment of each target).  Run `cargo fuzz run <target> OUT_DIR/<target>`
or pass the directory as an extra corpus.
"""

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pdfgen import assemble_pdf, stream

MAX_SEED = 1 << 20

# Encryption variants of crates/pdf_raster/tests/encryption.rs:
# (encrypt options, general options, user password).
VARIANTS = [
    (["--bits=40"], [], "user"),
    (["--bits=128"], [], "user"),
    (["--bits=128", "--force-V4"], [], "user"),
    (["--bits=128", "--use-aes=y"], [], "user"),
    (["--bits=128", "--use-aes=y", "--cleartext-metadata"], [], "user"),
    (["--bits=128", "--use-aes=y"], ["--object-streams=generate"], "user"),
    (["--bits=256", "--force-R5"], [], "user"),
    (["--bits=256"], [], "user"),
    (["--bits=256", "--cleartext-metadata"], [], "user"),
    (["--bits=256"], ["--object-streams=generate"], "user"),
    (["--bits=128"], [], ""),
    (["--bits=128", "--use-aes=y"], [], ""),
    (["--bits=256"], [], ""),
]

CONTENT_SEEDS = [
    b"0.2 0.4 0.8 rg 10 10 80 60 re f 0 g 1 w 5 5 m 95 95 l S",
    b"BT /F1 12 Tf 10 50 Td (Hello) Tj 0 -14 Td [(W) 120 (orld)] TJ ET",
    b"q /G1 gs 1 0 0 rg 20 20 60 60 re f Q q 10 10 80 80 re W n /S1 sh Q",
    b"BT 3 Tr /F1 30 Tf 10 30 Td (clip) Tj 7 Tr ET 0 0 100 100 re f",
    b"q 1 0 0 1 50 50 cm 0 0 m 40 0 40 40 0 40 c h 0.5 g B* Q",
    b"[4 2] 0 d 2 J 1 j 3 w 10 90 m 90 10 l S /DeviceCMYK cs 0 1 1 0 sc 30 30 40 40 re f",
    b"BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\xff\xff\x00 EI",
]

CMAP_SEEDS = [
    (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CMapName /Test def 1 begincodespacerange <00> <FF> endcodespacerange\n"
        b"2 beginbfchar <01> <0041> <02> <0042> endbfchar\n"
        b"1 beginbfrange <10> <20> <0061> endbfrange\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end\n"
    ),
    (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 6 >> def\n"
        b"/WMode 1 def 2 begincodespacerange <00> <80> <8140> <FCFC> endcodespacerange\n"
        b"1 begincidrange <8140> <817E> 633 endcidrange\n"
        b"1 begincidchar <20> 1 endcidchar\n"
        b"/Identity-H usecmap endcmap\n"
    ),
]

FUNCTION_SEEDS = [
    (b"\x00\x00\x00", b"{ dup 0.5 mul exch 1 exch sub }"),
    (b"\x00\x00\x00", b"{ 2 copy lt { exch } if pop dup sin abs exch cos abs }"),
    (b"\x01\x00\x00", b"{ 2 copy add 2 div 3 1 roll mul sqrt }"),
    (b"\x02\x03\x03", bytes(range(0, 255, 21))),
    (b"\x03\x03\x01", bytes(range(48))),
]


def write(out, target, data):
    """Store `data` as a seed of `target` under its content hash."""
    if not data or len(data) > MAX_SEED:
        return
    d = out / target
    d.mkdir(parents=True, exist_ok=True)
    (d / hashlib.sha1(data).hexdigest()).write_bytes(data)


def plaintext_pdf():
    """The plaintext page of the encryption integration suite."""
    content = (
        b"0.2 0.4 0.8 rg 40 40 120 60 re f\n"
        b"BT /F1 28 Tf 30 150 Td (Encrypted text) Tj ET\n"
        b"0 g 1 w 20 20 m 180 190 l S\n"
    )
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
                b"/Resources << /Font << /F1 5 0 R >> >> /Annots [6 0 R] >>"
            ),
            stream(b"", content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            (
                b"<< /Type /Annot /Subtype /FreeText /Rect [100 100 190 130] "
                b"/Contents (annot string) /DA (/Helv 10 Tf 0 g) /F 4 >>"
            ),
        ]
    )


def encrypted_seeds(out):
    if shutil.which("qpdf") is None:
        print(
            "qpdf not found: no fuzz_decrypt seeds from the encryption suite",
            file=sys.stderr,
        )
        return
    with tempfile.TemporaryDirectory() as tmp:
        plain = pathlib.Path(tmp) / "plain.pdf"
        plain.write_bytes(plaintext_pdf())
        for i, (enc_args, gen_args, user) in enumerate(VARIANTS):
            enc = pathlib.Path(tmp) / f"{i}.pdf"
            subprocess.run(
                [
                    "qpdf",
                    "--allow-weak-crypto",
                    "--encrypt",
                    f"--user-password={user}",
                    "--owner-password=owner",
                    *enc_args,
                    "--",
                    *gen_args,
                    str(plain),
                    str(enc),
                ],
                check=True,
            )
            data = enc.read_bytes()
            write(out, "fuzz_decrypt", data)
            write(out, "fuzz_document", data)
            write(out, "fuzz_parse", data)


def repo_seeds(out):
    # Images: [filter, colour space, bpc, width-1, height-1, flags, icc_len(2)].
    for jpg in sorted((ROOT / "tests/fixtures/jpeg").glob("*.jpg")):
        data = jpg.read_bytes()
        write(out, "fuzz_image", bytes([1, 1, 3, 31, 31, 0, 0, 0]) + data)
        write(out, "fuzz_image", bytes([1, 0, 3, 63, 63, 0, 0, 0]) + data)
    for jp2 in sorted(
        (ROOT / "crates/pdf_interp/src/resources/image/testdata").glob("*.j*2*")
    ):
        data = jp2.read_bytes()
        for space in (0, 1):
            write(out, "fuzz_image", bytes([2, space, 3, 15, 15, 0, 0, 0]) + data)
            write(out, "fuzz_image", bytes([2, space, 3, 15, 15, 0x18, 0, 0]) + data)
    raw = bytes(range(256)) * 3
    for space in range(8):
        write(out, "fuzz_image", bytes([0, space, 3, 15, 15, 0, 0, 0]) + raw)
        write(
            out,
            "fuzz_image",
            bytes([3, space, 3, 15, 15, 2, 0, 0]) + zlib.compress(raw),
        )
    write(out, "fuzz_image", bytes([0, 0, 0, 15, 15, 1, 0, 0]) + raw[:32])

    # Fonts: [face_index, size].
    for pattern in (
        "crates/font/src/tests/data/*",
        "crates/font_test_data/test_data/ttf/*",
        "crates/font_test_data/test_data/ttc/*",
        "crates/font_test_data/test_data/type1/*",
    ):
        for f in sorted(ROOT.glob(pattern)):
            if f.suffix.lower() in (
                ".ttf",
                ".otf",
                ".ttc",
                ".cid",
                ".pfa",
                ".pfb",
                ".cff",
            ):
                write(out, "fuzz_font", bytes([0, 24]) + f.read_bytes())

    # Filter chains: [count, f0, f1, f2, predictor, colors, bpc, columns, early].
    text = b"BT /F1 12 Tf (filter chain seed) Tj ET\n" * 8
    write(out, "fuzz_filters", bytes([0, 0, 0, 0, 0, 0, 3, 7, 1]) + zlib.compress(text))
    write(
        out,
        "fuzz_filters",
        bytes([1, 2, 0, 0, 0, 0, 3, 7, 1]) + zlib.compress(text).hex().encode(),
    )
    rows = b"".join(b"\x02" + bytes(range(i, i + 8)) for i in range(16))
    write(out, "fuzz_filters", bytes([0, 0, 0, 0, 4, 0, 3, 7, 1]) + zlib.compress(rows))
    write(out, "fuzz_filters", bytes([0, 4, 0, 0, 0, 0, 3, 7, 1]) + b"\x02abc\xfdz\x80")

    for c in CONTENT_SEEDS:
        write(out, "fuzz_content", c)
    for c in CMAP_SEEDS:
        write(out, "fuzz_cmap", c)
    for sel, body in FUNCTION_SEEDS:
        write(out, "fuzz_function", sel + body)
    write(out, "fuzz_document", plaintext_pdf())
    write(out, "fuzz_parse", plaintext_pdf())


def pdf_seeds(out, root):
    n = 0
    for pdf in sorted(root.rglob("*.pdf")):
        if pdf.is_file() and pdf.stat().st_size <= MAX_SEED:
            data = pdf.read_bytes()
            write(out, "fuzz_document", data)
            write(out, "fuzz_parse", data)
            if b"/Encrypt" in data:
                write(out, "fuzz_decrypt", data)
            n += 1
    print(f"{root}: {n} PDF seeds", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--pdfium", type=pathlib.Path, help="pdfium source checkout")
    ap.add_argument("--extra-pdfs", type=pathlib.Path, help="directory of PDFs to add")
    args = ap.parse_args()
    repo_seeds(args.out)
    encrypted_seeds(args.out)
    if args.pdfium:
        pdf_seeds(args.out, args.pdfium / "testing" / "resources")
    if args.extra_pdfs:
        pdf_seeds(args.out, args.extra_pdfs)
    for d in sorted(args.out.iterdir()):
        print(f"{d.name}: {sum(1 for _ in d.iterdir())} seeds")


if __name__ == "__main__":
    main()
