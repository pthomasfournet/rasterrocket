#!/usr/bin/env python3
"""Build seed corpora for the fuzz targets.

    python3 -I fuzz/seed_corpus.py OUT_DIR [--pdfium PDFIUM_SRC] [--extra-pdfs DIR]

Writes OUT_DIR/<target>/ for every target in fuzz/fuzz_targets, from:

- files committed to this repository (JPEG / JPEG 2000 test images, ICC
  profiles, comic archives and page images, test fonts, hand-written
  content streams, CMaps and functions), synthetic camera captures and
  synthetic PDFs whose parsing cost once grew faster than their size;
- the thirteen encrypted variants of the encryption integration suite,
  produced with qpdf (skipped with a note when qpdf is absent);
- with --pdfium, the PDFs under PDFIUM_SRC/testing/resources (BSD-3-Clause,
  read in place, never copied into the repository);
- with --extra-pdfs, every PDF under DIR (e.g. a local document archive).

Each seed carries the selector bytes its target's layout expects (see the
header comment of each target).  Run `cargo fuzz run <target> OUT_DIR/<target>`
or pass the directory as an extra corpus.  A target binary run directly,
not through `cargo fuzz run`, needs `ASAN_OPTIONS=detect_odr_violation=0`:
libdeflate's tables are linked twice, which AddressSanitizer reports at
start-up as an ODR violation.
"""

import argparse
import hashlib
import pathlib
import sys
import tempfile
import zlib

# The shared helpers live in scripts/; `python3 -I` puts no script directory on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from pdfgen import one_page
from runtool import find, run_checked

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_SEED = 1 << 20
FONT_SUFFIXES = frozenset((".ttf", ".otf", ".ttc", ".cid", ".pfa", ".pfb", ".cff"))

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


def write(out: pathlib.Path, target: str, data: bytes) -> None:
    """Store `data` as a seed of `target` under its content hash.

    Empty inputs and inputs above `MAX_SEED` bytes are dropped.
    """
    if not data or len(data) > MAX_SEED:
        return
    d = out / target
    d.mkdir(parents=True, exist_ok=True)
    (d / hashlib.sha256(data).hexdigest()).write_bytes(data)


def plaintext_pdf() -> bytes:
    """Return the plaintext page of the encryption integration suite."""
    return one_page(
        b"0.2 0.4 0.8 rg 40 40 120 60 re f\n"
        b"BT /F1 28 Tf 30 150 Td (Encrypted text) Tj ET\n"
        b"0 g 1 w 20 20 m 180 190 l S\n",
        b"/Font << /F1 5 0 R >>",
        [
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            (
                b"<< /Type /Annot /Subtype /FreeText /Rect [100 100 190 130] "
                b"/Contents (annot string) /DA (/Helv 10 Tf 0 g) /F 4 >>"
            ),
        ],
        media_box=b"0 0 200 200",
        page_entries=b"/Annots [6 0 R]",
    )


def encrypted_seeds(out: pathlib.Path, plaintext: bytes) -> None:
    """Write the encrypted variants of the PDF `plaintext`, produced with qpdf."""
    qpdf = find("qpdf")
    if qpdf is None:
        print("qpdf not found: no fuzz_decrypt seeds from the encryption suite", file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as tmp:
        plain = pathlib.Path(tmp) / "plain.pdf"
        plain.write_bytes(plaintext)
        for i, (enc_args, gen_args, user) in enumerate(VARIANTS):
            enc = pathlib.Path(tmp) / f"{i}.pdf"
            run_checked(
                [
                    qpdf,
                    "--allow-weak-crypto",
                    "--encrypt",
                    f"--user-password={user}",
                    "--owner-password=owner",
                    *enc_args,
                    "--",
                    *gen_args,
                    str(plain),
                    str(enc),
                ]
            )
            write_pdf(out, enc.read_bytes())


def write_pdf(out: pathlib.Path, data: bytes) -> None:
    """Store a whole PDF as a seed of every target that takes one."""
    write(out, "fuzz_document", data)
    write(out, "fuzz_parse", data)
    if b"/Encrypt" in data:
        write(out, "fuzz_decrypt", data)


def committed(pattern: str) -> list[pathlib.Path]:
    """Return the repository files matching `pattern`; exit when there are none.

    An empty match means the layout moved, which would silently thin the
    corpus.
    """
    found = sorted(f for f in ROOT.glob(pattern) if f.is_file())
    if not found:
        sys.exit(f"no committed files match {pattern}")
    return found


def image_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_image` seeds.

    Selector: [filter, colour space, bpc, width-1, height-1, flags, icc_len(2)].
    """
    for jpg in committed("tests/fixtures/jpeg/*.jpg"):
        data = jpg.read_bytes()
        write(out, "fuzz_image", bytes([1, 1, 3, 31, 31, 0, 0, 0]) + data)
        write(out, "fuzz_image", bytes([1, 0, 3, 63, 63, 0, 0, 0]) + data)
    for jp2 in committed("crates/pdf_interp/src/resources/image/testdata/*.j*2*"):
        data = jp2.read_bytes()
        for space in (0, 1):
            write(out, "fuzz_image", bytes([2, space, 3, 15, 15, 0, 0, 0]) + data)
            write(out, "fuzz_image", bytes([2, space, 3, 15, 15, 0x18, 0, 0]) + data)
    raw = bytes(range(256)) * 3
    for space in range(8):
        write(out, "fuzz_image", bytes([0, space, 3, 15, 15, 0, 0, 0]) + raw)
        write(out, "fuzz_image", bytes([3, space, 3, 15, 15, 2, 0, 0]) + zlib.compress(raw))
    write(out, "fuzz_image", bytes([0, 0, 0, 15, 15, 1, 0, 0]) + raw[:32])


# Component selector of an ICC profile, by a word of its file name: 0, 1
# and 2 select 1, 3 and 4 components in `fuzz_icc` and `fuzz_image`.
ICC_SPACES = (("gray", 0), ("rgb", 1), ("cmyk", 2))


def icc_profiles() -> list[tuple[bytes, int]]:
    """Return the committed ICC profiles with their component selectors.

    A profile whose name names no space (the Lab and CGATS profiles) is
    given three components.
    """
    found = []
    for pattern in ("tests/fixtures/icc/*.icc", "crates/jpeg2000/assets/*.icc"):
        for f in committed(pattern):
            name = f.name.lower()
            space = next((n for key, n in ICC_SPACES if key in name), 1)
            found.append((f.read_bytes(), space))
    return found


def icc_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_icc` seeds and the ICC-based `fuzz_image` seeds.

    `fuzz_icc` selector: [space, flags]; `fuzz_image` as in `image_seeds`.
    """
    raw = bytes(range(256)) * 3
    for profile, space in icc_profiles():
        for flags in (0, 1):
            write(out, "fuzz_icc", bytes([space, flags]) + profile)
        if len(profile) < 1 << 16:
            sel = bytes([0, 3 | space << 4, 3, 15, 15, 0]) + len(profile).to_bytes(2, "big")
            write(out, "fuzz_image", sel + profile + raw)


def jpeg_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_jpeg` seeds, and `fuzz_image` seeds of the same streams.

    Selector: [mode] (output space, strictness).
    """
    files = committed("tests/fixtures/jpeg/*.jpg") + committed("crates/zune_jpeg/tests/images/**/*.jpg")
    for jpg in files:
        data = jpg.read_bytes()
        name = jpg.name.lower()
        mode = 2 if "cmyk" in name else 0 if "gray" in name else 1
        write(out, "fuzz_jpeg", bytes([mode]) + data)
        write(out, "fuzz_jpeg", bytes([mode | 4]) + data)
        write(out, "fuzz_image", bytes([1, mode, 3, 15, 15, 0, 0, 0]) + data)


def jpeg2000_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_jpeg2000` seeds, and `fuzz_image` seeds of the HTJ2K streams.

    Selector: [mode, size].
    """
    files = committed("crates/jpeg2000/tests/streams/*") + committed(
        "crates/pdf_interp/src/resources/image/testdata/*.j*2*"
    )
    for f in files:
        data = f.read_bytes()
        write(out, "fuzz_jpeg2000", bytes([0, 0]) + data)
        write(out, "fuzz_jpeg2000", bytes([8, 16]) + data)
        if f.name.startswith("ht_"):
            write(out, "fuzz_image", bytes([2, 1, 3, 15, 15, 0, 0, 0]) + data)


# Container selector of `fuzz_comic`, by archive suffix.
COMIC_CONTAINERS = {".cbz": 0, ".cb7": 1, ".cbt": 2, ".cbr": 3}


def comic_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_comic` and `fuzz_comic_page` seeds.

    Selectors: [container, flags] and [flags].
    """
    for f in committed("crates/comic/tests/data/one-page*"):
        data = f.read_bytes()
        for flags in (0, 1, 6):
            write(out, "fuzz_comic", bytes([COMIC_CONTAINERS[f.suffix], flags]) + data)
    pages = [
        f
        for pattern in ("*.avif", "*.jxl", "*.png", "*.webp", "*.tif")
        for f in committed(f"crates/comic/tests/data/{pattern}")
    ] + committed("tests/fixtures/jpeg/*.jpg")
    for f in pages:
        data = f.read_bytes()
        for flags in (0, 1, 6):
            write(out, "fuzz_comic_page", bytes([flags]) + data)


def camera_capture(width: int, height: int, spread: bool) -> bytes:
    """Return a synthetic 8-bit camera capture of `width` x `height`.

    A light page whose corners a perspective pulls inwards lies on a darker
    desk; it carries rows of dark text and, for a spread, a dark gutter down
    the middle.
    """
    corners = [(0.12, 0.08), (0.9, 0.12), (0.86, 0.92), (0.1, 0.88)]
    quad = [(x * width, y * height) for x, y in corners]

    def inside(px: float, py: float) -> bool:
        # The quad is convex, its corners clockwise on screen (y down).
        for (x0, y0), (x1, y1) in zip(quad, quad[1:] + quad[:1]):
            if (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < 0:
                return False
        return True

    samples = bytearray()
    for y in range(height):
        for x in range(width):
            if not inside(x + 0.5, y + 0.5):
                v = 40 + (x * 7 + y * 3) % 20
            elif spread and abs(x - width // 2) < 2:
                v = 60
            elif (y // 4) % 3 == 0 and (x // 3) % 4 != 3 and 0.2 < x / width < 0.8:
                v = 30
            else:
                v = 225
            samples.append(v)
    return bytes(samples)


def camera_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_camera` seeds.

    Selector: [width_lo, width_hi, stride_pad, stages, threshold].
    """
    for width, height, spread in ((96, 64, False), (160, 100, True)):
        image = camera_capture(width, height, spread)
        w = (width - 1).to_bytes(2, "little")
        for stages, threshold in ((0x1F, 64), (0x2E, 140), (0x0F, 200)):
            write(out, "fuzz_camera", w + bytes([0, stages, threshold]) + image)
    image = camera_capture(64, 48, False)
    padded = b"".join(image[y * 64 : (y + 1) * 64] + bytes(5) for y in range(48))
    write(out, "fuzz_camera", (63).to_bytes(2, "little") + bytes([5, 0x0F, 100]) + padded)


def font_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_font` seeds.  Selector: [face_index, size]."""
    for pattern in (
        "crates/font/src/tests/data/*",
        "crates/font_test_data/test_data/ttf/*",
        "crates/font_test_data/test_data/ttc/*",
        "crates/font_test_data/test_data/type1/*",
    ):
        for f in committed(pattern):
            if f.suffix.lower() in FONT_SUFFIXES:
                write(out, "fuzz_font", bytes([0, 24]) + f.read_bytes())


def filter_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_filters` seeds.

    Selector: [count, f0, f1, f2, predictor, colors, bpc, columns, early].
    """
    text = b"BT /F1 12 Tf (filter chain seed) Tj ET\n" * 8
    rows = b"".join(b"\x02" + bytes(range(i, i + 8)) for i in range(16))
    for selector, body in (
        ([0, 0, 0, 0, 0, 0, 3, 7, 1], zlib.compress(text)),
        ([1, 2, 0, 0, 0, 0, 3, 7, 1], zlib.compress(text).hex().encode()),
        ([0, 0, 0, 0, 4, 0, 3, 7, 1], zlib.compress(rows)),
        ([0, 4, 0, 0, 0, 0, 3, 7, 1], b"\x02abc\xfdz\x80"),
    ):
        write(out, "fuzz_filters", bytes(selector) + body)


def classic_xref(body: bytearray, offsets: list[int], trailer: bytes) -> bytes:
    """Append a classic xref for objects 1..len(offsets) and `trailer` to `body`."""
    xref = len(body)
    body += b"xref\n0 %d\n0000000000 65535 f\r\n" % (len(offsets) + 1)
    for o in offsets:
        body += b"%010d 00000 n\r\n" % o
    body += b"trailer\n<<%s>>\nstartxref\n%d\n%%%%EOF\n" % (trailer, xref)
    return bytes(body)


def amplification_seeds(out: pathlib.Path) -> None:
    """Write `fuzz_parse` seeds whose cost once grew faster than their size.

    - Stream objects whose /Length misses and that share one `endstream`
      at the end of the file: each body must end at the next object.
    - Member lookups into an object stream that decodes to 64 MiB but whose
      /First lies past its data: the failed open must be decoded once.
    """
    head = b"%PDF-1.7\n"
    catalog = (
        b"1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n",
        b"2 0 obj <</Type/Pages/Kids[]/Count 0>> endobj\n",
    )

    body = bytearray(head)
    offsets = []
    for obj in catalog:
        offsets.append(len(body))
        body += obj
    for i in range(500):
        offsets.append(len(body))
        body += b"%d 0 obj <</Length 1>>stream\n" % (i + 3) + b"A" * 1500 + b"\n"
    body += b"endstream\nendobj\n"
    write_pdf(out, classic_xref(body, offsets, b"/Size %d/Root 1 0 R" % (len(offsets) + 1)))

    size = 64 << 20
    members = 500
    packed = zlib.compress(bytes(size), 9)
    body = bytearray(head)
    offsets = []
    for obj in catalog:
        offsets.append(len(body))
        body += obj
    offsets.append(len(body))
    body += b"3 0 obj <</Type/ObjStm/N 1/First %d/Filter/FlateDecode/Length %d>>stream\n" % (
        size + 10,
        len(packed),
    )
    body += packed + b"\nendstream\nendobj\n"
    entries = b"".join(b"\x02" + (3).to_bytes(4, "big") + bytes([i % 256]) for i in range(members))
    xref_stm = len(body)
    body += b"%d 0 obj <</Type/XRef/Size %d/W[1 4 1]/Index[4 %d]/Length %d>>stream\n" % (
        4 + members,
        5 + members,
        members,
        len(entries),
    )
    body += entries + b"\nendstream\nendobj\n"
    trailer = b"/Size %d/Root 1 0 R/XRefStm %d" % (5 + members, xref_stm)
    write_pdf(out, classic_xref(body, offsets, trailer))


def alternation_seeds(out: pathlib.Path) -> None:
    """Write the `fuzz_parse` seeds of object-stream thrash and header text in a body.

    - Members of two 768 KiB object streams read in alternation: together
      past the 1 MiB cache budget of a fuzzing build, so re-decodes of
      evicted containers must stop at the cache's allowance.
    - A content stream under an indirect /Length holding the text `1 0 obj`:
      its body must not end there.
    """
    head = b"%PDF-1.7\n"
    catalog = (
        b"1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n",
        b"2 0 obj <</Type/Pages/Kids[]/Count 0>> endobj\n",
    )
    # Two object streams of 768 KiB each, whose members alternate by
    # number, so that reading them in order alternates the containers.
    size = 768 << 10
    members = 250
    body = bytearray(head)
    offsets = []
    for obj in catalog:
        offsets.append(len(body))
        body += obj
    for c in range(2):
        nums = [5 + 2 * i + c for i in range(members)]
        directory = b"".join(b"%d %d " % (n, 5 * i) for i, n in enumerate(nums))
        data = directory + b"null " * members
        packed = zlib.compress(data + bytes(size - len(data)), 9)
        offsets.append(len(body))
        body += b"%d 0 obj <</Type/ObjStm/N %d/First %d/Filter/FlateDecode/Length %d>>stream\n" % (
            3 + c,
            members,
            len(directory),
            len(packed),
        )
        body += packed + b"\nendstream\nendobj\n"
    entries = b"".join(
        b"\x02" + (3 + i % 2).to_bytes(4, "big") + (i // 2).to_bytes(2, "big")
        for i in range(2 * members)
    )
    total = 5 + 2 * members
    xref_stm = len(body)
    body += b"%d 0 obj <</Type/XRef/Size %d/W[1 4 2]/Index[5 %d]/Length %d>>stream\n" % (
        total,
        total + 1,
        2 * members,
        len(entries),
    )
    body += entries + b"\nendstream\nendobj\n"
    trailer = b"/Size %d/Root 1 0 R/XRefStm %d" % (total + 1, xref_stm)
    write_pdf(out, classic_xref(body, offsets, trailer))

    # A page whose content stream, under an indirect /Length, shows the
    # text of an object header.
    content = b"BT /F1 12 Tf 72 712 Td (1 0 obj) Tj 0 -14 Td (2 0 obj <<) Tj ET"
    body = bytearray(head)
    offsets = []
    for obj in (
        b"1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n",
        b"2 0 obj <</Type/Pages/Kids[3 0 R]/Count 1>> endobj\n",
        b"3 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>> endobj\n",
        b"4 0 obj <</Length 5 0 R>>stream\n" + content + b"\nendstream\nendobj\n",
        b"5 0 obj %d endobj\n" % len(content),
    ):
        offsets.append(len(body))
        body += obj
    write_pdf(out, classic_xref(body, offsets, b"/Size 6/Root 1 0 R"))


def repo_seeds(out: pathlib.Path) -> None:
    """Write the seeds built from files committed to this repository."""
    image_seeds(out)
    icc_seeds(out)
    jpeg_seeds(out)
    jpeg2000_seeds(out)
    comic_seeds(out)
    camera_seeds(out)
    font_seeds(out)
    filter_seeds(out)
    amplification_seeds(out)
    alternation_seeds(out)
    for c in CONTENT_SEEDS:
        write(out, "fuzz_content", c)
    for c in CMAP_SEEDS:
        write(out, "fuzz_cmap", c)
    for sel, body in FUNCTION_SEEDS:
        write(out, "fuzz_function", sel + body)


def pdf_seeds(out: pathlib.Path, root: pathlib.Path) -> None:
    """Write every PDF of at most `MAX_SEED` bytes under `root`."""
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    n = 0
    for pdf in sorted(root.rglob("*.pdf")):
        if pdf.is_file() and pdf.stat().st_size <= MAX_SEED:
            write_pdf(out, pdf.read_bytes())
            n += 1
    print(f"{root}: {n} PDF seeds", file=sys.stderr)


def main() -> None:
    """Build the seed corpora under the output directory."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--pdfium", type=pathlib.Path, help="pdfium source checkout")
    ap.add_argument("--extra-pdfs", type=pathlib.Path, help="directory of PDFs to add")
    args = ap.parse_args()
    plaintext = plaintext_pdf()
    repo_seeds(args.out)
    write_pdf(args.out, plaintext)
    encrypted_seeds(args.out, plaintext)
    if args.pdfium:
        pdf_seeds(args.out, args.pdfium / "testing" / "resources")
    if args.extra_pdfs:
        pdf_seeds(args.out, args.extra_pdfs)
    for d in sorted(args.out.iterdir()):
        print(f"{d.name}: {sum(1 for _ in d.iterdir())} seeds")


if __name__ == "__main__":
    main()
