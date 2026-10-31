#!/usr/bin/env python3
"""Extract the Foxit base-14 font programs from a pdfium checkout.

pdfium ships them as C++ byte arrays under
``core/fxge/fontdata/chromefontdata/*.cpp``.  Each array is one bare CFF
program, except the two multiple-master faces (``FoxitSansMM``,
``FoxitSerifMM``), which are Type 1 MM programs in PFB layout and are
skipped: the renderer's Type 1 interpreter does not blend MM designs.
This script writes every other array to ``<out>/<Name>.cff`` and copies
pdfium's LICENSE next to them.

Usage: extract_foxit_fonts.py <pdfium-src> <out-dir>
"""

import argparse
import pathlib
import re
import shutil
import sys

ARRAY_RE = re.compile(
    r"const std::array<uint8_t, (\d+)> kFoxit(\w+)FontData = \{\{(.*?)\}\};",
    re.DOTALL,
)
SKIP = {"SansMM", "SerifMM"}


def main() -> int:
    """Extract the arrays; return 0, or 1 on a malformed or missing source."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=pathlib.Path, help="pdfium source checkout")
    parser.add_argument("out", type=pathlib.Path, help="output directory")
    args = parser.parse_args()
    src: pathlib.Path = args.src
    out: pathlib.Path = args.out
    data_dir = src / "core" / "fxge" / "fontdata" / "chromefontdata"
    sources = sorted(data_dir.glob("Foxit*.cpp"))
    if not sources:
        print(f"{data_dir}: no Foxit*.cpp font data (not a pdfium checkout?)", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for cpp in sources:
        text = cpp.read_text(encoding="utf-8")
        m = ARRAY_RE.search(text)
        if m is None:
            print(f"{cpp}: no font array found", file=sys.stderr)
            return 1
        length, name, body = int(m.group(1)), m.group(2), m.group(3)
        if name in SKIP:
            continue
        data = bytes(int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", body))
        if len(data) != length:
            print(f"{cpp}: {len(data)} bytes, header says {length}", file=sys.stderr)
            return 1
        (out / f"{name}.cff").write_bytes(data)
        print(f"{name}.cff: {length} bytes")
    shutil.copy(src / "LICENSE", out / "LICENSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
