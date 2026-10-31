#!/usr/bin/env python3
"""Vendor the predefined CJK CMaps from a cmap-resources checkout.

The source is a checkout of adobe-type-tools/cmap-resources.

The set is the CMaps ISO 32000-2 Table 116 lists as predefined (except
Identity-H / Identity-V, which need no file) plus every CMap they reach
through `usecmap`.  Each is written zlib-compressed to ``<out>/<name>.z``
along with the repository's LICENSE.md.

Usage: vendor_cmaps.py <cmap-resources> <out-dir>
"""

import argparse
import pathlib
import re
import shutil
import sys
import zlib

PREDEFINED = [
    # Adobe-GB1
    "GB-EUC-H",
    "GB-EUC-V",
    "GBpc-EUC-H",
    "GBpc-EUC-V",
    "GBK-EUC-H",
    "GBK-EUC-V",
    "GBKp-EUC-H",
    "GBKp-EUC-V",
    "GBK2K-H",
    "GBK2K-V",
    "UniGB-UCS2-H",
    "UniGB-UCS2-V",
    "UniGB-UTF16-H",
    "UniGB-UTF16-V",
    # Adobe-CNS1
    "B5pc-H",
    "B5pc-V",
    "HKscs-B5-H",
    "HKscs-B5-V",
    "ETen-B5-H",
    "ETen-B5-V",
    "ETenms-B5-H",
    "ETenms-B5-V",
    "CNS-EUC-H",
    "CNS-EUC-V",
    "UniCNS-UCS2-H",
    "UniCNS-UCS2-V",
    "UniCNS-UTF16-H",
    "UniCNS-UTF16-V",
    # Adobe-Japan1
    "83pv-RKSJ-H",
    "90ms-RKSJ-H",
    "90ms-RKSJ-V",
    "90msp-RKSJ-H",
    "90msp-RKSJ-V",
    "90pv-RKSJ-H",
    "Add-RKSJ-H",
    "Add-RKSJ-V",
    "EUC-H",
    "EUC-V",
    "Ext-RKSJ-H",
    "Ext-RKSJ-V",
    "H",
    "V",
    "UniJIS-UCS2-H",
    "UniJIS-UCS2-V",
    "UniJIS-UCS2-HW-H",
    "UniJIS-UCS2-HW-V",
    "UniJIS-UTF16-H",
    "UniJIS-UTF16-V",
    # Adobe-Korea1
    "KSC-EUC-H",
    "KSC-EUC-V",
    "KSCms-UHC-H",
    "KSCms-UHC-V",
    "KSCms-UHC-HW-H",
    "KSCms-UHC-HW-V",
    "KSCpc-EUC-H",
    "UniKS-UCS2-H",
    "UniKS-UCS2-V",
    "UniKS-UTF16-H",
    "UniKS-UTF16-V",
]

USECMAP_RE = re.compile(rb"/(\S+)\s+usecmap")


def main() -> int:
    """Vendor the CMaps; return 0, or 1 when a CMap is missing or ambiguous."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=pathlib.Path, help="cmap-resources checkout")
    parser.add_argument("out", type=pathlib.Path, help="output directory")
    args = parser.parse_args()
    src: pathlib.Path = args.src
    out: pathlib.Path = args.out
    candidates: dict[str, list[pathlib.Path]] = {}
    for found in sorted(src.glob("Adobe-*/CMap/*")):
        candidates.setdefault(found.name, []).append(found)
    out.mkdir(parents=True, exist_ok=True)
    todo, done = list(PREDEFINED), set()
    while todo:
        name = todo.pop()
        if name in done:
            continue
        paths = candidates.get(name)
        if paths is None:
            print(f"{name}: not in {src}", file=sys.stderr)
            return 1
        data = paths[0].read_bytes()
        conflict = next((q for q in paths[1:] if q.read_bytes() != data), None)
        if conflict is not None:
            print(f"{name}: differs between {paths[0]} and {conflict}", file=sys.stderr)
            return 1
        (out / f"{name}.z").write_bytes(zlib.compress(data, 9))
        done.add(name)
        todo.extend(m.group(1).decode() for m in USECMAP_RE.finditer(data))
    shutil.copy(src / "LICENSE.md", out / "LICENSE.md")
    print(f"{len(done)} CMaps written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
