"""Summarise two per-page latency tables (rasterrocket, pdfium).

Usage: latency_summary.py RR_TSV PF_TSV

Each table has the tab-separated columns pdf, page, open_us, page_us.  The
summary covers the pages both renderers produced.
"""

import argparse
import collections
import pathlib
import statistics
import sys
from collections.abc import Collection, Sequence
from typing import NamedTuple

COLUMNS = 4

PageKey = tuple[str, int]


class Table(NamedTuple):
    """Per-page render times and per-document open times, in microseconds."""

    pages: dict[PageKey, int]
    opens: dict[str, int]


def parse_line(line: str) -> tuple[str, int, int, int]:
    """Return `(pdf, page, open_us, page_us)` of one table line."""
    fields = line.rstrip("\n").split("\t")
    if len(fields) != COLUMNS:
        msg = f"{len(fields)} columns, expected {COLUMNS}"
        raise ValueError(msg)
    pdf, page, open_us, page_us = fields
    return pdf, int(page), int(open_us), int(page_us)


def load(path: pathlib.Path) -> Table:
    """Load `path`; exit naming the line when one is malformed."""
    table = Table({}, {})
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                pdf, page, open_us, page_us = parse_line(line)
            except ValueError as e:
                sys.exit(f"{path}:{n}: {e}")
            table.pages[pdf, page] = page_us
            table.opens[pdf] = open_us
    return table


def percentile(values: Sequence[int], p: float) -> int:
    """Return the nearest-rank `p` quantile (0 <= p <= 1) of non-empty `values`."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def print_totals(rr: Table, pf: Table, keys: list[PageKey], docs: Collection[str]) -> None:
    """Print the per-renderer distribution, the win count and the open times."""
    for name, table in (("rasterrocket", rr), ("pdfium", pf)):
        v = [table.pages[k] for k in keys]
        print(
            f"{name:>12}: total {sum(v) / 1e6:7.1f} s  mean {statistics.mean(v) / 1e3:6.2f} ms"
            f"  median {statistics.median(v) / 1e3:6.2f}  p90 {percentile(v, 0.9) / 1e3:6.2f}"
            f"  p99 {percentile(v, 0.99) / 1e3:6.1f}"
        )
    faster = sum(rr.pages[k] < pf.pages[k] for k in keys)
    print(f"rasterrocket faster on {faster} of {len(keys)} pages")
    print(
        f"document open: rasterrocket {sum(rr.opens[d] for d in docs) / 1e6:.3f} s"
        f"  pdfium {sum(pf.opens[d] for d in docs) / 1e6:.3f} s"
    )


def print_slowest(rr: Table, pf: Table, by_doc: dict[str, list[PageKey]], limit: int = 10) -> None:
    """Print the `limit` documents slowest relative to pdfium."""
    rows = []
    for d, ks in by_doc.items():
        r = sum(rr.pages[k] for k in ks)
        p = sum(pf.pages[k] for k in ks)
        rows.append((r / max(p, 1), r / 1e6, p / 1e6, len(ks), d))
    rows.sort(reverse=True)
    print("slowest relative to pdfium (ratio, rasterrocket s, pdfium s, pages, document):")
    for ratio, r_s, p_s, n, d in rows[:limit]:
        print(f"  {ratio:5.2f} {r_s:7.2f} {p_s:7.2f} {n:5d}  {d}")


def main() -> None:
    """Print the comparison of the two tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rr", type=pathlib.Path, help="rasterrocket table")
    ap.add_argument("pf", type=pathlib.Path, help="pdfium table")
    args = ap.parse_args()
    rr, pf = load(args.rr), load(args.pf)

    keys = sorted(rr.pages.keys() & pf.pages.keys())
    by_doc: dict[str, list[PageKey]] = collections.defaultdict(list)
    for key in keys:
        by_doc[key[0]].append(key)
    print(f"{len(keys)} pages in {len(by_doc)} documents rendered by both")
    if keys:
        print_totals(rr, pf, keys, by_doc)
        print_slowest(rr, pf, by_doc)


if __name__ == "__main__":
    main()
