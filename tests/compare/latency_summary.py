"""Summarise two per-page latency tables (rasterrocket, pdfium).

Usage: latency_summary.py RR_TSV PF_TSV
"""

import statistics
import sys


def load(path):
    pages, opens = {}, {}
    with open(path) as f:
        for line in f:
            pdf, page, open_us, page_us = line.rstrip("\n").split("\t")
            pages[(pdf, int(page))] = int(page_us)
            opens[pdf] = int(open_us)
    return pages, opens


rr, rr_open = load(sys.argv[1])
pf, pf_open = load(sys.argv[2])
keys = sorted(set(rr) & set(pf))
docs = sorted({pdf for pdf, _ in keys})
print(f"{len(keys)} pages in {len(docs)} documents rendered by both")
if not keys:
    sys.exit(0)


def pct(v, p):
    return sorted(v)[min(len(v) - 1, int(p * len(v)))]


for name, table in (("rasterrocket", rr), ("pdfium", pf)):
    v = [table[k] for k in keys]
    print(
        f"{name:>12}: total {sum(v) / 1e6:7.1f} s  mean {statistics.mean(v) / 1e3:6.2f} ms"
        f"  median {statistics.median(v) / 1e3:6.2f}  p90 {pct(v, 0.9) / 1e3:6.2f}"
        f"  p99 {pct(v, 0.99) / 1e3:6.1f}"
    )
faster = sum(rr[k] < pf[k] for k in keys)
print(f"rasterrocket faster on {faster} of {len(keys)} pages")
print(
    f"document open: rasterrocket {sum(rr_open[d] for d in docs) / 1e6:.3f} s"
    f"  pdfium {sum(pf_open[d] for d in docs) / 1e6:.3f} s"
)
rows = []
for d in docs:
    ks = [k for k in keys if k[0] == d]
    r = sum(rr[k] for k in ks)
    p = sum(pf[k] for k in ks)
    rows.append((r / max(p, 1), r / 1e6, p / 1e6, len(ks), d))
rows.sort(reverse=True)
print("slowest relative to pdfium (ratio, rasterrocket s, pdfium s, pages, document):")
for ratio, r, p, n, d in rows[:10]:
    print(f"  {ratio:5.2f} {r:7.2f} {p:7.2f} {n:5d}  {d}")
