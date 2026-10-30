"""Per-page pdfium render latency, in process.

Usage: pdfium_latency.py DPI PDF...; prints pdf, page, open_us, page_us.
"""

import sys
import time

import pypdfium2 as pdfium

dpi = float(sys.argv[1])
for pdf in sys.argv[2:]:
    t = time.perf_counter()
    try:
        doc = pdfium.PdfDocument(pdf)
        n = len(doc)
    except pdfium.PdfiumError:
        continue
    open_us = int((time.perf_counter() - t) * 1e6)
    for i in range(n):
        t = time.perf_counter()
        try:
            page = doc[i]
            page.render(scale=dpi / 72, grayscale=True)
            page.close()
        except pdfium.PdfiumError:
            continue
        us = int((time.perf_counter() - t) * 1e6)
        print(f"{pdf}\t{i + 1}\t{open_us}\t{us}", flush=True)
