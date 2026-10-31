"""Per-page pdfium render latency, in process.

Usage: pdfium_latency.py DPI PDF...

Prints one tab-separated line per rendered page: pdf, page, open_us,
page_us.  Documents that cannot be read or opened, and pages pdfium cannot
load, are reported on stderr and left out of the table.
"""

import argparse
import math
import sys
import time

import pypdfium2 as pdfium

POINTS_PER_INCH = 72


def positive(text: str) -> float:
    """Parse a finite, strictly positive number for argparse."""
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        msg = f"not a positive number: {text}"
        raise argparse.ArgumentTypeError(msg)
    return value


def elapsed_us(start: float) -> int:
    """Return the microseconds since the `time.perf_counter()` reading `start`."""
    return int((time.perf_counter() - start) * 1e6)


def time_document(pdf: str, scale: float) -> None:
    """Print the open time and every page's render time of `pdf`."""
    start = time.perf_counter()
    try:
        doc = pdfium.PdfDocument(pdf)
    except (OSError, pdfium.PdfiumError) as e:
        print(f"{pdf}: cannot open: {e}", file=sys.stderr)
        return
    try:
        n = len(doc)
        open_us = elapsed_us(start)
        for i in range(n):
            start = time.perf_counter()
            try:
                page = doc[i]
                try:
                    page.render(scale=scale, grayscale=True).close()
                finally:
                    page.close()
            except pdfium.PdfiumError as e:
                print(f"{pdf}: page {i + 1}: {e}", file=sys.stderr)
                continue
            print(f"{pdf}\t{i + 1}\t{open_us}\t{elapsed_us(start)}", flush=True)
    finally:
        doc.close()


def main() -> None:
    """Time every document on the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dpi", type=positive)
    ap.add_argument("pdfs", nargs="+", metavar="PDF")
    args = ap.parse_args()
    for pdf in args.pdfs:
        time_document(pdf, args.dpi / POINTS_PER_INCH)


if __name__ == "__main__":
    main()
