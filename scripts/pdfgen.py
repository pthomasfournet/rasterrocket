"""Minimal PDF assembly for test and benchmark scripts.

The Python counterpart of `crates/pdf_raster/tests/common`: documents are
built from object bodies with an exact cross-reference table.
"""

from collections.abc import Iterable

LETTER = b"0 0 612 792"


def assemble_pdf(objects: Iterable[bytes]) -> bytes:
    """Return a PDF of `objects` (numbered from 1) with an exact xref table.

    Object 1 is the catalog.
    """
    pdf = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(pdf)
    size = len(offsets) + 1
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(pdf)


def stream(dict_entries: bytes, data: bytes) -> bytes:
    """Return a stream object body with `dict_entries` and an exact /Length."""
    return b"<< %s /Length %d >>\nstream\n" % (dict_entries, len(data)) + data + b"\nendstream"


def one_page(
    content: bytes,
    resources: bytes = b"",
    rest: Iterable[bytes] = (),
    *,
    media_box: bytes = LETTER,
    page_entries: bytes = b"",
) -> bytes:
    """Return a one-page PDF drawing `content` with `resources`.

    The objects are the catalog (1), the page tree (2), the page (3), its
    content stream (4) and `rest` from object 5 on.  `page_entries` are
    further entries of the page dictionary.
    """
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [%s] /Contents 4 0 R /Resources << %s >>%s >>"
        % (media_box, resources, b" " + page_entries if page_entries else b"")
    )
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            stream(b"", content),
            *rest,
        ]
    )
