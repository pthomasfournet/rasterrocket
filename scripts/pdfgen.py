"""Minimal PDF assembly for test and benchmark scripts.

The Python counterpart of `crates/pdf_raster/tests/common`: documents are
built from object bodies with an exact cross-reference table.
"""


def assemble_pdf(objects):
    """A PDF whose objects are `objects` (bytes, numbered from 1) with an
    exact xref table; object 1 is the catalog."""
    pdf = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(pdf)


def stream(dict_entries, data):
    """A stream object body with `dict_entries` and an exact /Length."""
    return (
        b"<< %s /Length %d >>\nstream\n" % (dict_entries, len(data))
        + data
        + b"\nendstream"
    )


def one_page(page, rest):
    """Catalog 1, Pages 2, `page` as object 3 and `rest` from object 4 on."""
    return assemble_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page,
            *rest,
        ]
    )
