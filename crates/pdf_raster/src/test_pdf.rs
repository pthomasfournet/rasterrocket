//! In-memory PDF documents for unit tests.

/// A PDF holding `objects` as objects 1, 2, … in order, with a classic
/// cross-reference table and a trailer whose `/Root` is object 1.
pub(crate) fn assemble(objects: &[&str]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f\r\n", objects.len() + 1).as_bytes(),
    );
    for off in offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n\r\n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<</Size {} /Root 1 0 R>>\nstartxref\n{xref}\n%%EOF",
            objects.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

/// A content stream object holding `content`.
pub(crate) fn stream(content: &str) -> String {
    format!(
        "<</Length {}>>\nstream\n{content}\nendstream",
        content.len()
    )
}
