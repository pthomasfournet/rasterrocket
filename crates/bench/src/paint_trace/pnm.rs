//! Binary PPM (`P6`) and PGM (`P5`) images of 8-bit samples.

use std::path::Path;

/// An image: `channels` 8-bit samples per pixel, row-major.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub samples: Vec<u8>,
}

impl Image {
    /// The samples of pixel `(x, y)`.
    pub fn pixel(&self, x: usize, y: usize) -> &[u8] {
        let at = (y * self.width + x) * self.channels;
        &self.samples[at..at + self.channels]
    }
}

/// The image in the file `path`.
pub fn read(path: &Path) -> Result<Image, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The image `bytes` hold.
fn parse(bytes: &[u8]) -> Result<Image, String> {
    let channels = match bytes.get(..2) {
        Some(b"P6") => 3,
        Some(b"P5") => 1,
        _ => return Err("not a binary PPM or PGM (P6/P5) file".to_owned()),
    };
    let mut at = 2;
    let mut fields = [0_usize; 3];
    for field in &mut fields {
        // Whitespace and `#` comments separate the header fields.
        loop {
            match bytes.get(at) {
                Some(b'#') => {
                    while bytes.get(at).is_some_and(|&b| b != b'\n') {
                        at += 1;
                    }
                }
                Some(b) if b.is_ascii_whitespace() => at += 1,
                _ => break,
            }
        }
        let start = at;
        while bytes.get(at).is_some_and(u8::is_ascii_digit) {
            at += 1;
        }
        *field = std::str::from_utf8(&bytes[start..at])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or("malformed header")?;
    }
    let [width, height, max] = fields;
    if max != 255 {
        return Err(format!(
            "{max} as the largest sample: only 8-bit samples are read"
        ));
    }
    // One whitespace byte ends the header.
    let data = bytes.get(at + 1..).ok_or("truncated header")?;
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(channels))
        .ok_or("image too large")?;
    let samples = data.get(..len).ok_or("truncated samples")?.to_vec();
    Ok(Image {
        width,
        height,
        channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn reads_a_commented_pgm() {
        let img = parse(b"P5\n# c\n2 1\n255\n\x01\x02").expect("parses");
        assert_eq!((img.width, img.height, img.channels), (2, 1, 1));
        assert_eq!(img.pixel(1, 0), &[2]);
    }

    #[test]
    fn refuses_wide_samples_and_truncation() {
        assert!(parse(b"P6 1 1 65535\n\0\0\0\0\0\0").is_err());
        assert!(parse(b"P6 2 1 255\n\0\0\0").is_err());
        assert!(parse(b"P5 1 1 255").is_err());
    }
}
