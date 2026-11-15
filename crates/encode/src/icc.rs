//! An ICC profile to embed in an image file: the profile's bytes, checked
//! against the ICC.1 header, and the facts a writer needs from it.

use std::sync::Arc;

use crate::EncodeError;
use crate::rows::RowLayout;

/// Bytes of the ICC profile header (ICC.1:2022 §7.2).
const HEADER_BYTES: usize = 128;

/// An ICC profile of an RGB, gray or CMYK colour space, of version 2 or
/// 4, to embed in an image of the same colour space.  Cloning shares the bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IccProfile {
    bytes: Arc<[u8]>,
    channels: u8,
    version: (u8, u8),
}

impl IccProfile {
    /// The profile of `bytes`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for bytes shorter than the header,
    /// whose declared size (bytes 0–3) is not their length, without the
    /// `acsp` signature (bytes 36–39), of a major version other than 2 and
    /// 4, or of a data colour space (bytes 16–19) other than `RGB `,
    /// `GRAY` and `CMYK`.
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Result<Self, EncodeError> {
        let bytes = bytes.into();
        let Some(header) = bytes.first_chunk::<HEADER_BYTES>() else {
            return Err(EncodeError::InvalidInput(
                "an ICC profile holds at least its 128-byte header",
            ));
        };
        let declared = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        if usize::try_from(declared).ok() != Some(bytes.len()) {
            return Err(EncodeError::InvalidInput(
                "the ICC profile's declared size is not its length",
            ));
        }
        if &header[36..40] != b"acsp" {
            return Err(EncodeError::InvalidInput(
                "not an ICC profile: no 'acsp' signature",
            ));
        }
        let version = (header[8], header[9] >> 4);
        if !matches!(version.0, 2 | 4) {
            return Err(EncodeError::InvalidInput(
                "only ICC profiles of version 2 and 4 are embedded",
            ));
        }
        let channels = match &header[16..20] {
            b"RGB " => 3,
            b"GRAY" => 1,
            b"CMYK" => 4,
            _ => {
                return Err(EncodeError::InvalidInput(
                    "only RGB, gray and CMYK ICC profiles are embedded",
                ));
            }
        };
        Ok(Self {
            bytes,
            channels,
            version,
        })
    }

    /// The profile's bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The colour channels of the profile's data colour space: 3, 1 or 4.
    #[must_use]
    pub const fn channels(&self) -> u8 {
        self.channels
    }

    /// The profile's major and minor version.
    #[must_use]
    pub const fn version(&self) -> (u8, u8) {
        self.version
    }

    /// Check that the profile describes the colour of rows stored as
    /// `layout`: RGB (with or without alpha) for a profile of three
    /// channels, gray for one of one, CMYK for one of four; a bilevel image
    /// takes no profile.
    pub(crate) const fn check(&self, layout: RowLayout) -> Result<(), EncodeError> {
        let channels = match layout {
            RowLayout::Rgb | RowLayout::Rgba | RowLayout::Rgb16 | RowLayout::Rgba16 => 3,
            RowLayout::Gray | RowLayout::Gray16 => 1,
            RowLayout::Cmyk | RowLayout::Cmyk16 => 4,
            RowLayout::Bilevel => {
                return Err(EncodeError::InvalidInput(
                    "a bilevel image takes no ICC profile",
                ));
            }
        };
        if channels == self.channels {
            return Ok(());
        }
        Err(EncodeError::InvalidInput(match (self.channels, channels) {
            (1, 3) => "a gray ICC profile cannot tag an RGB image",
            (4, 3) => "a CMYK ICC profile cannot tag an RGB image",
            (3, 1) => "an RGB ICC profile cannot tag a gray image",
            (4, 1) => "a CMYK ICC profile cannot tag a gray image",
            (3, _) => "an RGB ICC profile cannot tag a CMYK image",
            _ => "a gray ICC profile cannot tag a CMYK image",
        }))
    }
}

/// [`IccProfile::check`] of `icc`, when given, against `layout`.
pub(crate) fn check(icc: Option<&IccProfile>, layout: RowLayout) -> Result<(), EncodeError> {
    icc.map_or(Ok(()), |icc| icc.check(layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header-only profile of `space` (`RGB `, `GRAY` or `CMYK`) at `major`
    /// version: valid as far as [`IccProfile::new`] reads.
    fn profile(space: [u8; 4], major: u8) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_BYTES];
        bytes[..4].copy_from_slice(&128u32.to_be_bytes());
        bytes[8] = major;
        bytes[12..16].copy_from_slice(b"mntr");
        bytes[16..20].copy_from_slice(&space);
        bytes[20..24].copy_from_slice(b"XYZ ");
        bytes[36..40].copy_from_slice(b"acsp");
        bytes
    }

    #[test]
    fn reads_the_header() {
        let rgb = IccProfile::new(profile(*b"RGB ", 4)).expect("valid");
        assert_eq!((rgb.channels(), rgb.version().0), (3, 4));
        let gray = IccProfile::new(profile(*b"GRAY", 2)).expect("valid");
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.bytes().len(), 128);
        let cmyk = IccProfile::new(profile(*b"CMYK", 2)).expect("valid");
        assert_eq!(cmyk.channels(), 4);
    }

    #[test]
    fn refuses_what_is_not_an_rgb_gray_or_cmyk_profile() {
        let mut short = profile(*b"RGB ", 4);
        short.truncate(100);
        let mut sized = profile(*b"RGB ", 4);
        sized.push(0);
        let mut unsigned = profile(*b"RGB ", 4);
        unsigned[36] = b'x';
        for bytes in [
            short,
            sized,
            unsigned,
            profile(*b"Lab ", 4),
            profile(*b"RGB ", 5),
        ] {
            assert!(
                matches!(IccProfile::new(bytes), Err(EncodeError::InvalidInput(_))),
                "accepted"
            );
        }
    }

    #[test]
    fn checks_the_colour_of_the_rows() {
        let rgb = IccProfile::new(profile(*b"RGB ", 4)).expect("valid");
        let gray = IccProfile::new(profile(*b"GRAY", 4)).expect("valid");
        assert!(rgb.check(RowLayout::Rgba16).is_ok());
        assert!(gray.check(RowLayout::Gray).is_ok());
        assert!(rgb.check(RowLayout::Gray).is_err());
        assert!(gray.check(RowLayout::Rgb).is_err());
        assert!(rgb.check(RowLayout::Bilevel).is_err());
        assert!(check(None, RowLayout::Bilevel).is_ok());
        let cmyk = IccProfile::new(profile(*b"CMYK", 4)).expect("valid");
        assert!(cmyk.check(RowLayout::Cmyk).is_ok());
        assert!(cmyk.check(RowLayout::Cmyk16).is_ok());
        assert!(cmyk.check(RowLayout::Rgb).is_err());
        assert!(rgb.check(RowLayout::Cmyk).is_err());
        assert!(gray.check(RowLayout::Cmyk16).is_err());
    }
}
