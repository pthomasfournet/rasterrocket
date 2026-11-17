#![no_main]

//! The JPEG 2000 decoder (the hayro-jpeg2000 fork): the JP2 boxes,
//! codestream markers, tile parts, packet headers, the EBCOT and HTJ2K
//! block decoders, dequantisation, the inverse wavelets and component
//! transforms, palettes and reduced-resolution decoding, as the
//! `JPXDecode` filter calls them.
//!
//! Layout: `[mode, size]`, then the JP2 file or codestream.  Bit 0 of
//! `mode` leaves palette indices unresolved; bits 1–2, when not zero, read
//! a raw codestream with 1, 3 or 4 colour channels; bit 3 decodes to 8-bit
//! samples rather than at the components' precision.  A non-zero `size`
//! asks for the image at no less than `size × size` pixels.

use hayro_jpeg2000::{DecodeSettings, DecoderContext, Image};
use libfuzzer_sys::fuzz_target;

/// Largest decoded image, in samples, the target decodes.
const MAX_SAMPLES: u64 = 16 << 20;

const CHANNELS: [u8; 4] = [0, 1, 3, 4];

fuzz_target!(|data: &[u8]| {
    let Some(([mode, size], stream)) = rasterrocket_fuzz::split_selectors::<2>(data) else {
        return;
    };
    let settings = DecodeSettings {
        resolve_palette_indices: mode & 1 == 0,
        target_resolution: (size != 0).then(|| (u32::from(size), u32::from(size))),
        ..DecodeSettings::default()
    };
    let channels = CHANNELS[usize::from(mode >> 1) & 3];
    let image = if channels == 0 {
        Image::new(stream, &settings)
    } else {
        Image::from_codestream(stream, channels, &settings)
    };
    let Ok(image) = image else {
        return;
    };
    let samples = u64::from(image.width())
        * u64::from(image.height())
        * (u64::from(image.color_space().num_channels()) + u64::from(image.has_alpha()));
    if samples > MAX_SAMPLES {
        return;
    }
    std::hint::black_box(image.survey().is_ok());
    let mut ctx = DecoderContext::default();
    if mode & 8 != 0 {
        drop(image.decode_u8(&mut ctx));
    } else if let Ok(decoded) = image.decode(&mut ctx) {
        std::hint::black_box(decoded.data_u8().len());
    }
    std::hint::black_box(ctx.completeness());
});
