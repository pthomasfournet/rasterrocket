/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_arguments,
    clippy::doc_markdown
)]

//! Color space conversion routines
//!
//! This files exposes functions to convert one colorspace to another in a jpeg
//! image
//!
//! Currently supported conversions are
//!
//! - `YCbCr` to `RGB,RGBA,GRAYSCALE,RGBX`.
//!
//!
//! Hey there, if your reading this it means you probably need something, so let me help you.
//!
//! There are 3 supported cpu extensions here.
//! 1. Scalar
//! 2. SSE
//! 3. AVX
//!
//! There are two types of the color convert functions
//!
//! 1. Acts on 16 pixels.
//! 2. Acts on 8 pixels.
//!
//! The reason for this is because when implementing the AVX part it occurred to me that we can actually
//! do better and process 2 MCU's if we change IDCT return type to be `i16's`, since a lot of
//! CPU's these days support AVX extensions, it becomes nice if we optimize for that path ,
//! therefore AVX routines can process 16 pixels directly and SSE and Scalar just compensate.
//!
//! By compensating, I mean I wrote the 16 pixels version operating on the 8 pixel version twice.
//!
//! Therefore if your looking to optimize some routines, probably start there.

pub use scalar::ycbcr_to_grayscale;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "x86")]
pub use crate::color_convert::avx::{ycbcr_to_rgb_avx2, ycbcr_to_rgba_avx2};
use crate::decoder::{ColorConvert16Ptr, ColorConvert32Ptr};

mod avx;
#[cfg(feature = "std")]
mod avx512;
mod neon64;
mod scalar;

#[allow(unused_variables)]
pub fn choose_ycbcr_to_rgb_convert_func(
    type_need: ColorSpace,
    options: &DecoderOptions,
) -> Option<ColorConvert16Ptr> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(feature = "x86")]
    {
        use zune_core::log::debug;
        if options.use_avx2() {
            debug!("Using AVX optimised color conversion functions");

            // I believe avx2 means sse4 is also available
            // match colorspace
            match type_need {
                ColorSpace::RGB => return Some(ycbcr_to_rgb_avx2),
                ColorSpace::RGBA => return Some(ycbcr_to_rgba_avx2),
                _ => (), // fall through to scalar, which has more types
            };
        }
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if options.use_neon() {
            use crate::color_convert::neon64::{ycbcr_to_rgb_neon, ycbcr_to_rgba_neon};
            match type_need {
                ColorSpace::RGB => return Some(ycbcr_to_rgb_neon),
                ColorSpace::RGBA => return Some(ycbcr_to_rgba_neon),
                _ => (), // fall through to scalar, which has more types
            }
        }
    }
    // when there is no x86 or we haven't returned by here, resort to scalar
    return match type_need {
        ColorSpace::RGB => Some(scalar::ycbcr_to_rgb_inner_16_scalar::<false>),
        ColorSpace::RGBA => Some(scalar::ycbcr_to_rgba_inner_16_scalar::<false>),
        ColorSpace::BGRA => Some(scalar::ycbcr_to_rgba_inner_16_scalar::<true>),
        ColorSpace::BGR => Some(scalar::ycbcr_to_rgb_inner_16_scalar::<true>),
        _ => None,
    };
}

/// A conversion of 32 YCbCr pixels at a time to `type_need`, when the CPU
/// has one: AVX-512 (F, BW and VBMI) to RGB, where AVX2 is allowed.  Its
/// output equals that of [`choose_ycbcr_to_rgb_convert_func`]'s conversion
/// of the two halves.
#[cfg_attr(
    not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "x86",
        feature = "std"
    )),
    expect(unused_variables, reason = "used only by the x86 AVX-512 branch")
)]
pub fn choose_ycbcr_to_rgb_convert_func_32(
    type_need: ColorSpace,
    options: &DecoderOptions,
) -> Option<ColorConvert32Ptr> {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "x86",
        feature = "std"
    ))]
    {
        if type_need == ColorSpace::RGB
            && crate::misc::avx512_allowed(options)
            && is_x86_feature_detected!("avx512vbmi")
        {
            return Some(|y, cb, cr, out| {
                // SAFETY: `avx512_allowed` holds only if AVX-512 F and BW are
                // supported, and VBMI is checked beside it.
                unsafe { avx512::ycbcr_to_rgb_avx512(y, cb, cr, out) }
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_convert::scalar::{
        ycbcr_to_rgb_inner_16_scalar, ycbcr_to_rgba_inner_16_scalar,
    };

    /// Every Cb/Cr pair at sixteen luma values, through the scalar and the
    /// dispatched converter of each packed layout, against BT.601 at 16
    /// fractional bits with halves rounded up; the bytes around the output
    /// stay untouched.
    #[test]
    fn ycbcr_to_rgb_matches_16_bit_reference() {
        let options = DecoderOptions::default();
        let y: [i16; 16] = core::array::from_fn(|index| (index * 17) as i16);
        let layouts: [(ColorSpace, ColorConvert16Ptr); 4] = [
            (ColorSpace::RGB, ycbcr_to_rgb_inner_16_scalar::<false>),
            (ColorSpace::RGBA, ycbcr_to_rgba_inner_16_scalar::<false>),
            (ColorSpace::BGR, ycbcr_to_rgb_inner_16_scalar::<true>),
            (ColorSpace::BGRA, ycbcr_to_rgba_inner_16_scalar::<true>),
        ];
        for (format, scalar) in layouts {
            let channels = format.num_components();
            let dispatched = choose_ycbcr_to_rgb_convert_func(format, &options).unwrap();
            for convert in [scalar, dispatched] {
                for cb in 0_i16..=255 {
                    for cr in 0_i16..=255 {
                        let mut output = [0xa5; 82];
                        let mut offset = 7;
                        convert(&y, &[cb; 16], &[cr; 16], &mut output, &mut offset);
                        assert_eq!(offset, 7 + channels * 16);
                        assert_eq!(&output[..7], &[0xa5; 7]);
                        assert!(output[offset..].iter().all(|&byte| byte == 0xa5));
                        let (cb, cr) = (i32::from(cb) - 128, i32::from(cr) - 128);
                        for (index, pixel) in output[7..offset].chunks_exact(channels).enumerate() {
                            let y = i32::from(y[index]);
                            let channel =
                                |sum: i32| (y + ((sum + 32_768) >> 16)).clamp(0, 255) as u8;
                            let red = channel(91_881 * cr);
                            let green = channel(-46_802 * cr - 22_554 * cb);
                            let blue = channel(116_130 * cb);
                            let expected = if matches!(format, ColorSpace::BGR | ColorSpace::BGRA) {
                                [blue, green, red, 255]
                            } else {
                                [red, green, blue, 255]
                            };
                            assert_eq!(pixel, &expected[..channels], "{format:?} cb {cb} cr {cr}");
                        }
                    }
                }
            }
        }
    }
}
