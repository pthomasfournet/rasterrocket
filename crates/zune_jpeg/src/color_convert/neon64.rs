/*
 * Copyright (c) 2025.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Aarch64 color conversion routines
//! NEON is mandatory on aarch64.

#![cfg(all(feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint8x16_t, uint8x16x3_t, uint8x16x4_t, vaddq_s16, vcombine_s16,
    vcombine_u8, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, vget_low_s16, vld1q_s16, vmlal_high_n_s16,
    vmlal_n_s16, vqmovun_s16, vshlq_n_s16, vshrn_n_s32, vst3q_u8, vst4q_u8, vsubq_s16,
};

use crate::color_convert::scalar::{CB_FRAC, CR_FRAC, C_G_CB_FRAC, C_G_CR_FRAC, YUV_RND};

/// `(bias + a * ka + b * kb) >> 16` per lane, narrowed to 16 bits.
#[inline(always)]
unsafe fn fraction(bias: int32x4_t, a: int16x8_t, ka: i16, b: int16x8_t, kb: i16) -> int16x8_t {
    unsafe {
        let low = vmlal_n_s16(vmlal_n_s16(bias, vget_low_s16(a), ka), vget_low_s16(b), kb);
        let high = vmlal_high_n_s16(vmlal_high_n_s16(bias, a, ka), b, kb);
        vcombine_s16(vshrn_n_s32::<16>(low), vshrn_n_s32::<16>(high))
    }
}

/// Eight samples to RGB: the whole part of each coefficient is applied as
/// additions and its 16-bit fraction as a multiply-accumulate, with the
/// arithmetic of the scalar conversion.
#[inline(always)]
unsafe fn ycbcr_to_rgb_8(
    y: int16x8_t,
    cb: int16x8_t,
    cr: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
    unsafe {
        let bias = vdupq_n_s32(YUV_RND);
        let bias16 = vdupq_n_s16(128);
        let cb = vsubq_s16(cb, bias16);
        let cr = vsubq_s16(cr, bias16);
        let r = vaddq_s16(vaddq_s16(y, cr), fraction(bias, cr, CR_FRAC, cb, 0));
        let g = vaddq_s16(
            vsubq_s16(y, cr),
            fraction(bias, cr, C_G_CR_FRAC, cb, C_G_CB_FRAC),
        );
        let b = vaddq_s16(
            vaddq_s16(y, vshlq_n_s16::<1>(cb)),
            fraction(bias, cb, CB_FRAC, cr, 0),
        );
        (r, g, b)
    }
}

#[inline(always)]
unsafe fn ycbcr_to_rgb_baseline_no_clamp(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    unsafe {
        let (r0, g0, b0) = ycbcr_to_rgb_8(
            vld1q_s16(y.as_ptr()),
            vld1q_s16(cb.as_ptr()),
            vld1q_s16(cr.as_ptr()),
        );
        let (r1, g1, b1) = ycbcr_to_rgb_8(
            vld1q_s16(y[8..].as_ptr()),
            vld1q_s16(cb[8..].as_ptr()),
            vld1q_s16(cr[8..].as_ptr()),
        );
        // Signed to unsigned narrowing saturates to 0..=255.
        (
            vcombine_u8(vqmovun_s16(r0), vqmovun_s16(r1)),
            vcombine_u8(vqmovun_s16(g0), vqmovun_s16(g1)),
            vcombine_u8(vqmovun_s16(b0), vqmovun_s16(b1)),
        )
    }
}

#[inline(always)]
pub fn ycbcr_to_rgb_neon(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    out: &mut [u8],
    offset: &mut usize,
) {
    // check if we have enough space to write.
    let out: &mut [u8; 48] = out
        .get_mut(*offset..*offset + 48)
        .expect("Slice to small cannot write")
        .try_into()
        .unwrap();

    unsafe {
        let (r, g, b) = ycbcr_to_rgb_baseline_no_clamp(y, cb, cr);
        vst3q_u8(out.as_mut_ptr(), uint8x16x3_t(r, g, b));
    }
    *offset += 48;
}

#[inline(always)]
pub fn ycbcr_to_rgba_neon(
    y: &[i16; 16],
    cb: &[i16; 16],
    cr: &[i16; 16],
    out: &mut [u8],
    offset: &mut usize,
) {
    // check if we have enough space to write.
    let out: &mut [u8; 64] = out
        .get_mut(*offset..*offset + 64)
        .expect("Slice to small cannot write")
        .try_into()
        .unwrap();

    unsafe {
        let (r, g, b) = ycbcr_to_rgb_baseline_no_clamp(y, cb, cr);
        vst4q_u8(out.as_mut_ptr(), uint8x16x4_t(r, g, b, vdupq_n_u8(255)));
    }
    *offset += 64;
}
