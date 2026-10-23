//! The irreversible multi-component transformation, as specified in
//! Annex G.2 and G.3.

use alloc::vec;
use alloc::vec::Vec;

use super::ComponentData;
use super::codestream::{Header, WaveletTransform};
use crate::error::{ColorError, Result, bail, err};
use crate::math::{self, Level, Simd, dispatch, f32x8};

/// An inverse MCT that has been validated but not yet applied, together with
/// the DC level shift of each of the first three components.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingMct {
    pub(crate) transform: WaveletTransform,
    pub(crate) offsets: [f32; 3],
}

/// Validate that the inverse multi-component transform (G.2, G.3) can be
/// applied to `components`, and return the transform to use.
///
/// `None` means the transform is skipped (fewer than three components in
/// non-strict mode).
pub(crate) fn inverse_transform(
    components: &[ComponentData],
    component_infos: &[super::codestream::ComponentInfo],
    header: &Header,
) -> Result<Option<WaveletTransform>> {
    if components.len() < 3 {
        return if header.strict {
            err!(ColorError::Mct)
        } else {
            Ok(None)
        };
    }

    let transform = component_infos[0].wavelet_transform();

    if transform != component_infos[1].wavelet_transform()
        || component_infos[1].wavelet_transform() != component_infos[2].wavelet_transform()
    {
        bail!(ColorError::Mct);
    }

    let len = components[0].container.len();
    if components[1].container.len() != len || components[2].container.len() != len {
        bail!(ColorError::Mct);
    }

    Ok(Some(transform))
}

/// Apply `pending` in place to the first three components: each sample
/// becomes its inverse-MCT value plus the component's DC level shift.
pub(crate) fn apply_inverse(components: &mut [ComponentData], pending: PendingMct) {
    let (s, _) = components.split_at_mut(3);
    let [s0, s1, s2] = s else { unreachable!() };
    dispatch!(Level::new(), simd => apply_inner_impl(
        simd,
        pending,
        &mut s0.container,
        &mut s1.container,
        &mut s2.container,
    ));
}

/// Apply `pending` to the first three components and write the result as
/// interleaved 8-bit RGB, rounding each sample as `DecodedImage::data_u8`
/// does.  The components are left unchanged.
///
/// Every sample goes through the same operations as [`apply_inverse`]
/// followed by `data_u8`, so the output is identical; the transformed
/// samples are only never written back to the component buffers.
pub(crate) fn inverse_into_rgb8(components: &[ComponentData], pending: PendingMct) -> Vec<u8> {
    let pixels = components[0].container.truncated().len();
    let mut out = vec![0_u8; pixels * 3];
    dispatch!(Level::new(), simd => into_rgb8_impl(
        simd,
        pending,
        &components[0].container,
        &components[1].container,
        &components[2].container,
        &mut out,
    ));
    out
}

#[inline(always)]
fn apply_inner_impl<S: Simd>(
    simd: S,
    pending: PendingMct,
    s0: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
) {
    for ((y0, y1), y2) in s0
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(s1.as_chunks_mut::<8>().0)
        .zip(s2.as_chunks_mut::<8>().0)
    {
        let [i0, i1, i2] = inverse_chunk(
            simd,
            pending,
            f32x8::from_slice(simd, y0),
            f32x8::from_slice(simd, y1),
            f32x8::from_slice(simd, y2),
        );
        i0.store(y0);
        i1.store(y1);
        i2.store(y2);
    }
}

#[inline(always)]
fn into_rgb8_impl<S: Simd>(
    simd: S,
    pending: PendingMct,
    s0: &[f32],
    s1: &[f32],
    s2: &[f32],
    out: &mut [u8],
) {
    for (((y0, y1), y2), rgb) in s0
        .as_chunks::<8>()
        .0
        .iter()
        .zip(s1.as_chunks::<8>().0)
        .zip(s2.as_chunks::<8>().0)
        .zip(out.chunks_mut(8 * 3))
    {
        let [i0, i1, i2] = inverse_chunk(
            simd,
            pending,
            f32x8::from_slice(simd, y0),
            f32x8::from_slice(simd, y1),
            f32x8::from_slice(simd, y2),
        );
        let (mut r, mut g, mut b) = ([0.0_f32; 8], [0.0_f32; 8], [0.0_f32; 8]);
        i0.store(&mut r);
        i1.store(&mut g);
        i2.store(&mut b);
        for (k, px) in rgb.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *px = [
                math::round_to_u8(r[k]),
                math::round_to_u8(g[k]),
                math::round_to_u8(b[k]),
            ];
        }
    }
}

/// The inverse MCT of one chunk of eight samples per component, plus the DC
/// level shifts.
#[inline(always)]
fn inverse_chunk<S: Simd>(
    simd: S,
    pending: PendingMct,
    y_0: f32x8<S>,
    y_1: f32x8<S>,
    y_2: f32x8<S>,
) -> [f32x8<S>; 3] {
    let o0 = f32x8::splat(simd, pending.offsets[0]);
    let o1 = f32x8::splat(simd, pending.offsets[1]);
    let o2 = f32x8::splat(simd, pending.offsets[2]);
    let (i0, i1, i2) = match pending.transform {
        // Irreversible MCT, specified in G.3.
        WaveletTransform::Irreversible97 => (
            y_2.mul_add(f32x8::splat(simd, 1.402), y_0),
            y_2.mul_add(
                f32x8::splat(simd, -0.71414),
                y_1.mul_add(f32x8::splat(simd, -0.34413), y_0),
            ),
            y_1.mul_add(f32x8::splat(simd, 1.772), y_0),
        ),
        // Reversible MCT, specified in G.2.
        WaveletTransform::Reversible53 => {
            let i1 = y_0 - ((y_2 + y_1) * 0.25).floor();
            (y_2 + i1, i1, y_1 + i1)
        }
    };
    [i0 + o0, i1 + o1, i2 + o2]
}
