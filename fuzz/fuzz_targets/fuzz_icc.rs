#![no_main]

//! ICC profiles, as a page's `/ICCBased` source profile and as the
//! session's output profile (`rrocket --output-profile`): tag parsing, the
//! matrix, curve and lookup-table conversions, their tabulated grids and
//! the links between CMYK profiles.
//!
//! Layout: `[space, flags]`, then the profile.  `space` selects the
//! component count of the `/ICCBased` space (1, 3 or 4); bit 0 of `flags`
//! loads the profile as the output profile as well.  The page fills with
//! the profile's space and the device spaces and draws an 8 × 8 image in
//! the profile's space.

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use rasterrocket::OutputProfile;
use rasterrocket_fuzz::common::stream_object;

const COMPONENTS: [usize; 3] = [1, 3, 4];

fuzz_target!(|data: &[u8]| {
    let Some(([space, flags], profile)) = rasterrocket_fuzz::split_selectors::<2>(data) else {
        return;
    };
    let n = COMPONENTS[usize::from(space) % COMPONENTS.len()];
    let mut config = rasterrocket_fuzz::cpu_config();
    if flags & 1 != 0 {
        let Ok(output) = OutputProfile::from_bytes(profile) else {
            return;
        };
        config.output_profile = Some(Arc::new(output));
    }
    let tint = ["0.2", "0.4", "0.6", "0.8"][..n].join(" ");
    let content = format!(
        "/CS0 cs {tint} sc 0 0 50 50 re f 0.1 0.5 0.9 rg 50 0 50 50 re f \
         0.1 0.2 0.3 0.4 k 0 50 50 50 re f 0.5 g 50 50 50 50 re f \
         q 100 0 0 100 0 0 cm /Im1 Do Q"
    );
    let image: Vec<u8> = (0..=u8::MAX).cycle().step_by(37).take(64 * n).collect();
    rasterrocket_fuzz::render_page_with(
        "<< /ColorSpace << /CS0 [/ICCBased 6 0 R] >> /XObject << /Im1 5 0 R >> >>",
        content.as_bytes(),
        vec![
            stream_object(
                "/Type /XObject /Subtype /Image /Width 8 /Height 8 \
                 /ColorSpace [/ICCBased 6 0 R] /BitsPerComponent 8",
                &image,
            ),
            stream_object(&format!("/N {n}"), profile),
        ],
        &config,
    );
});
