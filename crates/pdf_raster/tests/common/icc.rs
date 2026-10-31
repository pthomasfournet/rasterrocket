//! Minimal ICC v2 profiles assembled in memory: matrix/TRC display
//! profiles with a pure power-law tone curve on sRGB primaries.

/// `u8Fixed8Number` 1.8.
pub(crate) const GAMMA_1_8: u16 = 461;

/// sRGB primaries adapted to D50, as `s15Fixed16Number` XYZ columns.
const SRGB_D50_COLORANTS: [[f64; 3]; 3] = [
    [0.436_065_7, 0.222_488_4, 0.013_916_0],
    [0.385_147_1, 0.716_873_2, 0.097_076_4],
    [0.143_066_4, 0.060_607_9, 0.714_096_1],
];
const D50: [f64; 3] = [0.964_2, 1.0, 0.824_9];

fn s15f16(v: f64) -> [u8; 4] {
    #[expect(clippy::cast_possible_truncation, reason = "|v| < 2; fits s15Fixed16")]
    let fixed = (v * 65_536.0).round() as i32;
    fixed.to_be_bytes()
}

fn xyz_tag(xyz: [f64; 3]) -> Vec<u8> {
    let mut t = b"XYZ \0\0\0\0".to_vec();
    for v in xyz {
        t.extend_from_slice(&s15f16(v));
    }
    t
}

/// A one-entry `curv`: a pure power law; padded to a 4-byte boundary.
fn gamma_curve(gamma: u16) -> Vec<u8> {
    let mut t = b"curv\0\0\0\0".to_vec();
    t.extend_from_slice(&1u32.to_be_bytes());
    t.extend_from_slice(&gamma.to_be_bytes());
    t.extend_from_slice(&[0, 0]);
    t
}

/// A minimal ICC v2 display profile with the given colour space signature
/// and `(signature, data)` tags.
fn icc_profile(space: [u8; 4], tags: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let table_len = 4 + 12 * tags.len();
    let mut data = Vec::new();
    let mut table = Vec::new();
    table.extend_from_slice(&u32::try_from(tags.len()).unwrap().to_be_bytes());
    for (sig, body) in tags {
        let offset = 128 + table_len + data.len();
        table.extend_from_slice(*sig);
        table.extend_from_slice(&u32::try_from(offset).unwrap().to_be_bytes());
        table.extend_from_slice(&u32::try_from(body.len()).unwrap().to_be_bytes());
        data.extend_from_slice(body);
        while data.len() % 4 != 0 {
            data.push(0);
        }
    }
    let size = 128 + table.len() + data.len();
    let mut header = Vec::with_capacity(128);
    header.extend_from_slice(&u32::try_from(size).unwrap().to_be_bytes());
    header.extend_from_slice(b"none");
    header.extend_from_slice(&0x0210_0000u32.to_be_bytes());
    header.extend_from_slice(b"mntr");
    header.extend_from_slice(&space);
    header.extend_from_slice(b"XYZ ");
    header.extend_from_slice(&[0; 12]); // creation date
    header.extend_from_slice(b"acsp");
    header.extend_from_slice(&[0; 24]); // platform, flags, device, attributes
    header.extend_from_slice(&[0; 4]); // perceptual intent
    for v in D50 {
        header.extend_from_slice(&s15f16(v));
    }
    header.resize(128, 0);
    [header, table, data].concat()
}

/// An RGB profile with gamma `gamma` (`u8Fixed8Number`) on sRGB primaries.
pub(crate) fn rgb_profile(gamma: u16) -> Vec<u8> {
    let [r, g, b] = SRGB_D50_COLORANTS;
    icc_profile(
        *b"RGB ",
        &[
            (b"wtpt", xyz_tag(D50)),
            (b"rXYZ", xyz_tag(r)),
            (b"gXYZ", xyz_tag(g)),
            (b"bXYZ", xyz_tag(b)),
            (b"rTRC", gamma_curve(gamma)),
            (b"gTRC", gamma_curve(gamma)),
            (b"bTRC", gamma_curve(gamma)),
        ],
    )
}

/// A gray profile with gamma `gamma` (`u8Fixed8Number`).
pub(crate) fn gray_profile(gamma: u16) -> Vec<u8> {
    icc_profile(
        *b"GRAY",
        &[(b"wtpt", xyz_tag(D50)), (b"kTRC", gamma_curve(gamma))],
    )
}
