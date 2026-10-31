//! Cipher and digest primitives of the Standard security handler.
//!
//! Thin wrappers over the `RustCrypto` crates with the byte-level conventions
//! ISO 32000-2 §7.6 attaches to each use: a leading IV and PKCS#5 padding on
//! AES-CBC data, no IV and no padding on the AES operations of the R5/R6
//! key schedule.  Every function rejects or trims a malformed input instead
//! of panicking.

use aes::{Aes128, Aes256};
use cbc::cipher::{BlockModeDecrypt, BlockModeEncrypt, KeyIvInit, block_padding::NoPadding};
use md5::Md5;
use rc4::{KeyInit, Rc4, StreamCipher};
use sha2::{Digest, Sha256, Sha384, Sha512};

/// AES block (and IV) length in bytes.
pub(super) const AES_BLOCK: usize = 16;

/// MD5 of the concatenation of `parts`.
pub(super) fn md5(parts: &[&[u8]]) -> [u8; 16] {
    let mut h = Md5::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

/// SHA-256 of the concatenation of `parts`.
pub(super) fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

/// Whether `a` and `b` are equal, in time that depends only on their
/// lengths.
///
/// Used where a value derived from the password is compared with an entry
/// of the `/Encrypt` dictionary, so the comparison time does not reveal how
/// many leading bytes of a candidate hash match.
pub(super) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && std::hint::black_box(a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y))) == 0
}

/// Apply the RC4 keystream of `key` to `data` in place.
///
/// # Errors
/// A key outside RC4's 1–256 byte range.
pub(super) fn rc4_in_place(key: &[u8], data: &mut [u8]) -> Result<(), String> {
    let mut cipher = Rc4::new_from_slice(key)
        .map_err(|_| format!("RC4 key of {} bytes is outside 1..=256", key.len()))?;
    cipher.apply_keystream(data);
    Ok(())
}

/// RC4 of `data` under `key`, as a new buffer.
///
/// # Errors
/// See [`rc4_in_place`].
pub(super) fn rc4(key: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = data.to_vec();
    rc4_in_place(key, &mut out)?;
    Ok(out)
}

/// AES-CBC decryption of `data`, whose first 16 bytes are the IV, with the
/// PKCS#5 padding of the last block removed (ISO 32000-2 §7.6.3.1).
///
/// A trailing partial block is dropped with a warning: the bytes it holds
/// cannot be recovered without their complete block.  A final byte outside
/// 1..=16 is not valid padding; the block is then kept whole, with a
/// warning, rather than losing data to a producer's padding fault.
///
/// # Errors
/// `data` is shorter than the IV, or `key` is neither 16 (AES-128) nor
/// 32 bytes (AES-256).
pub(super) fn aes_cbc_decrypt(key: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < AES_BLOCK {
        return Err(format!(
            "AES ciphertext of {} bytes is shorter than its 16-byte IV",
            data.len()
        ));
    }
    let (iv, body) = data.split_at(AES_BLOCK);
    let whole = body.len() - body.len() % AES_BLOCK;
    if whole != body.len() {
        log::warn!(
            "AES ciphertext of {} bytes is not a whole number of blocks; \
             dropping the trailing {} bytes",
            body.len(),
            body.len() - whole
        );
    }
    let body = &body[..whole];
    let mut out = match key.len() {
        16 => cbc::Decryptor::<Aes128>::new_from_slices(key, iv)
            .map_err(|e| e.to_string())?
            .decrypt_padded_vec::<NoPadding>(body),
        32 => cbc::Decryptor::<Aes256>::new_from_slices(key, iv)
            .map_err(|e| e.to_string())?
            .decrypt_padded_vec::<NoPadding>(body),
        n => return Err(format!("AES key of {n} bytes is neither 16 nor 32")),
    }
    .map_err(|e| e.to_string())?;
    strip_pkcs5(&mut out);
    Ok(out)
}

/// Remove PKCS#5 padding: the final byte `p` in 1..=16 names the number of
/// padding bytes.  Any other final byte leaves `buf` unchanged with a
/// warning.
fn strip_pkcs5(buf: &mut Vec<u8>) {
    match buf.last().copied() {
        Some(p) if (1..=16).contains(&p) && usize::from(p) <= buf.len() => {
            buf.truncate(buf.len() - usize::from(p));
        }
        Some(p) => log::warn!("AES plaintext ends in {p:#04x}, not PKCS#5 padding; keeping it"),
        None => {}
    }
}

/// AES-128-CBC encryption without padding (`data.len()` must be a multiple
/// of 16), as used by the SHA-2 iteration of Algorithm 2.B.
///
/// # Errors
/// `key` or `iv` is not 16 bytes, or `data` is not block-aligned.
pub(super) fn aes128_cbc_encrypt_no_pad(
    key: &[u8],
    iv: &[u8],
    data: &[u8],
) -> Result<Vec<u8>, String> {
    let enc = cbc::Encryptor::<Aes128>::new_from_slices(key, iv).map_err(|e| e.to_string())?;
    let mut buf = data.to_vec();
    let len = enc
        .encrypt_padded::<NoPadding>(&mut buf, data.len())
        .map_err(|e| e.to_string())?
        .len();
    buf.truncate(len);
    Ok(buf)
}

/// AES-256-CBC decryption with a zero IV and no padding, as used to unwrap
/// the file key from `/UE` and `/OE` (Algorithm 2.A).
///
/// # Errors
/// `key` is not 32 bytes or `data` is not block-aligned.
pub(super) fn aes256_cbc_decrypt_no_pad_zero_iv(
    key: &[u8],
    data: &[u8],
) -> Result<Vec<u8>, String> {
    let dec = cbc::Decryptor::<Aes256>::new_from_slices(key, &[0u8; AES_BLOCK])
        .map_err(|e| e.to_string())?;
    dec.decrypt_padded_vec::<NoPadding>(data)
        .map_err(|e| e.to_string())
}

/// Algorithm 2.B of ISO 32000-2 §7.6.4.3.4 (revision 6): the password hash
/// built by iterating AES-128-CBC and SHA-256/384/512 over
/// `password ‖ salt ‖ udata` until at least 64 rounds have run and the last
/// byte of the round's ciphertext is at most `rounds − 32`.
///
/// `udata` is the 48-byte `/U` string when hashing an owner password and
/// empty otherwise.
///
/// # Errors
/// An AES failure, which the fixed-size inputs here cannot produce.
pub(super) fn hash_r6(password: &[u8], salt: &[u8], udata: &[u8]) -> Result<[u8; 32], String> {
    let mut k: Vec<u8> = sha256(&[password, salt, udata]).to_vec();
    let mut rounds = 0usize;
    loop {
        let mut k1 = Vec::with_capacity(64 * (password.len() + k.len() + udata.len()));
        for _ in 0..64 {
            k1.extend_from_slice(password);
            k1.extend_from_slice(&k);
            k1.extend_from_slice(udata);
        }
        let e = aes128_cbc_encrypt_no_pad(&k[..16], &k[16..32], &k1)?;
        // 256 ≡ 1 (mod 3), so the first 16 bytes as a big-endian number are
        // congruent to their byte sum.
        let selector = e[..16].iter().map(|&b| u32::from(b)).sum::<u32>() % 3;
        k = match selector {
            0 => sha256(&[&e]).to_vec(),
            1 => Sha384::digest(&e).to_vec(),
            _ => Sha512::digest(&e).to_vec(),
        };
        rounds += 1;
        let last = e.last().copied().map_or(0, usize::from);
        if rounds >= 64 && last <= rounds - 32 {
            break;
        }
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&k[..32]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
            .collect()
    }

    #[test]
    fn rc4_rfc6229_40_bit_vector() {
        // RFC 6229, key 0102030405: first 16 keystream bytes.
        let ks = rc4(&hex("0102030405"), &[0u8; 16]).expect("rc4");
        assert_eq!(ks, hex("b2396305f03dc027ccc3524a0a1118a8"));
    }

    #[test]
    fn rc4_rfc6229_128_bit_vector() {
        let key = hex("0102030405060708090a0b0c0d0e0f10");
        let ks = rc4(&key, &[0u8; 16]).expect("rc4");
        assert_eq!(ks, hex("9ac7cc9a609d1ef7b2932899cde41b97"));
    }

    #[test]
    fn rc4_rejects_empty_key() {
        assert!(rc4(&[], b"abc").is_err());
        assert!(rc4(&[0u8; 257], b"abc").is_err());
    }

    #[test]
    fn aes_cbc_round_trip_strips_padding() {
        // Encrypt with the no-pad primitive after padding by hand, then decrypt
        // through the IV-prefixed path.
        let key = [7u8; 16];
        let iv = [3u8; 16];
        let mut msg = b"seventeen bytes!!".to_vec(); // 17 bytes → 15 bytes of 0x0f
        msg.extend(std::iter::repeat_n(0x0f, 15));
        let ct = aes128_cbc_encrypt_no_pad(&key, &iv, &msg).expect("encrypt");
        let mut data = iv.to_vec();
        data.extend_from_slice(&ct);
        let pt = aes_cbc_decrypt(&key, &data).expect("decrypt");
        assert_eq!(pt, b"seventeen bytes!!");
    }

    #[test]
    fn aes_cbc_rejects_short_and_odd_input() {
        assert!(aes_cbc_decrypt(&[0u8; 16], &[0u8; 15]).is_err());
        assert!(aes_cbc_decrypt(&[0u8; 24], &[0u8; 32]).is_err());
        // IV only: zero blocks of ciphertext decrypt to nothing.
        assert_eq!(
            aes_cbc_decrypt(&[0u8; 16], &[0u8; 16]).expect("iv only"),
            b""
        );
        // A trailing partial block is dropped, not a panic.
        let pt = aes_cbc_decrypt(&[0u8; 32], &[0u8; 40]).expect("partial");
        assert!(pt.len() <= 16);
    }

    #[test]
    fn ct_eq_compares_whole_slices() {
        assert!(ct_eq(b"", b""));
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abz"));
        assert!(!ct_eq(b"xbc", b"abc"));
        assert!(!ct_eq(b"abc", b"abcd"));
    }

    #[test]
    fn bad_padding_byte_is_kept() {
        let mut buf = vec![1, 2, 3, 0x40];
        strip_pkcs5(&mut buf);
        assert_eq!(buf, [1, 2, 3, 0x40]);
        let mut buf = vec![1, 2, 3, 0x00];
        strip_pkcs5(&mut buf);
        assert_eq!(buf.len(), 4);
        let mut buf = vec![1, 2, 3, 0x05];
        strip_pkcs5(&mut buf);
        assert_eq!(buf.len(), 4);
    }

    #[test]
    fn hash_r6_is_deterministic_and_32_bytes() {
        let a = hash_r6(b"user", &[1u8; 8], &[]).expect("hash");
        let b = hash_r6(b"user", &[1u8; 8], &[]).expect("hash");
        assert_eq!(a, b);
        assert_ne!(a, hash_r6(b"user", &[2u8; 8], &[]).expect("hash"));
        assert_ne!(a, hash_r6(b"user", &[1u8; 8], &[0u8; 48]).expect("hash"));
    }

    #[test]
    fn unwrap_key_rejects_bad_lengths() {
        assert!(aes256_cbc_decrypt_no_pad_zero_iv(&[0u8; 16], &[0u8; 32]).is_err());
        assert!(aes256_cbc_decrypt_no_pad_zero_iv(&[0u8; 32], &[0u8; 31]).is_err());
        assert_eq!(
            aes256_cbc_decrypt_no_pad_zero_iv(&[0u8; 32], &[0u8; 32])
                .expect("aligned")
                .len(),
            32
        );
    }
}
