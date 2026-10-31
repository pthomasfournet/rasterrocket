//! The Standard security handler's `/Encrypt` dictionary and password
//! algorithms (ISO 32000-2 §7.6.4).
//!
//! [`StandardSecurity::parse`] validates the dictionary once;
//! [`StandardSecurity::authenticate`] turns a password into the file
//! encryption key, trying it as the user password and then as the owner
//! password.  Revisions 2–4 derive the key with MD5 and RC4 (Algorithms
//! 2–7); revisions 5 and 6 unwrap it with SHA-2 and AES-256 (Algorithms 2.A,
//! 2.B, 11 and 12).

use std::collections::HashMap;

use super::{
    Cipher,
    primitives::{aes256_cbc_decrypt_no_pad_zero_iv, hash_r6, md5, rc4, sha256},
};
use crate::{dictionary::Dictionary, object::Object};

/// The 32-byte padding string of Algorithm 2 (ISO 32000-2 §7.6.4.3.2).
const PAD: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

/// Longest password the revision 5/6 algorithms hash, in UTF-8 bytes.
const MAX_R6_PASSWORD: usize = 127;

/// The validated contents of a Standard security handler dictionary.
#[derive(Debug)]
pub(super) struct StandardSecurity {
    /// `/R`, 2..=6.
    revision: u8,
    /// File key length in bytes: 5..=16 for RC4 and AESV2, 32 for AESV3.
    key_len: usize,
    /// `/O`: 32 bytes (R ≤ 4) or 48 bytes (R ≥ 5).
    o: Vec<u8>,
    /// `/U`: 32 bytes (R ≤ 4) or 48 bytes (R ≥ 5).
    u: Vec<u8>,
    /// `/OE`, 32 bytes; empty for R ≤ 4.
    oe: Vec<u8>,
    /// `/UE`, 32 bytes; empty for R ≤ 4.
    ue: Vec<u8>,
    /// `/P` as the low 32 bits of its two's-complement value, little-endian.
    p: [u8; 4],
    /// `/EncryptMetadata`, default `true`.
    pub(super) encrypt_metadata: bool,
    /// First element of the trailer `/ID` array; empty when absent.
    id0: Vec<u8>,
    /// Cipher named by `/StmF`.
    pub(super) stream_cipher: Cipher,
    /// Cipher named by `/StrF`.
    pub(super) string_cipher: Cipher,
    /// Every crypt filter of `/CF` by name, for per-stream `/Crypt` filters.
    pub(super) filters: HashMap<Vec<u8>, Cipher>,
}

impl StandardSecurity {
    /// Validate an `/Encrypt` dictionary.  `id0` is the first string of the
    /// trailer `/ID` array (empty when the trailer has none).
    ///
    /// # Errors
    /// A description of the first unsupported or malformed entry: a
    /// `/Filter` other than `Standard`, an unknown `/V` or `/R`, a key
    /// length outside its cipher's range, an unknown crypt filter method,
    /// or an `/O`, `/U`, `/OE` or `/UE` string too short for the revision.
    pub(super) fn parse(dict: &Dictionary, id0: &[u8]) -> Result<Self, String> {
        let filter = dict.get(b"Filter").and_then(Object::as_name);
        if filter != Some(b"Standard") {
            return Err(format!(
                "security handler /{} is not supported (only /Standard is)",
                filter.map_or_else(
                    || "?".to_owned(),
                    |f| String::from_utf8_lossy(f).into_owned()
                )
            ));
        }
        let version = dict.get(b"V").and_then(Object::as_i64).unwrap_or(0);
        let revision = match dict.get(b"R").and_then(Object::as_i64).unwrap_or(0) {
            r @ 2..=6 => u8::try_from(r).unwrap_or(0),
            r => return Err(format!("unsupported revision /R {r}")),
        };
        if !(1..=5).contains(&version) {
            return Err(format!("unsupported algorithm /V {version}"));
        }
        let encrypt_metadata = dict
            .get(b"EncryptMetadata")
            .and_then(Object::as_bool)
            .unwrap_or(true);

        let ciphers = Ciphers::parse(dict, version)?;
        let key_len = key_length(dict, version, revision, &ciphers)?;
        let (o, u, oe, ue) = key_entries(dict, revision)?;
        // Two's-complement low 32 bits, whatever width the producer wrote.
        let perms = dict.get(b"P").and_then(Object::as_i64).unwrap_or(-1) & 0xFFFF_FFFF;
        let p = u32::try_from(perms).map_or([0xFF; 4], u32::to_le_bytes);

        Ok(Self {
            revision,
            key_len,
            o,
            u,
            oe,
            ue,
            p,
            encrypt_metadata,
            id0: id0.to_vec(),
            stream_cipher: ciphers.stream,
            string_cipher: ciphers.string,
            filters: ciphers.by_name,
        })
    }

    /// The file encryption key unlocked by `password`, tried first as the
    /// user password and then as the owner password; `None` when it is
    /// neither.
    ///
    /// Revisions 2–4 take the password in `PDFDocEncoding`: the UTF-8 bytes
    /// are tried first and, for a password of Latin-1 characters, their
    /// single-byte form second.  Revisions 5 and 6 take UTF-8, truncated to
    /// 127 bytes (`SASLprep` normalisation is not applied).
    pub(super) fn authenticate(&self, password: &str) -> Option<Vec<u8>> {
        for candidate in self.password_encodings(password) {
            if let Some(key) = self.check_user(&candidate) {
                return Some(key);
            }
            if !candidate.is_empty()
                && let Some(key) = self.check_owner(&candidate)
            {
                return Some(key);
            }
        }
        None
    }

    fn password_encodings(&self, password: &str) -> Vec<Vec<u8>> {
        let utf8 = password.as_bytes();
        if self.revision >= 5 {
            let mut end = utf8.len().min(MAX_R6_PASSWORD);
            while !password.is_char_boundary(end) {
                end -= 1;
            }
            return vec![utf8[..end].to_vec()];
        }
        let mut out = vec![utf8.to_vec()];
        if !password.is_ascii()
            && let Ok(latin1) = password
                .chars()
                .map(|c| u8::try_from(u32::from(c)))
                .collect::<Result<Vec<u8>, _>>()
        {
            out.push(latin1);
        }
        out
    }

    // ── Revision 2–4 (MD5 / RC4) ─────────────────────────────────────────────

    /// Algorithm 2: the file key for a (user) password.
    fn compute_key_legacy(&self, password: &[u8]) -> Vec<u8> {
        let padded = pad_password(password);
        let no_metadata: &[u8] = if self.revision >= 4 && !self.encrypt_metadata {
            &[0xFF, 0xFF, 0xFF, 0xFF]
        } else {
            &[]
        };
        let mut hash = md5(&[&padded, &self.o, &self.p, &self.id0, no_metadata]);
        let n = self.key_len.min(16);
        if self.revision >= 3 {
            for _ in 0..50 {
                hash = md5(&[&hash[..n]]);
            }
        }
        hash[..n].to_vec()
    }

    /// Algorithms 4 and 5: `/U` recomputed from a candidate key.
    fn user_entry_matches(&self, key: &[u8]) -> bool {
        if self.revision == 2 {
            return rc4(key, &PAD).is_ok_and(|u| u == self.u);
        }
        let seed = md5(&[&PAD, &self.id0]);
        let Ok(mut x) = rc4(key, &seed) else {
            return false;
        };
        for i in 1..=19u8 {
            let round_key: Vec<u8> = key.iter().map(|b| b ^ i).collect();
            x = match rc4(&round_key, &x) {
                Ok(x) => x,
                Err(_) => return false,
            };
        }
        x[..16] == self.u[..16]
    }

    /// Algorithm 7: the user password that an owner password encrypts.
    fn user_password_from_owner(&self, owner: &[u8]) -> Option<Vec<u8>> {
        let mut hash = md5(&[&pad_password(owner)]);
        let n = self.key_len.min(16);
        if self.revision >= 3 {
            for _ in 0..50 {
                hash = md5(&[&hash[..n]]);
            }
        }
        let key = &hash[..n];
        if self.revision == 2 {
            return rc4(key, &self.o).ok();
        }
        let mut x = self.o.clone();
        for i in (0..=19u8).rev() {
            let round_key: Vec<u8> = key.iter().map(|b| b ^ i).collect();
            x = rc4(&round_key, &x).ok()?;
        }
        Some(x)
    }

    // ── Revision 5–6 (SHA-2 / AES-256) ───────────────────────────────────────

    /// The password hash of this revision: Algorithm 2.B for R6, a single
    /// SHA-256 for the R5 extension.
    fn hash_r5_r6(&self, password: &[u8], salt: &[u8], udata: &[u8]) -> Option<[u8; 32]> {
        if self.revision >= 6 {
            hash_r6(password, salt, udata).ok()
        } else {
            Some(sha256(&[password, salt, udata]))
        }
    }

    /// Algorithms 11 and 12 with the key unwrap of Algorithm 2.A.
    /// `entry` is `/U` or `/O`, `wrapped` the matching `/UE` or `/OE`, and
    /// `udata` the `/U` string for the owner check.
    fn check_r5_r6(
        &self,
        password: &[u8],
        entry: &[u8],
        wrapped: &[u8],
        udata: &[u8],
    ) -> Option<Vec<u8>> {
        let (hash, validation_salt, key_salt) = (&entry[..32], &entry[32..40], &entry[40..48]);
        if self.hash_r5_r6(password, validation_salt, udata)? != hash {
            return None;
        }
        let intermediate = self.hash_r5_r6(password, key_salt, udata)?;
        aes256_cbc_decrypt_no_pad_zero_iv(&intermediate, wrapped).ok()
    }

    // ── Dispatch ─────────────────────────────────────────────────────────────

    fn check_user(&self, password: &[u8]) -> Option<Vec<u8>> {
        if self.revision >= 5 {
            return self.check_r5_r6(password, &self.u, &self.ue, &[]);
        }
        let key = self.compute_key_legacy(password);
        self.user_entry_matches(&key).then_some(key)
    }

    fn check_owner(&self, password: &[u8]) -> Option<Vec<u8>> {
        if self.revision >= 5 {
            return self.check_r5_r6(password, &self.o, &self.oe, &self.u);
        }
        let user = self.user_password_from_owner(password)?;
        self.check_user(&user)
    }
}

/// `password` padded or truncated to 32 bytes with [`PAD`].
fn pad_password(password: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let n = password.len().min(32);
    out[..n].copy_from_slice(&password[..n]);
    out[n..].copy_from_slice(&PAD[..32 - n]);
    out
}

/// The ciphers an `/Encrypt` dictionary selects for streams and strings.
struct Ciphers {
    stream: Cipher,
    string: Cipher,
    /// Every crypt filter of `/CF` by name.
    by_name: HashMap<Vec<u8>, Cipher>,
    /// Key length declared by the stream filter (or the string filter when
    /// streams are `/Identity`), in bytes.
    key_len: Option<usize>,
}

impl Ciphers {
    /// `/V` 4 and 5 name crypt filters through `/StmF` and `/StrF`
    /// (default `/Identity`); earlier versions are RC4 throughout.
    fn parse(dict: &Dictionary, version: i64) -> Result<Self, String> {
        if version < 4 {
            return Ok(Self {
                stream: Cipher::Rc4,
                string: Cipher::Rc4,
                by_name: HashMap::new(),
                key_len: None,
            });
        }
        let filters = parse_crypt_filters(dict)?;
        let named = |key: &[u8]| -> Result<(Cipher, Option<usize>), String> {
            match dict.get(key).and_then(Object::as_name) {
                None | Some(b"Identity") => Ok((Cipher::Identity, None)),
                Some(name) => filters.get(name).copied().ok_or_else(|| {
                    format!(
                        "/{} names crypt filter /{} which /CF does not define",
                        String::from_utf8_lossy(key),
                        String::from_utf8_lossy(name)
                    )
                }),
            }
        };
        let (stream, stream_len) = named(b"StmF")?;
        let (string, string_len) = named(b"StrF")?;
        Ok(Self {
            stream,
            string,
            by_name: filters
                .into_iter()
                .map(|(name, (cipher, _))| (name, cipher))
                .collect(),
            key_len: stream_len.or(string_len),
        })
    }

    /// The cipher applied to document data: the stream cipher unless
    /// streams are `/Identity`.
    const fn data_cipher(&self) -> Cipher {
        if matches!(self.stream, Cipher::Identity) {
            self.string
        } else {
            self.stream
        }
    }
}

/// Crypt filters by name, with the key length each declares.
type CryptFilters = HashMap<Vec<u8>, (Cipher, Option<usize>)>;

/// The `/CF` dictionary.
///
/// `/Length` is in bytes per the specification, but producers also write
/// it in bits; a value of 40 or more is taken as bits, since no key is 40
/// bytes long.
fn parse_crypt_filters(dict: &Dictionary) -> Result<CryptFilters, String> {
    let mut out = HashMap::new();
    let Some(cf) = dict.get(b"CF").and_then(Object::as_dict) else {
        return Ok(out);
    };
    for (name, filter) in cf {
        let Some(filter) = filter.as_dict() else {
            continue;
        };
        let cipher = match filter.get(b"CFM").and_then(Object::as_name) {
            None | Some(b"None") => Cipher::Identity,
            Some(b"V2") => Cipher::Rc4,
            Some(b"AESV2") => Cipher::AesV2,
            Some(b"AESV3") => Cipher::AesV3,
            Some(other) => {
                return Err(format!(
                    "crypt filter /{} uses unsupported method /{}",
                    String::from_utf8_lossy(name),
                    String::from_utf8_lossy(other)
                ));
            }
        };
        let len = match filter.get(b"Length").and_then(Object::as_i64) {
            Some(n) if n >= 40 && n % 8 == 0 => usize::try_from(n / 8).ok(),
            Some(n) if (1..40).contains(&n) => usize::try_from(n).ok(),
            _ => None,
        };
        let _ = out.insert(name.clone(), (cipher, len));
    }
    Ok(out)
}

/// The file key length in bytes: fixed by `/V` 1 (40 bits) and `/V` 5
/// (256 bits); `/Length` bits for `/V` 2–3; for `/V` 4 the crypt filter's
/// length, falling back to `/Length`.  AES fixes the length of its key, and
/// revision 2 always uses 5 bytes (Algorithm 2, step h).
///
/// Revisions 2–4 derive at most 16 bytes from MD5 and revisions 5–6 unwrap
/// exactly 32, so a length outside that range for the revision (AESV3
/// under revision 4, say) is an error rather than a key that every object
/// would then fail to decrypt under.
fn key_length(
    dict: &Dictionary,
    version: i64,
    revision: u8,
    ciphers: &Ciphers,
) -> Result<usize, String> {
    let bits_to_bytes = |bits: i64| -> Result<usize, String> {
        if bits <= 0 || bits % 8 != 0 {
            return Err(format!("/Length {bits} is not a positive multiple of 8"));
        }
        usize::try_from(bits / 8).map_err(|_| format!("/Length {bits} out of range"))
    };
    let length = |default: i64| {
        dict.get(b"Length")
            .and_then(Object::as_i64)
            .unwrap_or(default)
    };
    let declared = match version {
        1 => 5,
        2 | 3 => bits_to_bytes(length(40))?,
        4 => match ciphers.key_len {
            Some(n) => n,
            None => bits_to_bytes(length(128))?,
        },
        _ => 32,
    };
    let key_len = match ciphers.data_cipher() {
        Cipher::AesV2 => 16,
        Cipher::AesV3 => 32,
        Cipher::Rc4 | Cipher::Identity if revision >= 5 => 32,
        Cipher::Rc4 | Cipher::Identity if revision == 2 => 5,
        Cipher::Rc4 | Cipher::Identity => declared,
    };
    let valid = if revision >= 5 {
        key_len == 32
    } else {
        (5..=16).contains(&key_len)
    };
    if !valid {
        return Err(format!(
            "key length of {key_len} bytes is not valid for revision {revision} \
             (5..=16 bytes for revisions 2-4, 32 for revisions 5-6)"
        ));
    }
    Ok(key_len)
}

/// `(/O, /U, /OE, /UE)` truncated to the lengths the revision uses: 32-byte
/// `/O` and `/U` for revisions 2–4; 48-byte `/O` and `/U` with 32-byte
/// `/OE` and `/UE` for 5 and 6.
#[expect(
    clippy::type_complexity,
    reason = "the four key entries of the dictionary"
)]
fn key_entries(
    dict: &Dictionary,
    revision: u8,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let string = |key: &[u8], min: usize| -> Result<Vec<u8>, String> {
        let s = dict.get(key).and_then(Object::as_str).unwrap_or(&[]);
        if s.len() < min {
            return Err(format!(
                "/{} is {} bytes, shorter than the {min} this revision requires",
                String::from_utf8_lossy(key),
                s.len()
            ));
        }
        Ok(s[..min].to_vec())
    };
    if revision >= 5 {
        Ok((
            string(b"O", 48)?,
            string(b"U", 48)?,
            string(b"OE", 32)?,
            string(b"UE", 32)?,
        ))
    } else {
        Ok((string(b"O", 32)?, string(b"U", 32)?, Vec::new(), Vec::new()))
    }
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

    fn dict(entries: &[(&str, Object)]) -> Dictionary {
        let mut d = Dictionary::new();
        for (k, v) in entries {
            d.set(k, v.clone());
        }
        d
    }

    fn name(s: &str) -> Object {
        Object::Name(s.as_bytes().to_vec())
    }

    fn string(hex_str: &str) -> Object {
        Object::String(hex(hex_str), crate::object::StringFormat::Hexadecimal)
    }

    // Vectors from qpdf 12.3.2 output (user password "user", owner password
    // "owner").  Each file carries its own trailer /ID; the legacy /O entry
    // depends only on the passwords and so is shared.
    const ID_R2: &str = "fb4be3c504372a9d0c2d8fb0878c27bd";
    const ID_R3: &str = "fa122ec5cfc26128f7329ea2d9be22aa";
    const ID_R4: &str = "8720f3758bfb794e58c89d1749b5d0b8";
    const ID_R4_CLEAR: &str = "01c0a6f37cc139639579c9b383616b0c";
    const O_LEGACY: &str = "0ba3835f88f90388e74e54584125ce142be0de24c6b0d37746e075b891756671";

    fn r3_dict() -> Dictionary {
        dict(&[
            ("Filter", name("Standard")),
            ("V", Object::Integer(2)),
            ("R", Object::Integer(3)),
            ("Length", Object::Integer(128)),
            ("P", Object::Integer(-4)),
            ("O", string(O_LEGACY)),
            (
                "U",
                string("6f463e7cdcc28067e7ece4e785e6000d0021446990b9e4114071a4d9104984c1"),
            ),
        ])
    }

    fn r6_dict() -> Dictionary {
        dict(&[
            ("Filter", name("Standard")),
            ("V", Object::Integer(5)),
            ("R", Object::Integer(6)),
            ("Length", Object::Integer(256)),
            ("P", Object::Integer(-4)),
            ("StmF", name("StdCF")),
            ("StrF", name("StdCF")),
            (
                "CF",
                Object::Dictionary(dict(&[(
                    "StdCF",
                    Object::Dictionary(dict(&[
                        ("CFM", name("AESV3")),
                        ("Length", Object::Integer(32)),
                    ])),
                )])),
            ),
            (
                "O",
                string(
                    "096faa52e288d9f6bd46d383557bba782e0d4bc2dec3b54ea9e6c58da5929c33\
                     e78a491bb733d2d353c496a2d876553e",
                ),
            ),
            (
                "OE",
                string("ddf9b7d8dddb65eb8da607d8605203e29843a88bdd480d44ad634f13d5682ba7"),
            ),
            ("U", string(R6_U)),
            ("UE", string(R6_UE)),
        ])
    }

    const R6_U: &str = "8ee88b06e0e292f24fad490dc6fdd7f4e24e60df82eb8dec37e070759b56e8ef\
                        d0099fb3d6f9d196ef9d363c2b38a8b7";
    const R6_UE: &str = "cbcac04c46705f9e9c064589c6919f1a465946086e79a255ddcdeeda25805cfb";

    #[test]
    fn r3_user_and_owner_passwords_unlock_the_same_key() {
        let s = StandardSecurity::parse(&r3_dict(), &hex(ID_R3)).expect("parse");
        assert_eq!(s.key_len, 16);
        assert_eq!(s.stream_cipher, Cipher::Rc4);
        let user = s.authenticate("user").expect("user password");
        let owner = s.authenticate("owner").expect("owner password");
        assert_eq!(user, owner);
        assert_eq!(user.len(), 16);
        assert!(s.authenticate("").is_none());
        assert!(s.authenticate("wrong").is_none());
    }

    #[test]
    fn r2_40_bit_vector() {
        let d = dict(&[
            ("Filter", name("Standard")),
            ("V", Object::Integer(1)),
            ("R", Object::Integer(2)),
            ("Length", Object::Integer(40)),
            ("P", Object::Integer(-4)),
            (
                "O",
                string("94e8094419662a774442fb072e3d9f19e9d130ec09a4d0061e78fe920f7ab62f"),
            ),
            (
                "U",
                string("0f03b854e40ebed0ca259cfc5f4875ca0444e2f9ff37c24ca837410a78c2cc78"),
            ),
        ]);
        let s = StandardSecurity::parse(&d, &hex(ID_R2)).expect("parse");
        assert_eq!(s.key_len, 5);
        let user = s.authenticate("user").expect("user password");
        assert_eq!(user, s.authenticate("owner").expect("owner password"));
        assert_eq!(user.len(), 5);
        assert!(s.authenticate("").is_none());
    }

    #[test]
    fn r4_aesv2_forces_16_byte_key_and_reads_crypt_filters() {
        let d = dict(&[
            ("Filter", name("Standard")),
            ("V", Object::Integer(4)),
            ("R", Object::Integer(4)),
            ("Length", Object::Integer(128)),
            ("P", Object::Integer(-4)),
            ("StmF", name("StdCF")),
            ("StrF", name("Identity")),
            (
                "CF",
                Object::Dictionary(dict(&[(
                    "StdCF",
                    Object::Dictionary(dict(&[
                        ("CFM", name("AESV2")),
                        ("Length", Object::Integer(128)),
                    ])),
                )])),
            ),
            ("O", string(O_LEGACY)),
            (
                "U",
                string("1993e5d85b5996461c27bb51c45000f30021446990b9e4114071a4d9104984c1"),
            ),
        ]);
        let s = StandardSecurity::parse(&d, &hex(ID_R4)).expect("parse");
        assert_eq!(s.key_len, 16);
        assert_eq!(s.stream_cipher, Cipher::AesV2);
        assert_eq!(s.string_cipher, Cipher::Identity);
        assert_eq!(s.filters.get(&b"StdCF"[..]), Some(&Cipher::AesV2));
        assert!(s.authenticate("user").is_some());
        assert!(s.authenticate("owner").is_some());
        assert!(s.authenticate("nope").is_none());
    }

    #[test]
    fn r4_without_encrypt_metadata_changes_the_key() {
        let mut d = r3_dict();
        d.set("V", Object::Integer(4));
        d.set("R", Object::Integer(4));
        d.set("StmF", name("StdCF"));
        d.set("StrF", name("StdCF"));
        d.set(
            "CF",
            Object::Dictionary(dict(&[(
                "StdCF",
                Object::Dictionary(dict(&[
                    ("CFM", name("AESV2")),
                    ("Length", Object::Integer(16)),
                ])),
            )])),
        );
        d.set(
            "U",
            string("be1599e3e2973381d8677b71cc4b4f340021446990b9e4114071a4d9104984c1"),
        );
        // With /EncryptMetadata true the key does not match this /U …
        assert!(
            StandardSecurity::parse(&d, &hex(ID_R4_CLEAR))
                .expect("parse")
                .authenticate("user")
                .is_none()
        );
        // … and with false it does.
        d.set("EncryptMetadata", Object::Boolean(false));
        let s = StandardSecurity::parse(&d, &hex(ID_R4_CLEAR)).expect("parse");
        assert!(!s.encrypt_metadata);
        assert!(s.authenticate("user").is_some());
    }

    #[test]
    fn r6_user_and_owner_passwords_unlock_the_same_key() {
        let s = StandardSecurity::parse(&r6_dict(), &[]).expect("parse");
        assert_eq!(s.key_len, 32);
        assert_eq!(s.stream_cipher, Cipher::AesV3);
        let user = s.authenticate("user").expect("user password");
        let owner = s.authenticate("owner").expect("owner password");
        assert_eq!(user, owner);
        assert_eq!(user.len(), 32);
        assert!(s.authenticate("").is_none());
        assert!(s.authenticate("wrong").is_none());
    }

    #[test]
    fn latin1_password_is_tried_for_legacy_revisions() {
        let s = StandardSecurity::parse(&r3_dict(), &hex(ID_R3)).expect("parse");
        assert_eq!(s.password_encodings("é").len(), 2);
        assert_eq!(s.password_encodings("é")[1], vec![0xE9]);
        assert_eq!(s.password_encodings("a").len(), 1);
        let s6 = StandardSecurity::parse(&r6_dict(), &[]).expect("parse");
        assert_eq!(s6.password_encodings("é"), vec![vec![0xC3, 0xA9]]);
        // 127-byte truncation lands on a char boundary.
        let long = "é".repeat(100);
        let enc = &s6.password_encodings(&long)[0];
        assert_eq!(enc.len(), 126);
    }

    // ── Hostile dictionaries ───────────────────────────────────────────────

    #[test]
    fn rejects_other_security_handlers() {
        let mut d = r3_dict();
        d.set("Filter", name("Adobe.PubSec"));
        let err = StandardSecurity::parse(&d, &[]).expect_err("must refuse");
        assert!(err.contains("Adobe.PubSec"));
        let _ = d.remove(b"Filter");
        assert!(StandardSecurity::parse(&d, &[]).is_err());
    }

    #[test]
    fn rejects_unknown_revision_and_version() {
        let mut d = r3_dict();
        d.set("R", Object::Integer(7));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        d.set("R", Object::Integer(3));
        d.set("V", Object::Integer(6));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        d.set("V", Object::Integer(-1));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let _ = d.remove(b"R");
        assert!(StandardSecurity::parse(&d, &[]).is_err());
    }

    #[test]
    fn rejects_bad_key_lengths() {
        for len in [0, 7, 32, 136, 1 << 40, -128] {
            let mut d = r3_dict();
            d.set("Length", Object::Integer(len));
            assert!(
                StandardSecurity::parse(&d, &[]).is_err(),
                "/Length {len} must be rejected"
            );
        }
        // A missing /Length defaults to 40 bits.
        let mut d = r3_dict();
        let _ = d.remove(b"Length");
        assert_eq!(StandardSecurity::parse(&d, &[]).expect("parse").key_len, 5);
    }

    /// Revision 2 keys are 5 bytes whatever `/Length` says; a key length
    /// that the revision cannot derive is refused; a crypt filter
    /// `/Length 40` is 40 bits.
    #[test]
    fn key_length_follows_the_revision() {
        let mut d = r3_dict();
        d.set("R", Object::Integer(2));
        assert_eq!(StandardSecurity::parse(&d, &[]).expect("parse").key_len, 5);

        let mut d = r3_dict();
        d.set("Length", Object::Integer(256));
        assert!(
            StandardSecurity::parse(&d, &[]).is_err(),
            "32-byte RC4 key at R3"
        );

        let mut d = r6_dict();
        d.set("V", Object::Integer(4));
        d.set("R", Object::Integer(4));
        d.set("O", string(O_LEGACY));
        d.set("U", string(O_LEGACY));
        assert!(StandardSecurity::parse(&d, &[]).is_err(), "AESV3 at R4");

        let mut d = r3_dict();
        d.set("V", Object::Integer(4));
        d.set("R", Object::Integer(4));
        d.set("StmF", name("StdCF"));
        d.set("StrF", name("StdCF"));
        d.set(
            "CF",
            Object::Dictionary(dict(&[(
                "StdCF",
                Object::Dictionary(dict(&[
                    ("CFM", name("V2")),
                    ("Length", Object::Integer(40)),
                ])),
            )])),
        );
        assert_eq!(StandardSecurity::parse(&d, &[]).expect("parse").key_len, 5);
    }

    #[test]
    fn rejects_short_o_u_oe_ue() {
        let mut d = r3_dict();
        d.set("U", string("00"));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let mut d = r3_dict();
        let _ = d.remove(b"O");
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let mut d = r6_dict();
        d.set("UE", string("00"));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let mut d = r6_dict();
        d.set("O", string(O_LEGACY));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
    }

    #[test]
    fn rejects_unknown_crypt_filter_method_and_undefined_filter() {
        let mut d = r6_dict();
        d.set(
            "CF",
            Object::Dictionary(dict(&[(
                "StdCF",
                Object::Dictionary(dict(&[("CFM", name("AESV9"))])),
            )])),
        );
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let mut d = r6_dict();
        d.set("StmF", name("Nope"));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        // No /CF at all with /StmF naming a filter: refused, not a panic.
        let mut d = r6_dict();
        let _ = d.remove(b"CF");
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        // /CF with /StmF /Identity: an unencrypted-data document.
        let mut d = r6_dict();
        let _ = d.remove(b"CF");
        d.set("StmF", name("Identity"));
        d.set("StrF", name("Identity"));
        let s = StandardSecurity::parse(&d, &[]).expect("identity filters");
        assert_eq!(s.stream_cipher, Cipher::Identity);
    }

    #[test]
    fn wrong_types_are_errors_not_panics() {
        let mut d = r3_dict();
        d.set("O", Object::Integer(5));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
        let mut d = r3_dict();
        d.set("P", name("x"));
        assert!(StandardSecurity::parse(&d, &[]).is_ok());
        let mut d = r3_dict();
        d.set("P", Object::Integer(i64::MIN));
        assert!(StandardSecurity::parse(&d, &[]).is_ok());
        d.set("P", Object::Integer(i64::MAX));
        assert!(StandardSecurity::parse(&d, &[]).is_ok());
        let mut d = r6_dict();
        d.set("CF", Object::Integer(1));
        assert!(StandardSecurity::parse(&d, &[]).is_err());
    }

    #[test]
    fn pad_password_truncates_and_pads() {
        assert_eq!(pad_password(&[]), PAD);
        let long = [7u8; 40];
        assert_eq!(pad_password(&long), [7u8; 32]);
        let p = pad_password(b"ab");
        assert_eq!(&p[..2], b"ab");
        assert_eq!(&p[2..], &PAD[..30]);
    }
}
