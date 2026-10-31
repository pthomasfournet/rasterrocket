//! In-process decryption of documents protected by the Standard security
//! handler (ISO 32000-2 §7.6).
//!
//! A [`SecurityHandler`] is built once per document from the trailer's
//! `/Encrypt` dictionary and a password, and then decrypts every string and
//! stream of an indirect object as that object is parsed, keyed by the
//! object's number and generation (Algorithm 1).  Objects inside an object
//! stream are not decrypted individually (the containing stream was), and a
//! cross-reference stream is never encrypted.
//!
//! Whether decryption is *permitted* is the caller's decision, carried by
//! [`OpenOptions::decrypt_authorized`]; this module only performs it.  A
//! password is an input to the key derivation, not an authorisation.

mod primitives;
mod standard;

use std::collections::HashMap;

use self::{
    primitives::{aes_cbc_decrypt, md5, rc4_in_place},
    standard::StandardSecurity,
};
use crate::{
    dictionary::Dictionary,
    error::PdfError,
    object::{Object, ObjectId},
};

/// A document password.
///
/// `Debug` prints a placeholder rather than the password, so a
/// configuration holding one can be logged.
#[derive(Clone, PartialEq, Eq)]
pub struct Password(String);

impl Password {
    /// The password text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for Password {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Password {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl std::fmt::Debug for Password {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Password(<redacted>)")
    }
}

/// How to treat an encrypted document when opening it.
///
/// The default refuses encrypted input: an `/Encrypt` entry in the trailer
/// makes the open fail with [`PdfError::EncryptedDocument`].  Decryption of
/// an authorised document first tries the empty user password (a
/// permissions-only document opens without any password), then
/// [`password`](Self::password) as the user password and as the owner
/// password.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct OpenOptions {
    /// Whether the caller permits decrypting an encrypted document at all.
    /// Default `false`; no effect on an unencrypted document.
    pub decrypt_authorized: bool,
    /// Password tried after the empty user password, as the user password
    /// and then as the owner password.  Has no effect unless
    /// [`decrypt_authorized`](Self::decrypt_authorized) is `true`.
    pub password: Option<Password>,
}

impl OpenOptions {
    /// Options with the given decryption authorisation and password.
    #[must_use]
    pub const fn new(decrypt_authorized: bool, password: Option<Password>) -> Self {
        Self {
            decrypt_authorized,
            password,
        }
    }
}

/// Error message handed back when decryption of an encrypted document was
/// not authorised by the caller.
#[must_use]
pub fn msg_gate_declined() -> String {
    "document is encrypted (PDF Standard Security Handler); decryption was \
     not authorised. Re-run interactively and confirm the private-copy / \
     liability prompt, or set --decrypt-owned / RROCKET_DECRYPT_OWNED=1 for \
     unattended use."
        .to_owned()
}

/// The cipher a crypt filter applies to data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cipher {
    /// No encryption (`/Identity`, or `/CFM /None`).
    Identity,
    /// RC4 with the per-object key of Algorithm 1 (`/V` 1–2, `/CFM /V2`).
    Rc4,
    /// AES-128-CBC with the per-object key of Algorithm 1 (`/CFM /AESV2`).
    AesV2,
    /// AES-256-CBC with the file key itself (`/CFM /AESV3`).
    AesV3,
}

/// A document's authenticated Standard security handler.
pub(crate) struct SecurityHandler {
    /// The file encryption key.
    key: Vec<u8>,
    /// Cipher for stream data (`/StmF`).
    stream_cipher: Cipher,
    /// Cipher for strings (`/StrF`).
    string_cipher: Cipher,
    /// Named crypt filters a stream may select with its own `/Crypt` filter.
    filters: HashMap<Vec<u8>, Cipher>,
    /// `false` leaves `/Type /Metadata` streams in the clear.
    encrypt_metadata: bool,
}

impl std::fmt::Debug for SecurityHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityHandler")
            .field("key_len", &self.key.len())
            .field("stream_cipher", &self.stream_cipher)
            .field("string_cipher", &self.string_cipher)
            .field("filters", &self.filters)
            .field("encrypt_metadata", &self.encrypt_metadata)
            .finish()
    }
}

impl SecurityHandler {
    /// Validate `dict` (the `/Encrypt` dictionary) and unlock the file key.
    ///
    /// The empty user password is tried first, then `password` as the user
    /// and as the owner password.  `id0` is the first string of the trailer
    /// `/ID` array.
    ///
    /// # Errors
    /// [`PdfError::EncryptedDocument`] naming the unsupported or malformed
    /// entry, or stating that a password is required or that the supplied
    /// one is wrong.
    pub(crate) fn open(
        dict: &Dictionary,
        id0: &[u8],
        password: Option<&str>,
    ) -> Result<Self, PdfError> {
        let standard = StandardSecurity::parse(dict, id0).map_err(|detail| {
            PdfError::EncryptedDocument(format!(
                "document is encrypted and cannot be decrypted: {detail}"
            ))
        })?;
        let key = standard
            .authenticate("")
            .or_else(|| password.and_then(|p| standard.authenticate(p)))
            .ok_or_else(|| {
                PdfError::EncryptedDocument(if password.is_some() {
                    "document is encrypted and the supplied password is neither its user nor \
                     its owner password"
                        .to_owned()
                } else {
                    "document is encrypted and requires a password (none was given)".to_owned()
                })
            })?;
        Ok(Self {
            key,
            stream_cipher: standard.stream_cipher,
            string_cipher: standard.string_cipher,
            filters: standard.filters,
            encrypt_metadata: standard.encrypt_metadata,
        })
    }

    /// Decrypt every string and stream in the indirect object `id`, in
    /// place.  A cross-reference stream is left untouched.
    ///
    /// Data that cannot be decrypted (a ciphertext shorter than its IV) is
    /// replaced by empty data with a warning; nothing here fails the object.
    pub(crate) fn decrypt_object(&self, id: ObjectId, obj: &mut Object) {
        match obj {
            Object::String(bytes, _) => self.decrypt_bytes(self.string_cipher, id, bytes, "string"),
            Object::Array(items) => {
                for item in items {
                    self.decrypt_object(id, item);
                }
            }
            Object::Dictionary(dict) => self.decrypt_dict(id, dict),
            Object::Stream(stream) => {
                if stream.dict.get(b"Type").and_then(Object::as_name) == Some(b"XRef") {
                    return;
                }
                let cipher = self.stream_cipher_for(&mut stream.dict);
                self.decrypt_dict(id, &mut stream.dict);
                self.decrypt_bytes(cipher, id, &mut stream.content, "stream");
            }
            Object::Null
            | Object::Boolean(_)
            | Object::Integer(_)
            | Object::Real(_)
            | Object::Name(_)
            | Object::Reference(_) => {}
        }
    }

    fn decrypt_dict(&self, id: ObjectId, dict: &mut Dictionary) {
        for value in dict.values_mut() {
            self.decrypt_object(id, value);
        }
    }

    /// The cipher for a stream's data: `/Identity` for metadata left in the
    /// clear, the crypt filter named by the stream's own `/Crypt` filter
    /// (which is removed from the dictionary, its work being done here), or
    /// `/StmF`.
    fn stream_cipher_for(&self, dict: &mut Dictionary) -> Cipher {
        if !self.encrypt_metadata
            && dict.get(b"Type").and_then(Object::as_name) == Some(b"Metadata")
        {
            return Cipher::Identity;
        }
        match take_crypt_filter(dict) {
            None => self.stream_cipher,
            Some(name) if name == b"Identity" => Cipher::Identity,
            Some(name) => self.filters.get(&name).copied().unwrap_or_else(|| {
                log::warn!(
                    "stream names crypt filter /{} which /CF does not define; treating as /Identity",
                    String::from_utf8_lossy(&name)
                );
                Cipher::Identity
            }),
        }
    }

    fn decrypt_bytes(&self, cipher: Cipher, id: ObjectId, data: &mut Vec<u8>, what: &str) {
        if data.is_empty() {
            return;
        }
        let result = match cipher {
            Cipher::Identity => return,
            Cipher::Rc4 => rc4_in_place(&self.object_key(id, false), data),
            Cipher::AesV2 => aes_cbc_decrypt(&self.object_key(id, true), data).map(|pt| *data = pt),
            Cipher::AesV3 => aes_cbc_decrypt(&self.key, data).map(|pt| *data = pt),
        };
        if let Err(e) = result {
            log::warn!(
                "object {} {}: {what} could not be decrypted ({e}); dropping it",
                id.0,
                id.1
            );
            data.clear();
        }
    }

    /// Algorithm 1: the per-object key, MD5 of the file key, the low three
    /// bytes of the object number, the low two of the generation and, for
    /// AES, the bytes `sAlT`, truncated to `min(n + 5, 16)` bytes.
    fn object_key(&self, id: ObjectId, aes: bool) -> Vec<u8> {
        let num = id.0.to_le_bytes();
        let generation = id.1.to_le_bytes();
        let salt: &[u8] = if aes { b"sAlT" } else { b"" };
        let digest = md5(&[&self.key, &num[..3], &generation[..2], salt]);
        let n = (self.key.len() + 5).min(16);
        digest[..n].to_vec()
    }
}

/// Remove a `/Crypt` entry from the stream's filter chain, together with
/// its `/DecodeParms` entry, returning the crypt filter it named
/// (`/Identity` when the parameters name none).
fn take_crypt_filter(dict: &mut Dictionary) -> Option<Vec<u8>> {
    let is_crypt = |o: &Object| o.as_name() == Some(b"Crypt");
    let (index, single) = match dict.get(b"Filter")? {
        Object::Array(filters) => (filters.iter().position(is_crypt)?, filters.len() == 1),
        filter if is_crypt(filter) => (0, true),
        _ => return None,
    };
    let parms = match dict.get(b"DecodeParms") {
        Some(Object::Array(parms)) => parms.get(index).and_then(Object::as_dict),
        Some(parm) if single => parm.as_dict(),
        _ => None,
    };
    let name = parms
        .and_then(|p| p.get(b"Name"))
        .and_then(Object::as_name)
        .map_or_else(|| b"Identity".to_vec(), <[u8]>::to_vec);

    if let Some(Object::Array(filters)) = dict.get_mut(b"Filter")
        && !single
    {
        let _ = filters.remove(index);
    } else {
        let _ = dict.remove(b"Filter");
    }
    match dict.get_mut(b"DecodeParms") {
        Some(Object::Array(parms)) if !single => {
            if index < parms.len() {
                let _ = parms.remove(index);
            }
        }
        Some(_) => {
            let _ = dict.remove(b"DecodeParms");
        }
        None => {}
    }
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Stream, StringFormat};

    fn name(s: &str) -> Object {
        Object::Name(s.as_bytes().to_vec())
    }

    fn handler(cipher: Cipher) -> SecurityHandler {
        SecurityHandler {
            key: vec![0x11; 16],
            stream_cipher: cipher,
            string_cipher: cipher,
            filters: HashMap::from([(b"StdCF".to_vec(), cipher)]),
            encrypt_metadata: true,
        }
    }

    #[test]
    fn object_key_follows_algorithm_1() {
        let h = handler(Cipher::Rc4);
        let k = h.object_key((7, 0), false);
        assert_eq!(k.len(), 16);
        assert_eq!(k, md5(&[&[0x11; 16], &[7, 0, 0], &[0, 0], b""])[..16]);
        let k5 = SecurityHandler {
            key: vec![1; 5],
            ..handler(Cipher::Rc4)
        }
        .object_key((1, 0), false);
        assert_eq!(k5.len(), 10);
        assert_ne!(h.object_key((7, 0), true), k);
    }

    #[test]
    fn rc4_string_round_trips() {
        let h = handler(Cipher::Rc4);
        let mut obj = Object::String(b"hello".to_vec(), StringFormat::Literal);
        h.decrypt_object((3, 1), &mut obj);
        let Object::String(ct, _) = &obj else {
            panic!()
        };
        assert_ne!(ct, b"hello");
        // RC4 is an involution under the same key.
        h.decrypt_object((3, 1), &mut obj);
        assert_eq!(
            obj,
            Object::String(b"hello".to_vec(), StringFormat::Literal)
        );
    }

    #[test]
    fn aes_short_ciphertext_is_dropped_not_panicked() {
        let h = handler(Cipher::AesV2);
        let mut obj = Object::Array(vec![
            Object::String(vec![1, 2, 3], StringFormat::Literal),
            Object::String(Vec::new(), StringFormat::Literal),
        ]);
        h.decrypt_object((3, 0), &mut obj);
        assert_eq!(
            obj,
            Object::Array(vec![
                Object::String(Vec::new(), StringFormat::Literal),
                Object::String(Vec::new(), StringFormat::Literal),
            ])
        );
        let mut s = Object::Stream(Stream::new(Dictionary::new(), vec![0u8; 15]));
        h.decrypt_object((4, 0), &mut s);
        assert_eq!(s.as_stream().map(|s| s.content.len()), Some(0));
        let h3 = SecurityHandler {
            key: vec![2; 32],
            ..handler(Cipher::AesV3)
        };
        let mut s = Object::Stream(Stream::new(Dictionary::new(), vec![0u8; 15]));
        h3.decrypt_object((4, 0), &mut s);
        assert_eq!(s.as_stream().map(|s| s.content.len()), Some(0));
    }

    #[test]
    fn xref_stream_is_never_touched() {
        let h = handler(Cipher::Rc4);
        let mut dict = Dictionary::new();
        dict.set("Type", name("XRef"));
        dict.set("ID", Object::String(b"abc".to_vec(), StringFormat::Literal));
        let mut obj = Object::Stream(Stream::new(dict, b"raw".to_vec()));
        let before = obj.clone();
        h.decrypt_object((9, 0), &mut obj);
        assert_eq!(obj, before);
    }

    #[test]
    fn metadata_stays_clear_when_not_encrypted() {
        let h = SecurityHandler {
            encrypt_metadata: false,
            ..handler(Cipher::Rc4)
        };
        let mut dict = Dictionary::new();
        dict.set("Type", name("Metadata"));
        let mut obj = Object::Stream(Stream::new(dict, b"<xmp/>".to_vec()));
        h.decrypt_object((9, 0), &mut obj);
        assert_eq!(
            obj.as_stream().map(|s| s.content.as_slice()),
            Some(&b"<xmp/>"[..])
        );
    }

    #[test]
    fn identity_crypt_filter_is_consumed() {
        let h = handler(Cipher::Rc4);
        let mut dict = Dictionary::new();
        dict.set(
            "Filter",
            Object::Array(vec![name("Crypt"), name("FlateDecode")]),
        );
        let mut parms = Dictionary::new();
        parms.set("Name", name("Identity"));
        dict.set(
            "DecodeParms",
            Object::Array(vec![Object::Dictionary(parms), Object::Null]),
        );
        let mut obj = Object::Stream(Stream::new(dict, b"plain".to_vec()));
        h.decrypt_object((5, 0), &mut obj);
        let s = obj.as_stream().expect("stream");
        assert_eq!(s.content, b"plain");
        assert_eq!(
            s.dict.get(b"Filter"),
            Some(&Object::Array(vec![name("FlateDecode")]))
        );
        assert_eq!(
            s.dict.get(b"DecodeParms"),
            Some(&Object::Array(vec![Object::Null]))
        );
    }

    #[test]
    fn named_crypt_filter_selects_its_cipher() {
        let h = SecurityHandler {
            stream_cipher: Cipher::Identity,
            ..handler(Cipher::Rc4)
        };
        let mut dict = Dictionary::new();
        dict.set("Filter", name("Crypt"));
        let mut parms = Dictionary::new();
        parms.set("Name", name("StdCF"));
        dict.set("DecodeParms", Object::Dictionary(parms));
        let mut obj = Object::Stream(Stream::new(dict, b"plain".to_vec()));
        h.decrypt_object((5, 0), &mut obj);
        let s = obj.as_stream().expect("stream");
        assert_ne!(s.content, b"plain");
        assert!(s.dict.get(b"Filter").is_none());
        assert!(s.dict.get(b"DecodeParms").is_none());
        // An undefined name falls back to /Identity with a warning.
        let mut dict = Dictionary::new();
        dict.set("Filter", name("Crypt"));
        let mut parms = Dictionary::new();
        parms.set("Name", name("Nope"));
        dict.set("DecodeParms", Object::Dictionary(parms));
        let mut obj = Object::Stream(Stream::new(dict, b"plain".to_vec()));
        h.decrypt_object((5, 0), &mut obj);
        assert_eq!(
            obj.as_stream().map(|s| s.content.as_slice()),
            Some(&b"plain"[..])
        );
    }

    #[test]
    fn gate_declined_message_is_actionable() {
        let m = msg_gate_declined();
        assert!(m.contains("--decrypt-owned"));
        assert!(m.contains("RROCKET_DECRYPT_OWNED=1"));
        assert!(!m.contains("no pages"));
    }
}
