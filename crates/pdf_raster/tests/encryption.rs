//! Encrypted documents open and render in process, byte-identical to their
//! plaintext, under every revision of the Standard security handler.
//!
//! Fixtures are produced at test time by `qpdf` from one plaintext page;
//! each test skips with a note when `qpdf` is not installed.  The plaintext
//! is the reference: qpdf's own `--decrypt` output of each fixture renders
//! to the same bytes, which cross-checks the decrypted object graph
//! against an independent implementation.

#[expect(
    dead_code,
    reason = "shared helpers; this test uses PDF assembly and renders"
)]
mod common;

use std::{path::Path, process::Command};

use common::{assemble_pdf, render, stream_object};
use pdf::{Document, OpenOptions, Password};
use rasterrocket::{
    BackendPolicy, RasterError, RasterOptions, SessionConfig, open_session,
    open_session_from_bytes, raster_pdf_from_bytes,
};

const USER: &str = "user";
const OWNER: &str = "owner";

/// Content stream of the plaintext page; encrypted fixtures decrypt to
/// exactly these bytes.
const CONTENT: &[u8] = b"0.2 0.4 0.8 rg 40 40 120 60 re f\n\
    BT /F1 28 Tf 30 150 Td (Encrypted text) Tj ET\n\
    0 g 1 w 20 20 m 180 190 l S\n";

/// One page: a filled rectangle, a line of text, a stroked line and a
/// free-text annotation (so the file holds strings in several objects).
fn plaintext_pdf() -> Vec<u8> {
    assemble_pdf(&[
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R >> >> /Annots [6 0 R] >>"
            .to_vec(),
        stream_object("", CONTENT),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        b"<< /Type /Annot /Subtype /FreeText /Rect [100 100 190 130] \
           /Contents (annot string) /DA (/Helv 10 Tf 0 g) /F 4 >>"
            .to_vec(),
    ])
}

/// An encryption variant: the qpdf options that produce it.
struct Variant {
    name: &'static str,
    /// Options between `--encrypt` and `--`.
    encrypt_args: &'static [&'static str],
    /// Options after `--`.
    general_args: &'static [&'static str],
    user_password: &'static str,
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "r2-rc4-40",
        encrypt_args: &["--bits=40"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r3-rc4-128",
        encrypt_args: &["--bits=128"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r4-rc4-128-crypt-filter",
        encrypt_args: &["--bits=128", "--force-V4"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r4-aesv2",
        encrypt_args: &["--bits=128", "--use-aes=y"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r4-aesv2-cleartext-metadata",
        encrypt_args: &["--bits=128", "--use-aes=y", "--cleartext-metadata"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r4-aesv2-object-streams",
        encrypt_args: &["--bits=128", "--use-aes=y"],
        general_args: &["--object-streams=generate"],
        user_password: USER,
    },
    Variant {
        name: "r5-aesv3",
        encrypt_args: &["--bits=256", "--force-R5"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r6-aesv3",
        encrypt_args: &["--bits=256"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r6-aesv3-cleartext-metadata",
        encrypt_args: &["--bits=256", "--cleartext-metadata"],
        general_args: &[],
        user_password: USER,
    },
    Variant {
        name: "r6-aesv3-object-streams",
        encrypt_args: &["--bits=256"],
        general_args: &["--object-streams=generate"],
        user_password: USER,
    },
    // Permissions-only documents: empty user password, owner password set.
    Variant {
        name: "r3-rc4-128-owner-only",
        encrypt_args: &["--bits=128"],
        general_args: &[],
        user_password: "",
    },
    Variant {
        name: "r4-aesv2-owner-only",
        encrypt_args: &["--bits=128", "--use-aes=y"],
        general_args: &[],
        user_password: "",
    },
    Variant {
        name: "r6-aesv3-owner-only",
        encrypt_args: &["--bits=256"],
        general_args: &[],
        user_password: "",
    },
];

/// `true` when `qpdf` runs; otherwise prints why the test is skipped.
fn qpdf_available() -> bool {
    match Command::new("qpdf").arg("--version").output() {
        Ok(out) if out.status.success() => true,
        _ => {
            eprintln!("skipping: qpdf is not installed");
            false
        }
    }
}

/// The plaintext page encrypted as `variant`, and qpdf's own decryption of
/// that file, as bytes.
fn encrypt(dir: &Path, variant: &Variant) -> (Vec<u8>, Vec<u8>) {
    let plain = dir.join("plain.pdf");
    std::fs::write(&plain, plaintext_pdf()).expect("write plaintext");
    let enc = dir.join(format!("{}.pdf", variant.name));
    let dec = dir.join(format!("{}.dec.pdf", variant.name));
    let status = Command::new("qpdf")
        .arg("--allow-weak-crypto")
        .arg("--encrypt")
        .arg(format!("--user-password={}", variant.user_password))
        .arg(format!("--owner-password={OWNER}"))
        .args(variant.encrypt_args)
        .arg("--")
        .args(variant.general_args)
        .arg(&plain)
        .arg(&enc)
        .status()
        .expect("run qpdf --encrypt");
    assert!(status.success(), "{}: qpdf --encrypt failed", variant.name);
    let status = Command::new("qpdf")
        .arg(format!("--password={}", variant.user_password))
        .arg("--decrypt")
        .arg(&enc)
        .arg(&dec)
        .status()
        .expect("run qpdf --decrypt");
    assert!(status.success(), "{}: qpdf --decrypt failed", variant.name);
    (
        std::fs::read(&enc).expect("read encrypted"),
        std::fs::read(&dec).expect("read decrypted"),
    )
}

fn authorised(password: Option<&str>) -> SessionConfig {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.decrypt_authorized = true;
    config.password = password.map(Password::from);
    config
}

fn open_options(authorized: bool, password: Option<&str>) -> OpenOptions {
    OpenOptions {
        decrypt_authorized: authorized,
        password: password.map(Password::from),
    }
}

fn is_encrypted_error(err: &RasterError) -> bool {
    let text = err.to_string();
    text.contains("encrypted") && !text.contains("no pages")
}

#[test]
fn every_variant_decrypts_to_the_plaintext_and_renders_identically() {
    if !qpdf_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let plain = render(
        &open_session_from_bytes(plaintext_pdf(), &authorised(None)).expect("open plaintext"),
        1,
    );
    assert!(
        plain.2.iter().any(|&b| b != 255),
        "plaintext page must draw something"
    );

    for variant in VARIANTS {
        let (enc, dec) = encrypt(dir.path(), variant);
        let password = (!variant.user_password.is_empty()).then_some(variant.user_password);

        // Parser level: the content stream decrypts to the plaintext bytes.
        let doc = Document::from_bytes_with(enc.clone(), &open_options(true, password))
            .unwrap_or_else(|e| panic!("{}: open failed: {e}", variant.name));
        assert!(
            doc.is_encrypted(),
            "{}: handler must be installed",
            variant.name
        );
        let page = doc.get_page(0).expect("page 0");
        let content = doc.get_page_content(page).expect("page content");
        assert_eq!(
            content, CONTENT,
            "{}: content stream does not decrypt to the plaintext",
            variant.name
        );

        // Render level: byte-identical to the plaintext and to qpdf's
        // decryption of the same file.
        let ours = render(
            &open_session_from_bytes(enc.clone(), &authorised(password))
                .unwrap_or_else(|e| panic!("{}: open_session failed: {e}", variant.name)),
            1,
        );
        common::assert_identical(&plain, &ours, &format!("{} vs plaintext", variant.name));
        let qpdf = render(
            &open_session_from_bytes(dec, &authorised(None)).expect("open qpdf plaintext"),
            1,
        );
        common::assert_identical(&qpdf, &ours, &format!("{} vs qpdf --decrypt", variant.name));

        // The owner password unlocks the same key.
        let owner = render(
            &open_session_from_bytes(enc.clone(), &authorised(Some(OWNER)))
                .unwrap_or_else(|e| panic!("{}: owner password failed: {e}", variant.name)),
            1,
        );
        common::assert_identical(
            &plain,
            &owner,
            &format!("{} with owner password", variant.name),
        );

        // The path entry point decrypts the same way.
        let path = dir.path().join(format!("{}.pdf", variant.name));
        let from_path = render(
            &open_session(&path, &authorised(password))
                .unwrap_or_else(|e| panic!("{}: open_session(path) failed: {e}", variant.name)),
            1,
        );
        common::assert_identical(&plain, &from_path, &format!("{} from path", variant.name));
        assert!(rasterrocket::is_encrypted(&path), "{}: probe", variant.name);

        // A wrong password is an error, and so is no password, unless the
        // user password is empty: that one is tried first and opens the
        // document whatever else was supplied.
        let wrong = open_session_from_bytes(enc.clone(), &authorised(Some("wrong")));
        if password.is_some() {
            let err = wrong
                .err()
                .unwrap_or_else(|| panic!("{}: wrong password must fail", variant.name));
            assert!(is_encrypted_error(&err), "{}: {err}", variant.name);
            let err = open_session_from_bytes(enc.clone(), &authorised(None))
                .err()
                .unwrap_or_else(|| panic!("{}: no password must fail", variant.name));
            assert!(
                err.to_string().contains("password"),
                "{}: {err}",
                variant.name
            );
        } else {
            assert!(
                wrong.is_ok(),
                "{}: the empty user password must open the document",
                variant.name
            );
        }
    }
}

#[test]
fn the_authorisation_gate_holds_whatever_the_password() {
    if !qpdf_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for variant in [&VARIANTS[7], &VARIANTS[12]] {
        let (enc, _) = encrypt(dir.path(), variant);
        let path = dir.path().join(format!("{}.pdf", variant.name));
        for password in [None, Some(USER), Some(OWNER)] {
            let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
            config.password = password.map(Password::from);
            for err in [
                open_session_from_bytes(enc.clone(), &config).err(),
                open_session(&path, &config).err(),
            ] {
                let err =
                    err.unwrap_or_else(|| panic!("{}: unauthorised open must fail", variant.name));
                assert!(
                    err.to_string().contains("not authorised"),
                    "{}: {err}",
                    variant.name
                );
            }
            let mut opts = RasterOptions::default();
            opts.password = password.map(Password::from);
            let (_, page) = raster_pdf_from_bytes(enc.clone(), &opts)
                .next()
                .expect("one page result");
            let err = page.expect_err("unauthorised raster_pdf must fail");
            assert!(err.to_string().contains("not authorised"), "{err}");
            assert!(
                Document::from_bytes_with(enc.clone(), &open_options(false, password)).is_err()
            );
        }
        // Authorised, the simple API renders the document.
        let mut opts = RasterOptions::default();
        opts.decrypt_authorized = true;
        opts.password = (!variant.user_password.is_empty()).then(|| Password::from(USER));
        let (_, page) = raster_pdf_from_bytes(enc, &opts).next().expect("one page");
        assert!(page.is_ok(), "{}: authorised raster_pdf", variant.name);
    }
}

#[test]
fn unencrypted_input_ignores_the_decryption_options() {
    let pdf = plaintext_pdf();
    assert!(
        !Document::from_bytes_owned(pdf.clone())
            .expect("open")
            .is_encrypted()
    );
    let a = render(
        &open_session_from_bytes(
            pdf.clone(),
            &SessionConfig::with_policy(BackendPolicy::CpuOnly),
        )
        .expect("open"),
        1,
    );
    let b = render(
        &open_session_from_bytes(pdf, &authorised(Some("irrelevant"))).expect("open"),
        1,
    );
    common::assert_identical(&a, &b, "unencrypted with and without options");
}

#[test]
fn public_key_handler_is_refused_with_its_name() {
    // A trailer naming a security handler this parser does not implement.
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [] /Count 0 >>".to_vec(),
    ];
    objects.push(b"<< /Filter /Adobe.PubSec /V 4 /R 4 >>".to_vec());
    let pdf = assemble_pdf(&objects);
    let trailer = pdf
        .windows(8)
        .rposition(|w| w == b"/Root 1 ")
        .expect("trailer");
    let pdf = [&pdf[..trailer], b"/Encrypt 3 0 R ", &pdf[trailer..]].concat();
    let err = Document::from_bytes_with(pdf, &open_options(true, None))
        .err()
        .expect("must refuse");
    let text = err.to_string();
    assert!(text.contains("Adobe.PubSec"), "{text}");
    assert!(text.contains("Standard"), "{text}");
}
