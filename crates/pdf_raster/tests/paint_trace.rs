//! Paint-trace goldens: the trace of each synthetic page below is compared
//! with `tests/traces/<case>.trace`, and the corpus fixtures' traces, when
//! the fixtures are present, with `tests/fixtures/traces/`.
//!
//! A trace records every painting operation of a page in its device space
//! (see `rasterrocket::dev::paint_trace`), so a change that moves geometry,
//! a clip, a colour or a glyph changes a golden even when the pixels round
//! the same.  With `UPDATE_TRACES` set, each test writes its goldens from
//! the current build instead; a mismatch writes the trace it got next to
//! the golden as `<case>.trace.actual`.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use common::{assemble_pdf, corpus_fixture, isolated_fonts, stream_object};
use rasterrocket::{RasterSession, RenderParams, RenderRegion, open_session_from_bytes};

/// Set to write the goldens from the current build.
const UPDATE_VAR: &str = "UPDATE_TRACES";

/// A one-page PDF: catalog 1 (with `catalog` entries), page tree 2, a
/// 100 × 100 pt page 3 with resources `resources` and the entries
/// `page`, content 4 `content`, and `extra` from object 5 on.
fn pdf(catalog: &str, resources: &str, page: &str, content: &str, extra: &[Vec<u8>]) -> Vec<u8> {
    let mut objects = vec![
        format!("<< /Type /Catalog /Pages 2 0 R {catalog} >>").into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] \
             /Resources << {resources} >> /Contents 4 0 R {page} >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
    ];
    objects.extend_from_slice(extra);
    assemble_pdf(&objects)
}

/// [`pdf`] with no catalog or page entries.
fn page(resources: &str, content: &str, extra: &[Vec<u8>]) -> Vec<u8> {
    pdf("", resources, "", content, extra)
}

/// A session over `bytes` drawing non-embedded fonts from the library's
/// own faces only, so no installed font takes part.
fn session(bytes: Vec<u8>) -> RasterSession {
    open_session_from_bytes(bytes, &isolated_fonts(Vec::new())).expect("the PDF opens")
}

/// The trace of page `page` of `session` at 72 dpi.
fn trace(session: &RasterSession, page: u32, params: &RenderParams) -> String {
    rasterrocket::dev::paint_trace(session, page, params).expect("the page renders")
}

/// Compare `got` with the golden `path`, or write it there under
/// [`UPDATE_VAR`].
fn check_golden(path: &Path, got: &str) {
    let actual = path.with_extension("trace.actual");
    if std::env::var_os(UPDATE_VAR).is_some() {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        }
        std::fs::write(path, got)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        let _ = std::fs::remove_file(&actual);
        eprintln!("wrote {}", path.display());
        return;
    }
    let want = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "no golden {} ({e}); write it with {UPDATE_VAR}=1 cargo test -p rasterrocket \
             --test paint_trace",
            path.display()
        )
    });
    if want == got {
        let _ = std::fs::remove_file(&actual);
        return;
    }
    std::fs::write(&actual, got)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", actual.display()));
    let (want_lines, got_lines): (Vec<&str>, Vec<&str>) =
        (want.lines().collect(), got.lines().collect());
    let n = (0..want_lines.len().max(got_lines.len()))
        .find(|&i| want_lines.get(i) != got_lines.get(i))
        .unwrap_or(0);
    let line = |lines: &[&str]| lines.get(n).copied().unwrap_or("<end>").to_owned();
    let (w, g) = (line(&want_lines), line(&got_lines));
    panic!(
        "the trace differs from {} at line {}:\n  golden: {w}\n  actual: {g}\n\
         the whole trace is in {}; if the change is intended, rewrite the golden with \
         {UPDATE_VAR}=1",
        path.display(),
        n + 1,
        actual.display()
    );
}

/// Check the trace of page 1 of `pdf` at 72 dpi against the golden of
/// `case`.
fn golden(case: &str, pdf: Vec<u8>) {
    let got = trace(&session(pdf), 1, &RenderParams::new(1.0));
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/traces")
        .join(format!("{case}.trace"));
    check_golden(&path, &got);
}

#[test]
fn paths() {
    golden(
        "paths",
        page(
            "",
            "0 0 1 rg 10 10 30 20 re f \
             0 1 0 rg 50 10 m 90 10 l 70 40 l h 60 15 m 80 15 l 70 30 l h f* \
             1 0 0 RG 3 w 1 J 1 j [6 2] 1 d 10 50 m 30 90 50 50 90 90 c S \
             0.5 g 0 G 0.5 w 2 J 2 j 4 M 20 60 30 30 re B \
             0 g 10 95 m 90 95 l f \
             0 w 5 5 m 95 5 l S",
            &[],
        ),
    );
}

#[test]
fn clips() {
    golden(
        "clips",
        page(
            "/Font << /F1 5 0 R >>",
            "q 10 10 80 80 re W n q 20 20 m 80 20 l 50 80 l h W* n 0 g 0 0 100 100 re f Q \
             0 0 1 rg 0 0 50 50 re f Q 1 0 0 rg 60 60 30 30 re f \
             BT /F1 30 Tf 7 Tr 10 40 Td (Clip) Tj ET 0 1 0 rg 0 0 100 100 re f \
             q 2 w 20 20 30 30 re W S 0 0 100 100 re f Q",
            &[b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec()],
        ),
    );
}

#[test]
fn text() {
    golden(
        "text",
        page(
            "/Font << /F1 5 0 R /F2 6 0 R >>",
            "BT /F1 12 Tf 14 TL 5 85 Td (Fill) Tj T* 1 Tr 1 0 0 RG (Stroke) Tj \
             T* 2 Tr 3 Tc 2 Tw (Fill stroke) Tj T* 3 Tr (Invisible) Tj \
             0 Tr 0 Tc 0 Tw 80 Tz 4 Ts T* [(Ke) -250 (rn)] TJ \
             100 Tz 0 Ts /F2 10 Tf T* (Times) ' 2 1 (Spaced) \" ET",
            &[
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman \
                  /Encoding /WinAnsiEncoding >>"
                    .to_vec(),
            ],
        ),
    );
}

#[test]
fn type3_glyphs() {
    let proc_a = "10 0 0 0 8 8 d1 0 0 8 8 re f";
    let proc_b = "10 0 d0 1 0 0 rg 0 0 m 8 0 l 4 8 l f";
    golden(
        "type3_glyphs",
        page(
            "/Font << /T3 5 0 R >>",
            "0 0 1 rg BT /T3 24 Tf 10 50 Td (abca) Tj ET",
            &[
                b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 10 10] \
                  /FontMatrix [0.1 0 0 0.1 0 0] /CharProcs << /a 6 0 R /b 7 0 R >> \
                  /Encoding << /Differences [97 /a /b] >> /FirstChar 97 /LastChar 99 \
                  /Widths [10 10 10] >>"
                    .to_vec(),
                stream_object("", proc_a.as_bytes()),
                stream_object("", proc_b.as_bytes()),
            ],
        ),
    );
}

#[test]
fn images() {
    golden(
        "images",
        page(
            "/XObject << /Im 5 0 R /Mask 6 0 R >>",
            "q 40 0 0 30 5 5 cm /Im Do Q \
             q 1 0 0 rg 30 0 0 30 55 5 cm /Mask Do Q \
             q 20 0 10 20 10 60 cm BI /W 2 /H 2 /CS /G /BPC 8 /F /AHx ID 008080ff> EI Q",
            &[
                stream_object(
                    "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceRGB \
                     /BitsPerComponent 8 /Interpolate true",
                    b"\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\xff",
                ),
                stream_object(
                    "/Type /XObject /Subtype /Image /Width 8 /Height 2 /ImageMask true \
                     /BitsPerComponent 1",
                    b"\x5a\xa5",
                ),
            ],
        ),
    );
}

/// An axial shading from red to blue across `[0 0 100 0]`.
const AXIAL: &str = "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 100 0] \
     /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> \
     /Extend [true true] >>";

#[test]
fn shadings() {
    let radial = "<< /ShadingType 3 /ColorSpace /DeviceGray /Coords [50 50 0 50 50 40] \
         /Function << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >>";
    golden(
        "shadings",
        page(
            "/Shading << /Ax 5 0 R /Ra 6 0 R >> /Pattern << /P 7 0 R >>",
            "q 0 0 100 30 re W n /Ax sh Q q 0 40 100 30 re W n /Ra sh Q \
             /Pattern cs /P scn 10 75 80 20 re f",
            &[
                AXIAL.as_bytes().to_vec(),
                radial.as_bytes().to_vec(),
                b"<< /PatternType 2 /Shading 5 0 R /Matrix [1 0 0 1 0 0] >>".to_vec(),
            ],
        ),
    );
}

#[test]
fn tiling_patterns() {
    let cell = "0 0 1 rg 0 0 5 5 re f";
    let uncoloured = "0 0 5 5 re f";
    golden(
        "tiling_patterns",
        page(
            "/Pattern << /C 5 0 R /U 6 0 R >> \
             /ColorSpace << /UP [/Pattern /DeviceRGB] >>",
            "/Pattern cs /C scn 10 10 35 35 re f \
             /UP cs 0 1 0 /U scn 55 10 35 35 re f \
             /Pattern CS /C SCN 4 w 10 60 m 90 90 l S",
            &[
                stream_object(
                    "/PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 10 10] \
                     /XStep 10 /YStep 10 /Resources << >>",
                    cell.as_bytes(),
                ),
                stream_object(
                    "/PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 10 10] \
                     /XStep 10 /YStep 10 /Matrix [1 0 0 1 2 3] /Resources << >>",
                    uncoloured.as_bytes(),
                ),
            ],
        ),
    );
}

#[test]
fn groups() {
    let isolated = "0 0 1 rg 0 0 40 40 re f 1 0 0 rg 20 20 40 40 re f";
    golden(
        "groups",
        page(
            "/XObject << /Iso 5 0 R /Ko 6 0 R /Plain 7 0 R >> \
             /ExtGState << /Mul << /BM /Multiply /ca 0.5 >> >>",
            "q /Mul gs 1 0 0 1 5 5 cm /Iso Do Q q 1 0 0 1 40 40 cm /Ko Do Q \
             q 0.5 0 0 0.5 0 60 cm /Plain Do Q",
            &[
                stream_object(
                    "/Type /XObject /Subtype /Form /BBox [0 0 60 60] \
                     /Group << /S /Transparency /I true >>",
                    isolated.as_bytes(),
                ),
                stream_object(
                    "/Type /XObject /Subtype /Form /BBox [0 0 60 60] \
                     /Group << /S /Transparency /K true >> \
                     /Resources << /ExtGState << /H << /ca 0.5 >> >> >>",
                    b"/H gs 0 g 0 0 40 40 re f 10 10 40 40 re f",
                ),
                stream_object(
                    "/Type /XObject /Subtype /Form /BBox [0 0 50 50] /Matrix [1 0 0 1 2 2]",
                    b"0 1 0 rg 0 0 50 50 re f",
                ),
            ],
        ),
    );
}

#[test]
fn soft_masks() {
    let mask = "0.8 g 0 0 50 100 re f";
    golden(
        "soft_masks",
        page(
            "/ExtGState << /L << /SMask << /S /Luminosity /G 5 0 R /BC [0] >> >> \
             /A << /SMask << /S /Alpha /G 5 0 R >> >> /N << /SMask /None >> >>",
            "q /L gs 0 0 1 rg 0 0 100 50 re f /N gs 0 50 100 10 re f Q \
             q /A gs 1 0 0 rg 0 60 100 40 re f Q",
            &[stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                mask.as_bytes(),
            )],
        ),
    );
}

#[test]
fn colour_spaces() {
    let tint = "<< /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>";
    let tints = "<< /FunctionType 2 /Domain [0 1 0 1] /C0 [0 0 0 0] /C1 [1 0 0 0] /N 1 >>";
    golden(
        "colour_spaces",
        page(
            &format!(
                "/ColorSpace << /Spot [/Separation /Spot /DeviceCMYK {tint}] \
                 /Two [/DeviceN [/Cyan /Spot2] /DeviceCMYK {tints}] \
                 /Ix [/Indexed /DeviceRGB 1 <ff000000ff00>] \
                 /Cal [/CalRGB << /WhitePoint [0.9505 1 1.089] >>] \
                 /Lab [/Lab << /WhitePoint [0.9505 1 1.089] /Range [-100 100 -100 100] >>] >>"
            ),
            "0.1 0.2 0.3 0.4 k 0 0 20 20 re f /Spot cs 0.7 sc 20 0 20 20 re f \
             /Two cs 0.5 0.25 scn 40 0 20 20 re f /Ix cs 1 sc 60 0 20 20 re f \
             /Cal cs 0.3 0.6 0.9 sc 80 0 20 20 re f /Lab cs 50 10 -20 sc 0 20 20 20 re f \
             /Spot CS 2 w 0 50 m 100 50 l S",
            &[],
        ),
    );
}

#[test]
fn annotations() {
    golden(
        "annotations",
        pdf(
            "",
            "",
            "/Annots [5 0 R 6 0 R]",
            "0.9 g 0 0 100 100 re f",
            &[
                b"<< /Type /Annot /Subtype /Square /Rect [10 10 40 40] /AP << /N 7 0 R >> >>"
                    .to_vec(),
                b"<< /Type /Annot /Subtype /Square /Rect [50 50 90 90] /C [1 0 0] /IC [0 0 1] \
                  /BS << /W 4 >> >>"
                    .to_vec(),
                stream_object(
                    "/Type /XObject /Subtype /Form /BBox [0 0 30 30]",
                    b"0 1 0 rg 0 0 30 30 re f",
                ),
            ],
        ),
    );
}

#[test]
fn optional_content() {
    golden(
        "optional_content",
        pdf(
            "/OCProperties << /OCGs [5 0 R 6 0 R] /D << /OFF [6 0 R] >> >>",
            "/Properties << /On 5 0 R /Off 6 0 R >> /XObject << /Fm 7 0 R >>",
            "",
            "/OC /On BDC 0 g 0 0 50 50 re f EMC /OC /Off BDC 1 0 0 rg 50 50 50 50 re f EMC \
             /Fm Do",
            &[
                b"<< /Type /OCG /Name (On) >>".to_vec(),
                b"<< /Type /OCG /Name (Off) >>".to_vec(),
                stream_object(
                    "/Type /XObject /Subtype /Form /BBox [0 0 100 100] /OC 6 0 R",
                    b"0 0 1 rg 0 0 100 100 re f",
                ),
            ],
        ),
    );
}

/// A page drawing with every family of operation the trace records.
fn rich_page() -> Vec<u8> {
    let mask = "0.6 g 0 0 100 100 re f";
    page(
        "/Font << /F1 5 0 R >> /Shading << /Ax 6 0 R >> /XObject << /G 7 0 R >> \
         /ExtGState << /M << /SMask << /S /Luminosity /G 7 0 R >> >> >>",
        "q 5 5 90 90 re W n 0 0 1 rg 0 0 100 100 re f /Ax sh Q \
         1 0 0 RG 2 w [4 2] 0 d 10 10 m 40 90 60 10 90 90 c S \
         q /M gs 0 g 20 20 60 60 re f Q \
         BT /F1 14 Tf 10 50 Td 2 Tr (Window) Tj ET /G Do",
        &[
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            AXIAL.as_bytes().to_vec(),
            stream_object(
                "/Type /XObject /Subtype /Form /BBox [0 0 100 100] \
                 /Group << /S /Transparency /CS /DeviceGray >>",
                mask.as_bytes(),
            ),
        ],
    )
}

/// The trace does not depend on how the page is drawn: whole, in bands of
/// any height, or as a region anywhere on it.
#[test]
fn the_trace_is_the_same_for_every_band_and_region() {
    let session = session(rich_page());
    let whole = trace(&session, 1, &RenderParams::new(1.0));
    assert!(whole.lines().count() > 10, "{whole}");
    for rows in [1, 7, 33, 99] {
        let mut params = RenderParams::new(1.0);
        params.band_rows = NonZeroU32::new(rows);
        assert_eq!(trace(&session, 1, &params), whole, "bands of {rows} rows");
    }
    for (x, y, w, h) in [(0, 0, 100, 1), (37, 61, 20, 20), (99, 99, 1, 1)] {
        let params = RenderParams::new(1.0).with_region(RenderRegion::new(x, y, w, h));
        assert_eq!(trace(&session, 1, &params), whole, "region {x},{y} {w}x{h}");
    }
}

/// The corpus fixtures traced by [`corpus_fixtures`].
const CORPUS: [&str; 11] = [
    "corpus-01-native-text-small",
    "corpus-02-native-vector-text",
    "corpus-03-native-text-dense",
    "corpus-04-ebook-mixed",
    "corpus-05-academic-book",
    "corpus-06-modern-layout-dct",
    "corpus-07-journal-dct-heavy",
    "corpus-08-scan-dct-1927",
    "corpus-09-scan-dct-1836",
    "corpus-10-scan-jbig2-jpx",
    "corpus-11-cmyk-swatches",
];

/// The first three pages of each corpus fixture, against goldens kept with
/// the fixtures in `tests/fixtures/traces/`.  Neither is in the
/// repository: write the goldens with the build to compare against, then
/// run the build under test.
#[test]
#[ignore = "needs the corpus fixtures; run with --include-ignored"]
fn corpus_fixtures() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/traces");
    for stem in CORPUS {
        let Some(path) = corpus_fixture(&format!("{stem}.pdf")) else {
            continue;
        };
        let bytes =
            std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let session = session(bytes);
        for page in 1..=session.total_pages().min(3) {
            let got = trace(&session, page, &RenderParams::new(1.0));
            check_golden(&dir.join(format!("{stem}-p{page}.trace")), &got);
        }
    }
}
