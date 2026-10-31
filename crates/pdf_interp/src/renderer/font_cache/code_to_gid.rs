//! The code → glyph table of a loaded face.
//!
//! A face's own charmap answers for a code only when the code is what
//! the charmap is keyed by.  [`resolved_code_to_gid`] builds the table
//! that replaces it otherwise: by glyph name for a simple embedded Type 1
//! / CFF program, through the names' Unicode values for an embedded
//! TrueType program and for every substitute, by glyph index for a
//! cmap-less TrueType subset, and through the character collection or
//! the `ToUnicode` map for a composite font drawn by a substitute.

use font::face::FontFace;

use crate::renderer::encodings::{base_encoding_table, glyph_name_for_code};
use crate::renderer::fallback::first_scalar;
use crate::resources::cmap::CMap;
use crate::resources::cmap_predefined::{CollectionUnicode, cid_to_unicode};
use crate::resources::font::{BaseEncoding, CidEncoding, FontDescriptor, PdfFontKind};

/// The `code_to_gid` table a face of `desc` needs in place of the
/// descriptor's, or `None` when the descriptor's table (or the program's own
/// charmap) applies.
///
/// A simple embedded Type1/Type1C/CFF font is resolved by glyph *name*: its
/// char codes are not glyph indices (subset programs pack glyphs
/// arbitrarily) and it often has no usable Unicode cmap, so each code's name
/// (`Differences` override, else the base encoding) is looked up in the
/// program's charset.  An embedded font drawn through a Unicode cmap, and
/// every substitute (whose charmap is Unicode, from a `cmap` subtable or
/// from its glyph names), is resolved through the names' Unicode values.
/// CID/Type0 keep the descriptor's table; symbolic TrueType keeps the
/// program's own charmap.
///
/// An embedded simple sfnt with no code-keyed `cmap` subtable (a legacy
/// subset) resolves through [`resolve_cmapless`].
///
/// `embedded` is whether `face` draws the descriptor's own program.
pub(super) fn resolved_code_to_gid(
    face: &FontFace,
    desc: &FontDescriptor,
    embedded: bool,
) -> Option<Vec<u32>> {
    let simple = desc.cid_encoding.is_none();
    let outline = desc.type3.is_none();
    if embedded && simple && outline && face.program().lacks_cmap() {
        Some(resolve_cmapless(
            face,
            &desc.differences,
            desc.base_encoding,
        ))
    } else if wants_name_resolution(embedded, simple, outline, desc.kind) {
        Some(resolve_simple_to_gid(
            face,
            &desc.differences,
            desc.base_encoding,
        ))
    } else if wants_unicode_resolution(embedded, simple, outline, desc.kind)
        && (!embedded || face.has_unicode_cmap())
    {
        Some(resolve_via_unicode_cmap(
            face,
            &desc.differences,
            desc.base_encoding,
            embedded,
        ))
    } else if !embedded
        && !simple
        && let Some(enc) = &desc.cid_encoding
    {
        desc.substitute.script.and_then(cid_to_unicode).map_or_else(
            || {
                enc.to_unicode
                    .as_ref()
                    .map(|to_unicode| resolve_cids_via_to_unicode(face, enc, to_unicode))
            },
            |table| Some(resolve_cids_via_unicode(face, &table, enc.vertical)),
        )
    } else {
        None
    }
}

/// Build a 256-entry `code_to_gid` table for an embedded simple sfnt that
/// has no code-keyed `cmap` subtable.
///
/// A code whose name the font's encoding declares (a `Differences` entry,
/// else an explicit base encoding) selects the glyph of that name in the
/// program's `post` table (PDF §9.6.6.4).  Every other code is read as a
/// glyph index: without a `cmap`, the glyph order is the program's only
/// built-in encoding, and producers that drop the table address glyphs by
/// index.  A Unicode charmap derived from the `post` names is not
/// consulted for raw codes, since a code is not a Unicode value here.
fn resolve_cmapless(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
) -> Vec<u32> {
    (0..=u8::MAX)
        .map(|code| {
            let idx = usize::from(code);
            differences[idx]
                .as_deref()
                .or_else(|| {
                    (base != BaseEncoding::FontProgram)
                        .then(|| base_encoding_table(base)[idx])
                        .flatten()
                })
                .map(|name| face.raw_get_name_index(name))
                .filter(|&gid| gid != 0)
                .unwrap_or_else(|| u32::from(code))
        })
        .collect()
}

/// Build a 65 536-entry CID → glyph table for a composite font of no
/// known character collection drawn by a substitute, through the font's
/// `ToUnicode` map: each code it maps is read as its CID through the
/// `Encoding` `CMap` and its glyph found by Unicode value in the
/// substitute's charmap (vertical form in vertical writing mode).  A CID
/// the map leaves out, or whose glyph the substitute lacks, keeps the
/// identity reading, under which the substitutes of the metric-compatible
/// families draw by glyph order.
fn resolve_cids_via_to_unicode(face: &FontFace, enc: &CidEncoding, to_unicode: &CMap) -> Vec<u32> {
    let mut table: Vec<u32> = (0..1u32 << 16).collect();
    for (&code, &value) in &to_unicode.map {
        let Some(unicode) = first_scalar(value) else {
            continue;
        };
        let gid = face.raw_get_char_index(unicode);
        if gid == 0 {
            continue;
        }
        if let Some(slot) = table.get_mut(enc.encoding_cmap.lookup(code) as usize) {
            *slot = if enc.vertical {
                face.program().vertical_gid(gid)
            } else {
                gid
            };
        }
    }
    table
}

/// Build a 65 536-entry CID → glyph table for a composite font drawn by a
/// substitute, through the CID → Unicode table of its character
/// collection and the substitute's Unicode charmap: the glyph of the
/// first of a CID's code points the substitute maps.  In `vertical`
/// writing mode, and for a CID that is the collection's vertical form of
/// its code point in either mode, each glyph is replaced by its vertical
/// form.  The table spans the whole CID space so that no CID falls
/// through to the charmap.
fn resolve_cids_via_unicode(
    face: &FontFace,
    collection: &CollectionUnicode,
    vertical: bool,
) -> Vec<u32> {
    let mut table = vec![0u32; 1 << 16];
    for (cid, codes) in collection.iter() {
        if let Some(slot) = table.get_mut(cid as usize) {
            let gid = codes
                .iter()
                .map(|&unicode| face.raw_get_char_index(unicode))
                .find(|&gid| gid != 0)
                .unwrap_or(0);
            *slot = if vertical || collection.is_vertical(cid) {
                face.program().vertical_gid(gid)
            } else {
                gid
            };
        }
    }
    table
}

/// Whether a face resolves codes through glyph names in the program's
/// charset ([`resolve_simple_to_gid`]): only a simple (`simple`), outline
/// (`outline`, not Type 3) font drawn by its own `embedded` program of
/// kind `Type1` (`MMType1` included) or `Other` (OpenType-CFF wrappers).
///
/// - Drawn by a substitute: the substitute's Unicode charmap applies
///   ([`wants_unicode_resolution`]).
/// - Composite: the CID path owns resolution.
/// - Type 3: glyphs are content streams, not a program with a charset.
/// - `TrueType`: the program's cmap applies, by Unicode value or by code.
const fn wants_name_resolution(
    embedded: bool,
    simple: bool,
    outline: bool,
    kind: PdfFontKind,
) -> bool {
    embedded && simple && outline && matches!(kind, PdfFontKind::Type1 | PdfFontKind::Other)
}

/// Decide whether a simple font's codes are resolved through glyph names
/// and the Unicode cmap of the loaded face ([`resolve_via_unicode_cmap`]).
///
/// - Embedded TrueType: PDF §9.6.5.4 resolves code → glyph name → Unicode
///   → (3,1) cmap for a non-symbolic font.
/// - Non-embedded font of any simple kind: the substitute face is a Unicode
///   TrueType font that knows nothing of the PDF encoding, so a code means
///   the glyph its encoding names, never the substitute's glyph at that
///   code point.
///
/// Composite and Type 3 fonts own their resolution and never qualify.
/// The parameters are those of [`wants_name_resolution`].
const fn wants_unicode_resolution(
    embedded: bool,
    simple: bool,
    outline: bool,
    kind: PdfFontKind,
) -> bool {
    simple && outline && (!embedded || matches!(kind, PdfFontKind::TrueType))
}

/// Build a 256-entry `code_to_gid` table for a simple embedded
/// Type1/Type1C/CFF font by resolving each char code's glyph *name* against
/// the font program's own charset.
///
/// For every code 0..256 the glyph name is `differences[code]` if the
/// `Encoding/Differences` array overrides it, otherwise the name the
/// `base_encoding` assigns to that code (PDF §9.6.6).  With no base encoding
/// named ([`BaseEncoding::FontProgram`]) a code without a `Differences`
/// entry resolves through the program's built-in encoding, the only
/// mapping a subset with a custom encoding (glyphs named for their codes,
/// no standard names) carries.  A name is resolved to a GID through the
/// program's own glyph names (charset / `CharStrings` / `post` table — the
/// only correct source for a subsetted program where the char code is
/// neither the GID nor a usable Unicode-cmap key).
///
/// Resolution order per code:
/// 1. no `Differences` entry and no base encoding named → the program's
///    built-in encoding (`raw_get_builtin_index`), when it assigns the code
/// 2. name (Differences ▸ base encoding, `StandardEncoding` when none is
///    named) → `raw_get_name_index`
/// 3. name → Adobe Glyph List → Unicode → `raw_get_char_index` (handles
///    programs that expose a usable Unicode cmap but no charset names)
/// 4. `.notdef` (0) — an absent glyph.  The code is never read as a
///    glyph index: in a subset program it selects an unrelated glyph.
fn resolve_simple_to_gid(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
) -> Vec<u32> {
    let mut table = vec![0u32; 256];
    for (code, slot) in (0u8..=u8::MAX).zip(table.iter_mut()) {
        if base == BaseEncoding::FontProgram
            && differences[usize::from(code)].is_none()
            && let Some(gid) = face.raw_get_builtin_index(code)
        {
            *slot = gid;
            continue;
        }
        let Some(name) = glyph_name_for_code(differences, base, usize::from(code)) else {
            continue;
        };

        let mut gid = face.raw_get_name_index(name);
        if gid == 0
            && let Some(unicode) = font::glyph_name_to_unicode(name)
        {
            gid = face.raw_get_char_index(unicode);
        }
        *slot = gid;
    }
    table
}

/// Build a 256-entry `code_to_gid` table for a simple font drawn with a face
/// whose active cmap is Unicode (PDF §9.6.5.4: code → glyph name → Unicode →
/// (3,1) cmap): an embedded TrueType program, or the substitute face of a
/// non-embedded font.
///
/// A code is resolved through its glyph name when the name comes from a
/// `Differences` entry or from an explicit `WinAnsi` / `MacRoman` base
/// encoding and the font maps the name's Unicode value; this is what places
/// `WinAnsi` 0x80–0x9F (`oe` at 0x9C, `quoteright` at 0x92, …) on their glyphs.
/// A substitute program carries the name itself in its charset, which is
/// consulted first.  Every other code is looked up as a raw Unicode code
/// point, the identity that holds for the ASCII and Latin-1 ranges of those
/// encodings.
///
/// A substitute that lacks the glyph of a name it does understand draws
/// nothing for the code: the glyph at the raw code point would be an
/// unrelated character (`uni4E2D` at code 65 is not `A`).
///
/// For an `embedded` program, a code that neither lookup maps resolves
/// through the program's byte-keyed (3,0) / (1,0) subtables
/// ([`FontFace::raw_get_code_index`], PDF §9.6.6.4).  A Unicode charmap
/// derived from `post` glyph names keys its glyphs by the Unicode value of
/// the name (`therefore` → U+2234), so the code a symbolic subset assigns
/// such a glyph is reachable only through those subtables.
fn resolve_via_unicode_cmap(
    face: &FontFace,
    differences: &[Option<Box<str>>; 256],
    base: BaseEncoding,
    embedded: bool,
) -> Vec<u32> {
    let nonzero = |gid: u32| (gid != 0).then_some(gid);
    (0..=u8::MAX)
        .map(|code| {
            let idx = usize::from(code);
            let name = differences[idx].as_deref().or_else(|| {
                matches!(
                    base,
                    BaseEncoding::WinAnsi
                        | BaseEncoding::MacRoman
                        | BaseEncoding::Symbol
                        | BaseEncoding::ZapfDingbats
                )
                .then(|| base_encoding_table(base)[idx])
                .flatten()
            });
            let unicode = name.and_then(font::glyph_name_to_unicode);
            name.and_then(|name| {
                (!embedded)
                    .then(|| face.raw_get_name_index(name))
                    .and_then(nonzero)
                    .or_else(|| {
                        unicode
                            .map(|u| face.raw_get_char_index(u))
                            .and_then(nonzero)
                    })
            })
            .or_else(|| {
                (embedded || unicode.is_none())
                    .then(|| face.raw_get_char_index(u32::from(code)))
                    .and_then(nonzero)
            })
            .or_else(|| embedded.then(|| face.raw_get_code_index(code)))
            .unwrap_or(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::font_cache::test_fonts::{
        ahem, ahem_without_cmap, cache, composite_substitute, diffs, empty_diffs, face, simple,
    };
    use crate::resources::cmap::parse_cmap;
    use crate::resources::font_substitute::{Script, SubstituteStyle};

    /// The name-resolution path must fire ONLY for a simple embedded
    /// Type1/Type1C/CFF font.  A misfire would steal resolution from CID,
    /// symbolic TrueType, Type 3, or the non-embedded standard-14 path.
    #[test]
    fn name_resolution_dispatch_is_isolated() {
        // Simple embedded Type1 / OpenType-CFF (`Other`): the only YES cases.
        assert!(wants_name_resolution(true, true, true, PdfFontKind::Type1));
        assert!(wants_name_resolution(true, true, true, PdfFontKind::Other));

        // Drawn by a substitute: its Unicode charmap applies.
        assert!(!wants_name_resolution(
            false,
            true,
            true,
            PdfFontKind::Type1
        ));
        // CID/Type0 (cid_encoding present → simple == false).
        assert!(!wants_name_resolution(
            true,
            false,
            true,
            PdfFontKind::Other
        ));
        // Type 3 (type3 present → outline == false).
        assert!(!wants_name_resolution(
            true,
            true,
            false,
            PdfFontKind::Type1
        ));
        // Embedded TrueType: keeps the program's own charmap.
        assert!(!wants_name_resolution(
            true,
            true,
            true,
            PdfFontKind::TrueType
        ));
    }

    /// The simple-font name path must yield `.notdef` (0) for an unresolvable
    /// code — never code-identity.  Code-identity for a subsetted program
    /// produces wrong characters on a pixel-correct page; this pins that
    /// regression.
    #[test]
    fn unresolvable_code_is_notdef_never_identity() {
        let diffs = empty_diffs();
        // StandardEncoding leaves 0x80 unmapped: no name → no GID lookup.
        assert_eq!(
            glyph_name_for_code(&diffs, BaseEncoding::Standard, 0x80),
            None
        );
        // The control range is unmapped in every base encoding.
        for base in [
            BaseEncoding::Standard,
            BaseEncoding::WinAnsi,
            BaseEncoding::MacRoman,
        ] {
            assert_eq!(glyph_name_for_code(&diffs, base, 0), None);
            assert_eq!(glyph_name_for_code(&diffs, base, 31), None);
        }
    }

    #[test]
    fn unicode_resolution_dispatch() {
        // Non-embedded simple fonts of every kind resolve through the
        // substitute face's Unicode cmap.
        for kind in [
            PdfFontKind::Type1,
            PdfFontKind::TrueType,
            PdfFontKind::Other,
        ] {
            assert!(
                wants_unicode_resolution(false, true, true, kind),
                "{kind:?}"
            );
        }
        // Embedded: only TrueType; Type1/CFF programs resolve by name.
        assert!(wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::TrueType
        ));
        assert!(!wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::Type1
        ));
        assert!(!wants_unicode_resolution(
            true,
            true,
            true,
            PdfFontKind::Other
        ));
        // CID/Type0 and Type 3 never qualify.
        assert!(!wants_unicode_resolution(
            false,
            false,
            true,
            PdfFontKind::Other
        ));
        assert!(!wants_unicode_resolution(
            false,
            true,
            false,
            PdfFontKind::Other
        ));
    }

    /// A non-embedded font whose `Differences` array places glyph names on
    /// low codes (`0 /Q /eacute`) must draw those glyphs, not the substitute
    /// face's glyphs at code points 0 and 1.
    #[test]
    fn non_embedded_font_resolves_differences_on_substitute_face() {
        let desc = simple(
            PdfFontKind::Type1,
            None,
            diffs(&[(0, "Q"), (1, "eacute")]),
            BaseEncoding::MacRoman,
        );
        let mut cache = cache();
        let face = face(&mut cache, desc);
        let q = face.raw_get_char_index(u32::from('Q'));
        let eacute = face.raw_get_char_index(0xE9);
        assert_ne!(q, 0);
        assert_ne!(eacute, 0);
        assert_eq!(face.code_to_gid.first(), Some(&q));
        assert_eq!(face.code_to_gid.get(1), Some(&eacute));
        // A code left to the base encoding resolves through its name too:
        // MacRoman 0x8E is `eacute`, which is not U+008E.
        assert_eq!(face.code_to_gid.get(0x8E), Some(&eacute));
    }

    /// A `Differences` name the substitute understands but has no glyph for
    /// draws nothing: `uni4E2D` at code 65 must not select the Latin `A`
    /// the substitute keeps at that code point.
    #[test]
    fn missing_glyph_of_a_known_name_is_notdef_on_a_substitute() {
        let desc = simple(
            PdfFontKind::TrueType,
            None,
            diffs(&[(65, "uni4E2D"), (66, "afii10017")]),
            BaseEncoding::WinAnsi,
        );
        let mut cache = cache();
        let face = face(&mut cache, desc);
        assert_eq!(
            face.raw_get_char_index(0x4E2D),
            0,
            "the Latin substitute has no CJK"
        );
        assert_eq!(face.code_to_gid.get(65), Some(&0));
        // The Cyrillic glyph comes from an installed Latin face; the bundled
        // base-14 faces have none.
        let cyrillic_a = face.raw_get_char_index(0x0410);
        if cyrillic_a == 0 {
            eprintln!("no installed Latin face maps U+0410; skipping the Cyrillic glyph");
            return;
        }
        assert_eq!(face.code_to_gid.get(66), Some(&cyrillic_a));
        assert_eq!(
            face.code_to_gid.get(67),
            Some(&face.raw_get_char_index(u32::from('C')))
        );
    }

    /// An embedded non-symbolic TrueType font whose `Differences` array
    /// shifts the digit names off their ASCII codes (`0x33 /zero /one …`)
    /// draws the glyph each name selects through the (3,1) cmap, not the
    /// glyph the program's cmap assigns to the raw code.
    #[test]
    fn embedded_truetype_differences_shift_digits() {
        const DIGITS: [&str; 10] = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        let names: Vec<(usize, &str)> = (0x33..).zip(DIGITS).collect();
        let desc = simple(
            PdfFontKind::TrueType,
            Some(ahem()),
            diffs(&names),
            BaseEncoding::FontProgram,
        );
        let mut cache = cache();
        let face = face(&mut cache, desc);
        for (offset, digit) in ('0'..='9').enumerate() {
            let code = 0x33 + offset;
            let expected = face.raw_get_char_index(u32::from(digit));
            assert_ne!(expected, 0, "font maps U+{:04X}", u32::from(digit));
            assert_eq!(
                face.code_to_gid.get(code),
                Some(&expected),
                "code {code:#04x} draws {digit:?}"
            );
        }
    }

    /// An embedded TrueType subset with `post` names but no `cmap` and no
    /// `Encoding` addresses its glyphs by index: code 7 draws glyph 7, not
    /// the `.notdef` a Unicode lookup of U+0007 through the names gives.
    #[test]
    fn cmapless_truetype_without_encoding_reads_codes_as_glyph_indices() {
        let desc = simple(
            PdfFontKind::TrueType,
            Some(ahem_without_cmap()),
            empty_diffs(),
            BaseEncoding::FontProgram,
        );
        let mut cache = cache();
        let face = face(&mut cache, desc);
        assert!(face.program().lacks_cmap());
        assert_eq!(face.raw_get_name_index("dollar"), 7);
        for code in [2usize, 3, 7, 0x41, 0xFF] {
            assert_eq!(face.code_to_gid[code] as usize, code, "code {code:#04x}");
        }
    }

    /// A cmap-less subset under a declared encoding resolves each named
    /// code through the `post` table: `Differences` first, then the base
    /// encoding; a code whose name the program lacks keeps its glyph index.
    #[test]
    fn cmapless_truetype_resolves_declared_names_through_post() {
        let desc = simple(
            PdfFontKind::TrueType,
            Some(ahem_without_cmap()),
            diffs(&[(0x41, "dollar"), (0x42, "nosuchglyph")]),
            BaseEncoding::WinAnsi,
        );
        let mut cache = cache();
        let face = face(&mut cache, desc);
        let exclam = face.raw_get_name_index("exclam");
        assert_ne!(exclam, 0x21, "ahem's glyph order is not its code order");
        assert_eq!(face.code_to_gid[0x41], 7, "Differences name");
        assert_eq!(face.code_to_gid[0x21], exclam, "WinAnsi name");
        assert_eq!(face.code_to_gid[0x42], 0x42, "unknown name keeps the index");
        assert_eq!(face.code_to_gid[0x05], 0x05, "unnamed code keeps the index");
    }

    /// A composite font without a program and of no known collection draws
    /// each code its `ToUnicode` map covers by Unicode value: Identity-H
    /// code 3 → `H` selects the substitute's `H`, while an unmapped code
    /// keeps the identity reading.
    #[test]
    fn non_embedded_cid_font_resolves_codes_through_to_unicode() {
        let to_unicode = parse_cmap(
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
              1 beginbfchar\n<0003> <0048>\nendbfchar\n",
        )
        .expect("parses");
        let mut cache = cache();
        let face = face(
            &mut cache,
            composite_substitute(Some(to_unicode), SubstituteStyle::default()),
        );
        let h = face.raw_get_char_index(u32::from('H'));
        assert_ne!(h, 0);
        assert_eq!(face.code_to_gid[3], h);
        assert_eq!(
            face.code_to_gid[43], 43,
            "unmapped CIDs keep the identity reading"
        );
    }

    /// A composite font without a program, of a known character
    /// collection, draws through a CID-indexed table built from the
    /// collection's CID → Unicode mapping and the substitute's charmap:
    /// half-width `A` (Japan1 CID 264) selects the substitute's `A`.
    #[test]
    fn non_embedded_cid_font_resolves_cids_through_unicode() {
        let japanese = SubstituteStyle {
            script: Some(Script::Japanese),
            ..SubstituteStyle::default()
        };
        let mut cache = cache();
        let face = face(&mut cache, composite_substitute(None, japanese));
        assert_eq!(face.code_to_gid.len(), 1 << 16);
        let a = face.raw_get_char_index(u32::from('A'));
        assert_ne!(a, 0);
        assert_eq!(face.code_to_gid[264], a);
        assert_eq!(face.code_to_gid[34], a);
        assert_eq!(face.code_to_gid[0], 0);
        // The vertical-form CIDs a `-V` `CMap` selects (long-vowel mark
        // 7891, ideographic comma 7887, wave dash 7894) draw the vertical
        // form of their character, in horizontal mode too.  The forms come
        // from an installed Japanese face; the bundled faces have none.
        if face.raw_get_char_index(0x30FC) == 0 {
            eprintln!("no installed Japanese face maps U+30FC; skipping the vertical forms");
            return;
        }
        for (cid, unicode) in [(7891, 0x30FC), (7887, 0x3001), (7894, 0x301C)] {
            let gid = face.raw_get_char_index(unicode);
            let vertical = face.program().vertical_gid(gid);
            assert_ne!(gid, 0, "U+{unicode:04X}");
            assert_eq!(face.code_to_gid[cid], vertical, "CID {cid}");
        }
    }
}
