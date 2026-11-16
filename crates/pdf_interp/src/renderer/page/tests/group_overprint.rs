//! Overprint and transparency groups, and overprint under blend modes
//! other than `Normal`.
//!
//! Each case is checked against the separations Ghostscript 10.06
//! (`tiffsep`) renders for the same page, `MuPDF` 1.27 agreeing on the
//! process inks; their values are taken within one count, Ghostscript
//! rounding a tint of 0.5 to 127.
//!
//! A group composite paints every colorant, whatever the overprint
//! parameters in effect when it is painted (ISO 32000-1 Table 149, last
//! row): no record is kept of the colorants a group's objects painted.  An
//! elementary object that overprints under a blend mode other than `Normal`
//! is painted with `CompatibleOverprint` in an implicit non-isolated group
//! composited under that mode (§11.7.4.3), so its unpainted colorants blend
//! with themselves.  Spot colorants take a white-preserving separable mode
//! and `Normal` under any other (§11.7.4.2).

use pdf::Document;

use super::*;

/// `/Spot`, a spot colorant printing as 91 % magenta and 76 % yellow.
fn spot_space() -> String {
    format!(
        "/Spot [/Separation /Spot /DeviceCMYK {}]",
        tint_to_cmyk("0 0.91 0.76 0")
    )
}

/// The overprinting state the group's content sets.
const GS_OP: &str = "/GSop << /OP true /op true /OPM 1 >>";

/// Page `(4, 0)` with the extended graphics state `/P` of entries `state`,
/// and the form `/F`, a transparency group of the attributes `group` whose
/// content is `form`.  Both have `/GSop` and `/Spot`.
fn case_doc(state: &str, group: &str, form: &str) -> Document {
    let resources = format!(
        "/ExtGState << /P << {state} >> {GS_OP} >> /ColorSpace << {} >>",
        spot_space()
    );
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!("<< /Type /Page /Resources << /XObject << /F 5 0 R >> {resources} >> >>"),
        ),
        (
            5,
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Form /BBox [0 0 10 10] \
                     /Group << /S /Transparency {group} >> /Resources << {resources} >>"
                ),
                form,
            ),
        ),
    ])
}

/// Half-tint cyan over the page.
const CYAN: &str = "0.5 0 0 0 k 0 0 10 10 re f";
/// Half-tint spot over the page.
const SPOT_HALF: &str = "/Spot cs 0.5 scn 0 0 10 10 re f";
/// Full-tint spot over the page.
const SPOT_FULL: &str = "/Spot cs 1 scn 0 0 10 10 re f";
/// Black overprinting the left half under nonzero overprint mode.
const K_OP: &str = "/GSop gs 0 0 0 1 k 0 0 5 10 re f";

/// One rendered case: the page's backdrop, its graphics state `/P` set
/// before the last paint, the group's attributes and content, and the
/// last paint of the page.
struct Case {
    name: &'static str,
    backdrop: &'static str,
    state: &'static str,
    group: &'static str,
    form: &'static str,
    paint: &'static str,
    /// C, M, Y, K and the spot at `(2, 5)`, in the left half, and at
    /// `(7, 5)`, in the right.
    left: [u8; 5],
    right: [u8; 5],
}

impl Case {
    fn check(&self) {
        let doc = case_doc(self.state, self.group, self.form);
        let content = format!("{} /P gs {}", self.backdrop, self.paint);
        let ink = render_ink(&doc, content.as_bytes());
        for ((x, want), side) in [(2, self.left), (7, self.right)]
            .into_iter()
            .zip(["left", "right"])
        {
            let i = (5 * 10 + x) * 8;
            let got = [ink[i], ink[i + 1], ink[i + 2], ink[i + 3], ink[i + 4]];
            assert!(
                got.iter().zip(&want).all(|(g, w)| g.abs_diff(*w) <= 1),
                "{} {side}: got {got:?}, want {want:?}",
                self.name
            );
            assert_eq!(
                &ink[i + 5..i + 8],
                &[0; 3],
                "{} {side}: spare slots",
                self.name
            );
        }
    }
}

/// The group's own overprint and blend mode applied to a group that
/// overprints black inside it.
const GROUP_CASES: &[Case] = &[
    Case {
        name: "non-isolated",
        backdrop: CYAN,
        state: "",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [127, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated",
        backdrop: CYAN,
        state: "",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, overprint at Do",
        backdrop: CYAN,
        state: "/OP true /op true /OPM 1",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, overprint mode 0 at Do",
        backdrop: CYAN,
        state: "/OP true /op true /OPM 0",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, overprint at Do only",
        backdrop: CYAN,
        state: "/OP true /op true /OPM 1",
        group: "/I true",
        form: "0 0 0 1 k 0 0 5 10 re f",
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "non-isolated, Multiply",
        backdrop: CYAN,
        state: "/BM /Multiply",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [191, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "non-isolated, Multiply, overprint at Do",
        backdrop: CYAN,
        state: "/BM /Multiply /OP true /op true /OPM 1",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [191, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, Multiply, overprint at Do",
        backdrop: CYAN,
        state: "/BM /Multiply /OP true /op true /OPM 1",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [127, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, Screen, overprint at Do",
        backdrop: CYAN,
        state: "/BM /Screen /OP true /op true /OPM 1",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 0, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, Screen",
        backdrop: CYAN,
        state: "/BM /Screen",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 0, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "non-isolated, Screen",
        backdrop: CYAN,
        state: "/BM /Screen",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [63, 0, 0, 0, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "non-isolated, Screen, overprint at Do",
        backdrop: CYAN,
        state: "/BM /Screen /OP true /op true /OPM 1",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [63, 0, 0, 0, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "non-isolated knockout",
        backdrop: CYAN,
        state: "",
        group: "/I false /K true",
        form: K_OP,
        paint: "/F Do",
        left: [127, 0, 0, 255, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated over a spot",
        backdrop: SPOT_FULL,
        state: "",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [0, 0, 0, 0, 255],
    },
    Case {
        name: "isolated over a spot, overprint at Do",
        backdrop: SPOT_FULL,
        state: "/OP true /op true /OPM 1",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [0, 0, 0, 0, 255],
    },
    Case {
        name: "non-isolated over a spot",
        backdrop: SPOT_FULL,
        state: "",
        group: "/I false",
        form: K_OP,
        paint: "/F Do",
        left: [0, 0, 0, 255, 255],
        right: [0, 0, 0, 0, 255],
    },
    Case {
        name: "isolated, half alpha, overprint at Do",
        backdrop: CYAN,
        state: "/OP true /op true /OPM 1 /ca 0.5",
        group: "/I true",
        form: K_OP,
        paint: "/F Do",
        left: [63, 0, 0, 128, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "isolated, black and magenta, overprint at Do",
        backdrop: CYAN,
        state: "/OP true /op true /OPM 1",
        group: "/I true",
        form: "/GSop gs 0 0 0 1 k 0 0 5 10 re f 0 0.5 0 0 k 5 0 5 10 re f",
        paint: "/F Do",
        left: [0, 0, 0, 255, 0],
        right: [0, 127, 0, 0, 0],
    },
];

/// A group composite paints every colorant: the overprint parameters in
/// effect when it is painted are not consulted, an isolated group knocks
/// out the colorants its objects left, and a group's blend mode blends
/// every colorant of a non-isolated group's result with the backdrop.
#[test]
fn group_composites_paint_every_colorant() {
    for case in GROUP_CASES {
        case.check();
    }
}

/// Elementary objects under a blend mode other than `Normal`: an
/// overprinting object's unpainted colorants blend with themselves, and
/// the spot colorant is blended by a white-preserving separable mode and
/// painted as under `Normal` otherwise.
const BLEND_CASES: &[Case] = &[
    Case {
        name: "Screen black overprinting cyan",
        backdrop: CYAN,
        state: "/BM /Screen /OP true /op true /OPM 1",
        group: "",
        form: "",
        paint: "0 0 0 1 k 0 0 5 10 re f",
        left: [63, 0, 0, 0, 0],
        right: [127, 0, 0, 0, 0],
    },
    Case {
        name: "Multiply magenta and black overprinting cyan and magenta",
        backdrop: "0.5 0.5 0 0 k 0 0 10 10 re f",
        state: "/BM /Multiply /OP true /op true /OPM 1",
        group: "",
        form: "",
        paint: "0 0.5 0 1 k 0 0 5 10 re f",
        left: [191, 191, 0, 255, 0],
        right: [127, 127, 0, 0, 0],
    },
    Case {
        name: "Multiply spot",
        backdrop: SPOT_HALF,
        state: "/BM /Multiply",
        group: "",
        form: "",
        paint: "/Spot cs 1 scn 0 0 5 10 re f",
        left: [0, 0, 0, 0, 255],
        right: [0, 0, 0, 0, 127],
    },
    Case {
        name: "Multiply half spot",
        backdrop: SPOT_HALF,
        state: "/BM /Multiply",
        group: "",
        form: "",
        paint: "/Spot cs 0.5 scn 0 0 5 10 re f",
        left: [0, 0, 0, 0, 191],
        right: [0, 0, 0, 0, 127],
    },
    Case {
        name: "Hue spot",
        backdrop: SPOT_HALF,
        state: "/BM /Hue",
        group: "",
        form: "",
        paint: "/Spot cs 0.25 scn 0 0 5 10 re f",
        left: [0, 0, 0, 0, 63],
        right: [0, 0, 0, 0, 127],
    },
    Case {
        name: "Difference spot",
        backdrop: SPOT_HALF,
        state: "/BM /Difference",
        group: "",
        form: "",
        paint: "/Spot cs 0.25 scn 0 0 5 10 re f",
        left: [255, 255, 255, 255, 63],
        right: [0, 0, 0, 0, 127],
    },
    Case {
        name: "Screen cyan over a spot",
        backdrop: SPOT_HALF,
        state: "/BM /Screen",
        group: "",
        form: "",
        paint: "1 0 0 0 k 0 0 5 10 re f",
        left: [0, 0, 0, 0, 0],
        right: [0, 0, 0, 0, 127],
    },
    Case {
        name: "Screen black overprinting a spot",
        backdrop: SPOT_HALF,
        state: "/BM /Screen /OP true /op true /OPM 1",
        group: "",
        form: "",
        paint: "0 0 0 1 k 0 0 5 10 re f",
        left: [0, 0, 0, 0, 63],
        right: [0, 0, 0, 0, 127],
    },
];

#[test]
fn blend_modes_composite_unpainted_and_spot_colorants() {
    for case in BLEND_CASES {
        case.check();
    }
}
