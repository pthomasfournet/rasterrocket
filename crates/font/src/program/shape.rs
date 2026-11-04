//! Shaping: the glyphs and positions an sfnt's OpenType layout tables
//! (`GSUB`, `GPOS`) give a run of text, with the default features of the
//! run's script.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use harfrust::font::Font;
use harfrust::{Buffer, Direction, Script, ShapeOptions, ShapePlan, ShaperFont};

use super::FontProgram;

/// One glyph of a shaped run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    /// The glyph's composite-font selector, as the `*_by_gid` methods of
    /// [`crate::face::FontFace`] take it (see [`FontProgram::cid_to_gid`]).
    pub selector: u32,
    /// The advance after the glyph, in em.
    pub advance: f64,
    /// The displacement of the glyph from its pen position, in em, y up.
    pub offset: (f64, f64),
}

/// Why a run was not shaped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// The program is not an sfnt: it has no OpenType layout tables.
    NotSfnt,
    /// Shaping produced the glyph index held, which has no selector:
    /// a CID-keyed program whose charset omits it.
    NoSelector(u32),
    /// The shaper read no font from the program.  The condition is a
    /// property of the program, so every run of it fails the same way.
    NoShaperFont,
    /// The shaper refused the run, for the reason held.
    Refused(harfrust::ShapeError),
    /// The run exceeded the shaper's length, operation or nesting limits,
    /// or its buffer could not grow; the glyphs it holds are incomplete.
    Exhausted,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSfnt => f.write_str("the program is not an sfnt"),
            Self::NoSelector(gid) => write!(f, "shaped glyph {gid} has no selector"),
            Self::NoShaperFont => f.write_str("the shaper read no font from the program"),
            Self::Refused(e) => write!(f, "the shaper refused the run: {e}"),
            Self::Exhausted => f.write_str("the run exceeded the shaper's limits"),
        }
    }
}

impl std::error::Error for ShapeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(e) => Some(e),
            _ => None,
        }
    }
}

/// The outcome of a shaping call that returned `result` and left its
/// buffer's allocation flag at `allocation_successful`: a run is shaped
/// only when the call succeeded and the buffer never ran out of room.
fn shaping_outcome(
    result: Result<(), harfrust::ShapeError>,
    allocation_successful: bool,
) -> Result<(), ShapeError> {
    result.map_err(ShapeError::Refused)?;
    if allocation_successful {
        Ok(())
    } else {
        Err(ShapeError::Exhausted)
    }
}

/// The script and direction a shape plan is compiled for.
type PlanKey = (Option<Script>, Direction);

/// The shaping state of an sfnt program: the shaper's view of the font,
/// which caches its layout tables, and a shape plan per script and
/// direction, each compiled once.
pub(super) struct Shaping {
    font: Option<Font>,
    plans: Mutex<HashMap<PlanKey, Arc<ShapePlan>>>,
}

impl FontProgram {
    /// `text`, in logical order, shaped right to left when `rtl`, else left
    /// to right.  The script is the one `text`'s characters determine.  The
    /// glyphs are in visual order, left to right.
    ///
    /// # Errors
    ///
    /// [`ShapeError::NotSfnt`] for a program that is not an sfnt;
    /// [`ShapeError::NoShaperFont`] when the shaper reads no font from it;
    /// [`ShapeError::Refused`] when the shaper refuses the run;
    /// [`ShapeError::Exhausted`] when the run exceeds the shaper's limits;
    /// [`ShapeError::NoSelector`] when a shaped glyph has no selector.
    pub fn shape(&self, text: &str, rtl: bool) -> Result<Vec<ShapedGlyph>, ShapeError> {
        let face_index = self.sfnt_face_index().ok_or(ShapeError::NotSfnt)?;
        let shaping = self.shaper.get_or_init(|| {
            let data: Arc<dyn AsRef<[u8]> + Send + Sync> = self.shared_data();
            Shaping {
                font: Font::new(harfrust::font::Blob::from(data), face_index),
                plans: Mutex::new(HashMap::new()),
            }
        });
        let font = shaping.font.as_ref().ok_or(ShapeError::NoShaperFont)?;
        let shaper = ShaperFont::new(font);
        let mut buffer = Buffer::new();
        buffer.push_str(text);
        buffer.set_direction(if rtl {
            Direction::RightToLeft
        } else {
            Direction::LeftToRight
        });
        buffer.guess_segment_properties();
        // A plan depends on the script, direction and language; runs carry
        // no language, so the first two key it.
        let plan = buffer.language().is_none().then(|| {
            let key = (buffer.script(), buffer.direction());
            let mut plans = shaping.plans.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(
                plans
                    .entry(key)
                    .or_insert_with(|| Arc::new(ShapePlan::new(font, key.1, key.0, None, &[]))),
            )
        });
        let result = harfrust::shape(
            &shaper,
            &mut buffer,
            ShapeOptions::new().plan(plan.as_deref()),
        );
        shaping_outcome(result, buffer.allocation_successful())?;
        let glyphs = buffer;
        let em = |units: i32| f64::from(units) / self.upem;
        glyphs
            .glyph_infos()
            .iter()
            .zip(glyphs.glyph_positions())
            .map(|(info, pos)| {
                Ok(ShapedGlyph {
                    selector: self
                        .gid_to_selector(info.glyph_id)
                        .ok_or(ShapeError::NoSelector(info.glyph_id))?,
                    advance: em(pos.x_advance),
                    offset: (em(pos.x_offset), em(pos.y_offset)),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{FontProgram, ShapeError, shaping_outcome};

    #[test]
    fn a_run_is_shaped_only_when_the_call_succeeds_with_room_to_spare() {
        assert_eq!(shaping_outcome(Ok(()), true), Ok(()));
        assert_eq!(shaping_outcome(Ok(()), false), Err(ShapeError::Exhausted));
        let refused = harfrust::ShapeError::AlreadyShaped;
        assert_eq!(
            shaping_outcome(Err(refused.clone()), true),
            Err(ShapeError::Refused(refused.clone()))
        );
        assert_eq!(
            shaping_outcome(Err(refused.clone()), false),
            Err(ShapeError::Refused(refused))
        );
    }

    fn ahem() -> FontProgram {
        FontProgram::parse(font_test_data::AHEM.to_vec(), 0).expect("Ahem parses")
    }

    /// Ahem has no layout tables, so shaping keeps its charmap glyphs and
    /// `hmtx` advances, and a second run reuses the cached plan.
    #[test]
    fn a_program_without_layout_tables_keeps_charmap_glyphs() {
        let program = ahem();
        let glyphs = program.shape("Ap", false).expect("an sfnt shapes");
        let expected: Vec<(u32, Option<f64>)> = "Ap"
            .chars()
            .map(|c| {
                let gid = program.char_index(u32::from(c));
                (gid, program.advance(gid))
            })
            .collect();
        let shaped: Vec<(u32, Option<f64>)> = glyphs
            .iter()
            .map(|g| (g.selector, Some(g.advance)))
            .collect();
        assert_eq!(shaped, expected);
        assert!(glyphs.iter().all(|g| g.offset == (0.0, 0.0)));
        assert_eq!(program.shape("Ap", false), Ok(glyphs));
    }

    #[test]
    fn right_to_left_glyphs_come_in_visual_order() {
        let program = ahem();
        let ltr = program.shape("Ap", false).expect("an sfnt shapes");
        let mut rtl = program.shape("Ap", true).expect("an sfnt shapes");
        rtl.reverse();
        assert_eq!(rtl, ltr);
        assert_ne!(
            ltr[0].selector, ltr[1].selector,
            "the test needs two glyphs"
        );
    }
}
