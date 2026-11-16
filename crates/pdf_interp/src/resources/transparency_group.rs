//! Transparency group dictionaries (ISO 32000-1 §11.6.6) of form
//! `XObject`s and pages.

use pdf::{Dictionary, Document, Object};

use super::{GroupSpace, resolve_dict};

/// The attributes of a transparency group dictionary (PDF §11.6.6), of a
/// form `XObject` or of a page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransparencyGroupParams {
    /// `I`: the group is isolated, so blend modes inside it do not see the
    /// backdrop under it (default: false).
    pub isolated: bool,
    /// `K`: the group is a knockout group, in which each object composites
    /// with the group's initial backdrop rather than over the objects
    /// before it (default: false).
    pub knockout: bool,
    /// `CS`: the colour space the group's colours convert into and
    /// composite in; `None` when absent or not a group colour space, the
    /// group then compositing in its parent's.
    pub space: Option<GroupSpace>,
}

/// The transparency group attributes of `dict`, a form `XObject` stream
/// dictionary or a page dictionary: `Some` when its `Group` has
/// `/S /Transparency`, `None` otherwise.
#[must_use]
pub fn read_transparency_group(
    doc: &Document,
    dict: &Dictionary,
) -> Option<TransparencyGroupParams> {
    let grp_obj = dict.get(b"Group")?;
    let grp = resolve_dict(doc, grp_obj)?;
    if grp.get(b"S")?.as_name()? != b"Transparency" {
        return None;
    }
    let bool_flag = |key: &[u8]| grp.get(key).and_then(Object::as_bool).unwrap_or(false);
    Some(TransparencyGroupParams {
        isolated: bool_flag(b"I"),
        knockout: bool_flag(b"K"),
        space: grp.get(b"CS").and_then(|cs| GroupSpace::read(doc, cs)),
    })
}
