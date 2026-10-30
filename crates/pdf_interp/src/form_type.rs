//! Interactive form technology of a document: none, an `AcroForm`, or an
//! XFA form (ISO 32000-1 §12.7.8; XFA is deprecated in ISO 32000-2).

use pdf::Document;

use crate::resources::dict_ext::DictExt;

/// The interactive form a document carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FormType {
    /// No `/AcroForm` in the catalog.
    #[default]
    NoForm,
    /// An `/AcroForm` without `/XFA`.
    AcroForm,
    /// An `/AcroForm` with `/XFA` and catalog `/NeedsRendering true`: a
    /// dynamic XFA form, whose pages are laid out by an XFA processor.  The
    /// page content in the file is a placeholder, typically a notice asking
    /// for an XFA-capable viewer, and is what is rendered.
    XfaFull,
    /// An `/AcroForm` with `/XFA` whose pages are ordinary page content
    /// (static XFA): the `AcroForm` fields render as usual.
    XfaForeground,
}

/// The form type of `doc`, from the catalog's `/AcroForm` (and its `/XFA`
/// entry) and `/NeedsRendering`.  [`FormType::NoForm`] when the catalog is
/// unreadable.
#[must_use]
pub fn form_type(doc: &Document) -> FormType {
    let Ok(catalog) = doc.catalog() else {
        return FormType::NoForm;
    };
    let Some(acroform) = catalog
        .get(b"AcroForm")
        .and_then(|o| crate::resources::resolve_dict(doc, o))
    else {
        return FormType::NoForm;
    };
    if acroform.get(b"XFA").is_none() {
        FormType::AcroForm
    } else if catalog.get_bool(b"NeedsRendering") == Some(true) {
        FormType::XfaFull
    } else {
        FormType::XfaForeground
    }
}

/// Warn when `doc` is a dynamic XFA form: what renders is its placeholder
/// page content.
pub(crate) fn warn_if_dynamic_xfa(doc: &Document) {
    if form_type(doc) == FormType::XfaFull {
        log::warn!(
            "document is a dynamic XFA form (/NeedsRendering true); XFA is not processed — \
             the rendered pages are the placeholder content stored in the file"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{FormType, form_type};
    use crate::test_helpers::make_doc;

    #[test]
    fn the_catalog_entries_decide_the_form_type() {
        let cases = [
            ("", FormType::NoForm),
            (" /AcroForm << /Fields [] >>", FormType::AcroForm),
            (
                " /AcroForm << /Fields [] /XFA [] >>",
                FormType::XfaForeground,
            ),
            (
                " /AcroForm << /Fields [] /XFA [] >> /NeedsRendering true",
                FormType::XfaFull,
            ),
            (
                " /AcroForm << /Fields [] /XFA [] >> /NeedsRendering false",
                FormType::XfaForeground,
            ),
        ];
        for (catalog, expected) in cases {
            assert_eq!(form_type(&make_doc(catalog)), expected, "{catalog}");
        }
    }
}
