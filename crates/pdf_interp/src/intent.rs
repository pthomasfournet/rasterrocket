//! Rendering intents (ISO 32000-1 §8.6.5.8) and the set of intents a page's
//! marking operators were painted under.

/// A colour rendering intent: how colours outside the destination gamut,
/// and the media white point, are mapped by a colour conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RenderingIntent {
    /// Colorimetric match relative to the media white points; colours out
    /// of gamut are clipped.  The initial intent of a graphics state.
    #[default]
    RelativeColorimetric,
    /// Colorimetric match of the colours as measured, the source white
    /// reproduced on the destination medium.
    AbsoluteColorimetric,
    /// Pleasing reproduction of the whole gamut, compressed to fit.
    Perceptual,
    /// Saturation preserved at the expense of hue and lightness accuracy.
    Saturation,
}

impl RenderingIntent {
    /// Every intent, in the order of [`Self::bit`].
    pub const ALL: [Self; 4] = [
        Self::RelativeColorimetric,
        Self::AbsoluteColorimetric,
        Self::Perceptual,
        Self::Saturation,
    ];

    /// The intent of the PDF name `name` (without the `/`).  An unknown
    /// name is relative colorimetric (§8.6.5.8).
    #[must_use]
    pub const fn from_pdf_name(name: &[u8]) -> Self {
        match name {
            b"AbsoluteColorimetric" => Self::AbsoluteColorimetric,
            b"Perceptual" => Self::Perceptual,
            b"Saturation" => Self::Saturation,
            _ => Self::RelativeColorimetric,
        }
    }

    /// The PDF name of the intent (without the `/`).
    #[must_use]
    pub const fn pdf_name(self) -> &'static str {
        match self {
            Self::RelativeColorimetric => "RelativeColorimetric",
            Self::AbsoluteColorimetric => "AbsoluteColorimetric",
            Self::Perceptual => "Perceptual",
            Self::Saturation => "Saturation",
        }
    }

    /// The position of the intent in [`Self::ALL`].
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::RelativeColorimetric => 0,
            Self::AbsoluteColorimetric => 1,
            Self::Perceptual => 2,
            Self::Saturation => 3,
        }
    }

    /// The colour-management module's form of the intent.
    pub(crate) const fn cms(self) -> moxcms::RenderingIntent {
        match self {
            Self::RelativeColorimetric => moxcms::RenderingIntent::RelativeColorimetric,
            Self::AbsoluteColorimetric => moxcms::RenderingIntent::AbsoluteColorimetric,
            Self::Perceptual => moxcms::RenderingIntent::Perceptual,
            Self::Saturation => moxcms::RenderingIntent::Saturation,
        }
    }

    /// The intent's bit in an [`IntentSet`].
    pub(crate) const fn bit(self) -> u8 {
        match self {
            Self::RelativeColorimetric => 1,
            Self::AbsoluteColorimetric => 2,
            Self::Perceptual => 4,
            Self::Saturation => 8,
        }
    }
}

impl std::fmt::Display for RenderingIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.pdf_name())
    }
}

/// A set of rendering intents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IntentSet(u8);

impl IntentSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Add `intent`.
    pub const fn insert(&mut self, intent: RenderingIntent) {
        self.0 |= intent.bit();
    }

    /// Whether `intent` is in the set.
    #[must_use]
    pub const fn contains(self, intent: RenderingIntent) -> bool {
        self.0 & intent.bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set's one intent, when it holds exactly one.
    #[must_use]
    pub fn single(self) -> Option<RenderingIntent> {
        let mut members = self.iter();
        let first = members.next()?;
        members.next().is_none().then_some(first)
    }

    /// The intents of the set, in the order of [`RenderingIntent::ALL`].
    pub fn iter(self) -> impl Iterator<Item = RenderingIntent> {
        RenderingIntent::ALL
            .into_iter()
            .filter(move |&i| self.contains(i))
    }

    /// The union of the two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::fmt::Display for IntentSet {
    /// The intents' names, separated by `, `.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, intent) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(intent.pdf_name())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_intents_and_unknown_ones_to_relative() {
        for (i, intent) in RenderingIntent::ALL.into_iter().enumerate() {
            assert_eq!(intent.index(), i);
            assert_eq!(
                RenderingIntent::from_pdf_name(intent.pdf_name().as_bytes()),
                intent
            );
        }
        assert_eq!(
            RenderingIntent::from_pdf_name(b"Vivid"),
            RenderingIntent::RelativeColorimetric
        );
    }

    #[test]
    fn a_set_reports_its_single_member() {
        let mut set = IntentSet::EMPTY;
        assert_eq!(set.single(), None);
        set.insert(RenderingIntent::Perceptual);
        set.insert(RenderingIntent::Perceptual);
        assert_eq!(set.single(), Some(RenderingIntent::Perceptual));
        set.insert(RenderingIntent::Saturation);
        assert_eq!(set.single(), None);
        assert_eq!(set.to_string(), "Perceptual, Saturation");
        assert_eq!(
            set.union(IntentSet::EMPTY).iter().count(),
            2,
            "union keeps both"
        );
    }
}
