//! PDF content stream parsing: tokenizer → operand stack → typed operators.

use std::borrow::Cow;

pub(crate) mod operands;
pub(crate) mod operator;
pub(crate) mod tokenizer;

pub use operator::{Operator, TextArrayElement};
pub use tokenizer::{InlineComponents, Tokenizer};

/// Parse a raw content stream byte slice into a `Vec` of decoded operators,
/// whose names and strings borrow from `src` where they need no decoding.
///
/// - Unrecognised operator keywords become [`Operator::Unknown`].
/// - Operand tokens that do not match the expected type for an operator are
///   treated as zero/empty (lenient, matching Acrobat's behaviour).
/// - Stray operands with no following operator are silently discarded.
///
/// An inline image whose colour space is named from the resources has its
/// data length computed only through [`parse_with`].
#[must_use]
pub fn parse(src: &[u8]) -> Vec<Operator<'_>> {
    parse_with(src, None)
}

/// [`parse`], resolving an inline image's named colour space to its
/// component count through `components` (see
/// [`Tokenizer::with_inline_components`]).
#[must_use]
pub fn parse_with<'a>(
    src: &'a [u8],
    components: Option<InlineComponents<'_>>,
) -> Vec<Operator<'a>> {
    let mut operands: Vec<tokenizer::Token<'_>> = Vec::new();
    let mut ops = Vec::new();

    for token in Tokenizer::new(src).with_inline_components(components) {
        match token {
            tokenizer::Token::Op(kw) => {
                ops.push(operator::decode(kw, &mut operands));
            }
            tokenizer::Token::InlineImage { params, data } => {
                // The tokenizer consumed BI..ID..EI and emits a single token;
                // any preceding operands are stray — clear them.
                operands.clear();
                ops.push(Operator::InlineImage {
                    params: Cow::Borrowed(params),
                    data: Cow::Borrowed(data),
                });
            }
            other => operands.push(other),
        }
    }

    ops
}
