//! Operand stack extraction helpers.
//!
//! PDF content stream operands accumulate on a stack left-to-right, so the
//! last operand pushed is the rightmost value written in the stream.
//! All `pop_*` helpers consume the topmost (rightmost) element.

use super::tokenizer::Token;

/// A number token's value: a number, or a boolean as 0 or 1.
const fn numeric(token: &Token<'_>) -> Option<f64> {
    match token {
        Token::Number(n) => Some(*n),
        Token::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Pop the topmost token as a number. Returns 0.0 on underflow or wrong type.
pub fn pop_f64(stack: &mut Vec<Token<'_>>) -> f64 {
    stack.pop().as_ref().and_then(numeric).unwrap_or(0.0)
}

/// Pop the topmost token as a truncated integer. Returns 0 on underflow.
#[expect(
    clippy::cast_possible_truncation,
    reason = "f64-to-i32 `as` truncates toward zero and saturates; callers map \
              out-of-range values to their defaults"
)]
pub fn pop_i32(stack: &mut Vec<Token<'_>>) -> i32 {
    pop_f64(stack) as i32
}

/// Pop the topmost token as a name or string byte vector.
/// Returns an empty `Vec` on underflow or wrong type.
pub fn pop_name(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::Name(n)) => n.to_vec(),
        Some(Token::String(s)) => s,
        _ => Vec::new(),
    }
}

/// Pop the topmost token as a string byte vector.
/// Returns an empty `Vec` on underflow or wrong type.
pub fn pop_string(stack: &mut Vec<Token<'_>>) -> Vec<u8> {
    match stack.pop() {
        Some(Token::String(s)) => s,
        Some(Token::Name(n)) => n.to_vec(),
        _ => Vec::new(),
    }
}

/// Drain the contiguous run of numbers (and booleans) at the top of the
/// stack, returning them in stream order (leftmost first).
///
/// The run stops at the topmost non-numeric token, which stays on the
/// stack with everything beneath it.
pub fn drain_numbers(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    let run_start = stack
        .iter()
        .rposition(|t| numeric(t).is_none())
        .map_or(0, |i| i + 1);
    stack
        .drain(run_start..)
        .filter_map(|t| numeric(&t))
        .collect()
}

/// Result of parsing `scn` / `SCN` operands: either a pattern name or numeric
/// colour components.
///
/// PDF §8.6.8: `scn` operands are zero or more numbers followed by an optional
/// name.  When the last token is a name the colour space is Pattern and the name
/// selects the pattern resource.  For all other colour spaces only numbers appear.
#[non_exhaustive]
pub enum ScnOperands {
    /// `/PatternName [c1 … cn]` — Pattern colour space; name is the resource key,
    /// components are the underlying non-Pattern colour (may be empty for
    /// `PaintType` 1 / coloured patterns).
    Pattern {
        /// Pattern resource name (key into the page `Pattern` resource dict).
        name: Vec<u8>,
        /// Optional tint components for uncoloured (`PaintType` 2) patterns.
        components: Vec<f64>,
    },
    /// Plain numeric components for non-pattern colour spaces.
    Components(Vec<f64>),
}

/// Parse the operand stack for `scn` / `SCN`.
///
/// The PDF grammar is `c1 … cn name?`, where the optional trailing `name`
/// selects a Pattern resource.  The components are the run of numbers
/// directly beneath the name, or at the top of the stack when there is no
/// name (see [`drain_numbers`]).
pub fn pop_scn(stack: &mut Vec<Token<'_>>) -> ScnOperands {
    let name = if matches!(stack.last(), Some(Token::Name(_))) {
        match stack.pop() {
            Some(Token::Name(n)) => Some(n.to_vec()),
            _ => None,
        }
    } else {
        None
    };
    let components = drain_numbers(stack);
    match name {
        Some(name) => ScnOperands::Pattern { name, components },
        None => ScnOperands::Components(components),
    }
}

/// Pop an array on top of the stack as its numeric elements, in order.
///
/// Returns an empty `Vec`, leaving the stack unchanged, when the top is not
/// an array.
pub fn pop_number_array(stack: &mut Vec<Token<'_>>) -> Vec<f64> {
    if !matches!(stack.last(), Some(Token::Array(_))) {
        return Vec::new();
    }
    match stack.pop() {
        Some(Token::Array(arr)) => arr.iter().filter_map(numeric).collect(),
        _ => Vec::new(),
    }
}

/// Pop the top `N` numbers in stream order (leftmost first); each missing
/// or non-numeric operand reads as 0 (see [`pop_f64`]).
pub fn pop_n<const N: usize>(stack: &mut Vec<Token<'_>>) -> [f64; N] {
    let mut out = [0.0; N];
    for v in out.iter_mut().rev() {
        *v = pop_f64(stack);
    }
    out
}

/// Pop the top 2 numbers as `(first, second)` in stream order.
pub fn pop2(stack: &mut Vec<Token<'_>>) -> (f64, f64) {
    pop_n::<2>(stack).into()
}

/// Pop the top 3 numbers as `(first, second, third)` in stream order.
pub fn pop3(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64) {
    pop_n::<3>(stack).into()
}

/// Pop the top 4 numbers as `(first, second, third, fourth)` in stream order.
pub fn pop4(stack: &mut Vec<Token<'_>>) -> (f64, f64, f64, f64) {
    pop_n::<4>(stack).into()
}

/// Pop the top 6 numbers as a PDF matrix `[a b c d e f]` in stream order.
pub fn pop_matrix(stack: &mut Vec<Token<'_>>) -> [f64; 6] {
    pop_n(stack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_components_are_the_numbers_at_the_top_of_the_stack() {
        let mut stack = vec![
            Token::String(b"stray".to_vec()),
            Token::Number(1.0),
            Token::Number(0.5),
            Token::Number(0.0),
        ];
        let ScnOperands::Components(c) = pop_scn(&mut stack) else {
            panic!("no pattern name");
        };
        assert_eq!(c, [1.0, 0.5, 0.0]);
        assert_eq!(stack, [Token::String(b"stray".to_vec())]);

        let mut stack = vec![
            Token::Name(b"stray".as_slice().into()),
            Token::Number(0.25),
            Token::Name(b"P0".as_slice().into()),
        ];
        let ScnOperands::Pattern { name, components } = pop_scn(&mut stack) else {
            panic!("a pattern name");
        };
        assert_eq!(name, b"P0");
        assert_eq!(components, [0.25]);
    }

    #[test]
    fn pop_n_reads_stream_order_and_zero_fills() {
        let mut stack = vec![Token::Number(1.0), Token::Number(2.0)];
        assert_eq!(pop_n::<3>(&mut stack), [0.0, 1.0, 2.0]);
    }
}
