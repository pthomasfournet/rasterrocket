//! Type 4 (PostScript calculator) functions, PDF §7.10.5.
//!
//! A program is parsed once into a token tree ([`Program::parse`]) and run
//! any number of times ([`Program::run`]).  The interpreter implements the
//! operator set of §7.10.5 — arithmetic, relational, boolean and bitwise,
//! stack, and the `if`/`ifelse` conditionals.  The language has no loops,
//! definitions, strings or arrays, so a run executes each token of the
//! program at most once: the parser's bounds on the token count
//! ([`MAX_PS_TOKENS`]) and the nesting depth ([`MAX_PS_DEPTH`]) bound the work
//! and the native recursion of every run.  Only the operand stack can grow
//! faster than one entry per token (`copy` doubles it), and it is bounded by
//! [`MAX_PS_STACK`].

use std::fmt;

/// A parsed Type 4 program: the token tree of its outermost `{ }` block.
#[derive(Debug)]
pub(super) struct Program(Vec<PsTok>);

impl Program {
    /// Parse the program text `src`, a single `{ }` block.
    ///
    /// # Errors
    ///
    /// A [`ParseError`] naming the first construct that is not a valid
    /// calculator program, or the limit the program exceeds.
    pub(super) fn parse(src: &[u8]) -> Result<Self, ParseError> {
        let lexemes = ps_lex(src);
        if lexemes.len() > MAX_PS_TOKENS {
            return Err(ParseError::TooLong(lexemes.len()));
        }
        let body = match lexemes.split_first() {
            Some((&first, body)) if first == b"{" => body,
            _ => return Err(ParseError::NoProcedure),
        };
        let (toks, rest) = parse_block(body, 0)?;
        if !rest.is_empty() {
            return Err(ParseError::TrailingTokens);
        }
        Ok(Self(toks))
    }

    /// Run the program on `inputs` (already clipped to `Domain`) and return
    /// the top `n` stack values, deepest first.  `None` on any execution
    /// error, on fewer than `n` results, or on a boolean result.
    pub(super) fn run(&self, inputs: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut stack: Vec<PsVal> = inputs.iter().map(|&t| PsVal::Num(t)).collect();
        exec(&self.0, &mut stack)?;
        let first = stack.len().checked_sub(n)?;
        stack[first..]
            .iter()
            .map(|v| match *v {
                PsVal::Num(x) => Some(x),
                PsVal::Bool(_) => None,
            })
            .collect()
    }
}

/// Why a Type 4 program failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ParseError {
    /// The program does not open with `{`.
    NoProcedure,
    /// A `{` has no matching `}`.
    Unbalanced,
    /// Tokens follow the `}` that closes the program.
    TrailingTokens,
    /// A word that is neither a number nor a calculator operator.
    UnknownWord(String),
    /// A numeric literal outside the range of a real number.
    NumberOutOfRange(String),
    /// A procedure that is not an operand of `if` or `ifelse`, or an `if` or
    /// `ifelse` without its procedure operands.
    MisplacedProcedure,
    /// More than [`MAX_PS_TOKENS`] tokens.
    TooLong(usize),
    /// Procedures nested deeper than [`MAX_PS_DEPTH`].
    TooDeep,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoProcedure => f.write_str("the program does not open with `{`"),
            Self::Unbalanced => f.write_str("a `{` has no matching `}`"),
            Self::TrailingTokens => f.write_str("tokens follow the closing `}` of the program"),
            Self::UnknownWord(w) => write!(f, "`{w}` is not a calculator operator"),
            Self::NumberOutOfRange(w) => write!(f, "the number `{w}` is out of range"),
            Self::MisplacedProcedure => {
                f.write_str("a procedure is not the operand of `if` or `ifelse`")
            }
            Self::TooLong(n) => write!(f, "{n} tokens exceed the limit of {MAX_PS_TOKENS}"),
            Self::TooDeep => write!(f, "procedures nest deeper than {MAX_PS_DEPTH}"),
        }
    }
}

/// Upper bound on the tokens of a program.  A run executes each token at
/// most once, so this bounds the work of one evaluation; tint transforms
/// and shading functions run to tens or hundreds of tokens.
pub(super) const MAX_PS_TOKENS: usize = 100_000;

/// Upper bound on `{ }` nesting.  Parsing and execution recurse once per
/// nested procedure; real programs nest a handful of levels.
pub(super) const MAX_PS_DEPTH: usize = 64;

/// Upper bound on the operand stack.  `copy` pushes up to the current depth
/// for one token, so the token bound alone does not bound the stack.
pub(super) const MAX_PS_STACK: usize = 100_000;

/// A value on the operand stack.  Booleans arise from the relational and
/// boolean operators (§7.10.5).
#[derive(Clone, Copy, Debug)]
enum PsVal {
    Num(f64),
    Bool(bool),
}

/// A parsed token: a literal, an operator, or a conditional with its
/// procedure operands.
#[derive(Debug)]
enum PsTok {
    Num(f64),
    Op(PsOp),
    If(Vec<Self>),
    IfElse(Vec<Self>, Vec<Self>),
}

#[derive(Clone, Copy, Debug)]
enum PsOp {
    Abs,
    Add,
    Atan,
    Ceiling,
    Cos,
    Cvi,
    Cvr,
    Div,
    Exp,
    Floor,
    Idiv,
    Ln,
    Log,
    Mod,
    Mul,
    Neg,
    Round,
    Sin,
    Sqrt,
    Sub,
    Truncate,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    And,
    Or,
    Xor,
    Not,
    Bitshift,
    True,
    False,
    Copy,
    Dup,
    Exch,
    Index,
    Pop,
    Roll,
}

/// The operator named `word`; `if` and `ifelse` are resolved by the parser.
const fn parse_op(word: &[u8]) -> Option<PsOp> {
    Some(match word {
        b"abs" => PsOp::Abs,
        b"add" => PsOp::Add,
        b"atan" => PsOp::Atan,
        b"ceiling" => PsOp::Ceiling,
        b"cos" => PsOp::Cos,
        b"cvi" => PsOp::Cvi,
        b"cvr" => PsOp::Cvr,
        b"div" => PsOp::Div,
        b"exp" => PsOp::Exp,
        b"floor" => PsOp::Floor,
        b"idiv" => PsOp::Idiv,
        b"ln" => PsOp::Ln,
        b"log" => PsOp::Log,
        b"mod" => PsOp::Mod,
        b"mul" => PsOp::Mul,
        b"neg" => PsOp::Neg,
        b"round" => PsOp::Round,
        b"sin" => PsOp::Sin,
        b"sqrt" => PsOp::Sqrt,
        b"sub" => PsOp::Sub,
        b"truncate" => PsOp::Truncate,
        b"eq" => PsOp::Eq,
        b"ne" => PsOp::Ne,
        b"gt" => PsOp::Gt,
        b"ge" => PsOp::Ge,
        b"lt" => PsOp::Lt,
        b"le" => PsOp::Le,
        b"and" => PsOp::And,
        b"or" => PsOp::Or,
        b"xor" => PsOp::Xor,
        b"not" => PsOp::Not,
        b"bitshift" => PsOp::Bitshift,
        b"true" => PsOp::True,
        b"false" => PsOp::False,
        b"copy" => PsOp::Copy,
        b"dup" => PsOp::Dup,
        b"exch" => PsOp::Exch,
        b"index" => PsOp::Index,
        b"pop" => PsOp::Pop,
        b"roll" => PsOp::Roll,
        _ => return None,
    })
}

/// PDF white-space characters (ISO 32000-2 Table 1).
const fn is_white(b: u8) -> bool {
    matches!(b, 0x00 | 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

/// Split the program text into words, with `{` and `}` as words of their
/// own.  A `%` comment runs to the end of its line and separates words.
fn ps_lex(src: &[u8]) -> Vec<&[u8]> {
    let ends_word = |c: &u8| is_white(*c) || matches!(c, b'{' | b'}' | b'%');
    let mut out = Vec::new();
    let mut rest = src;
    while let Some((&b, tail)) = rest.split_first() {
        rest = match b {
            b'{' | b'}' => {
                out.push(&rest[..1]);
                tail
            }
            b'%' => {
                let eol = tail.iter().position(|c| matches!(c, b'\n' | b'\r'));
                &tail[eol.unwrap_or(tail.len())..]
            }
            _ if is_white(b) => tail,
            _ => {
                let end = rest.iter().position(ends_word).unwrap_or(rest.len());
                let (word, after) = rest.split_at(end);
                out.push(word);
                after
            }
        };
    }
    out
}

/// Parse the words after a `{` up to its matching `}`; returns the tokens
/// and the words after the `}`.  `depth` is the nesting of this block.
fn parse_block<'w>(
    mut words: &'w [&'w [u8]],
    depth: usize,
) -> Result<(Vec<PsTok>, &'w [&'w [u8]]), ParseError> {
    if depth > MAX_PS_DEPTH {
        return Err(ParseError::TooDeep);
    }
    let mut toks = Vec::new();
    // Procedures read but not yet consumed by `if` / `ifelse`.
    let mut procs: Vec<Vec<PsTok>> = Vec::new();
    loop {
        let Some((&word, rest)) = words.split_first() else {
            return Err(ParseError::Unbalanced);
        };
        words = rest;
        match word {
            b"}" => {
                return if procs.is_empty() {
                    Ok((toks, words))
                } else {
                    Err(ParseError::MisplacedProcedure)
                };
            }
            b"{" => {
                if procs.len() == 2 {
                    return Err(ParseError::MisplacedProcedure);
                }
                let (inner, rest) = parse_block(words, depth + 1)?;
                procs.push(inner);
                words = rest;
            }
            b"if" => {
                let [then] = <[_; 1]>::try_from(std::mem::take(&mut procs))
                    .map_err(|_| ParseError::MisplacedProcedure)?;
                toks.push(PsTok::If(then));
            }
            b"ifelse" => {
                let [then, otherwise] = <[_; 2]>::try_from(std::mem::take(&mut procs))
                    .map_err(|_| ParseError::MisplacedProcedure)?;
                toks.push(PsTok::IfElse(then, otherwise));
            }
            _ => {
                if !procs.is_empty() {
                    return Err(ParseError::MisplacedProcedure);
                }
                toks.push(parse_word(word)?);
            }
        }
    }
}

/// A number or operator token.
fn parse_word(word: &[u8]) -> Result<PsTok, ParseError> {
    let text = || String::from_utf8_lossy(word).into_owned();
    if let Some(op) = parse_op(word) {
        return Ok(PsTok::Op(op));
    }
    match parse_number(word) {
        Some(n) if n.is_finite() => Ok(PsTok::Num(n)),
        Some(_) => Err(ParseError::NumberOutOfRange(text())),
        None => Err(ParseError::UnknownWord(text())),
    }
}

/// A PostScript number: an integer, a real with optional exponent, or a
/// radix integer `base#digits` (base 2–36, read as a 32-bit two's-complement
/// value).  `None` when `word` is not a number.
fn parse_number(word: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(word).ok()?;
    if let Some((base, digits)) = text.split_once('#') {
        let base = base.parse::<u32>().ok().filter(|b| (2..=36).contains(b))?;
        if digits.starts_with(['+', '-']) {
            return None;
        }
        let value = u32::from_str_radix(digits, base).ok()?;
        return Some(f64::from(value.cast_signed()));
    }
    // `f64::from_str` also accepts `inf` and `nan`, which are names in
    // PostScript, not numbers.
    if !text
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E'))
    {
        return None;
    }
    text.parse().ok()
}

fn pop(stack: &mut Vec<PsVal>) -> Option<PsVal> {
    stack.pop()
}

fn pop_num(stack: &mut Vec<PsVal>) -> Option<f64> {
    match stack.pop()? {
        PsVal::Num(n) => Some(n),
        PsVal::Bool(_) => None,
    }
}

fn pop_bool(stack: &mut Vec<PsVal>) -> Option<bool> {
    match stack.pop()? {
        PsVal::Bool(b) => Some(b),
        PsVal::Num(_) => None,
    }
}

/// Push a number result; a non-finite one (`1 0 div`, `-1 sqrt`, an `exp`
/// overflow) is the `undefinedresult` error of §7.10.5.
fn push_num(stack: &mut Vec<PsVal>, r: f64) -> Option<()> {
    r.is_finite().then(|| stack.push(PsVal::Num(r)))
}

/// Execute `toks` on `stack`.  `None` on any execution error.
fn exec(toks: &[PsTok], stack: &mut Vec<PsVal>) -> Option<()> {
    for tok in toks {
        match tok {
            PsTok::Num(n) => stack.push(PsVal::Num(*n)),
            PsTok::Op(op) => run_op(*op, stack)?,
            PsTok::If(then) => {
                if pop_bool(stack)? {
                    exec(then, stack)?;
                }
            }
            PsTok::IfElse(then, otherwise) => {
                let branch = if pop_bool(stack)? { then } else { otherwise };
                exec(branch, stack)?;
            }
        }
        if stack.len() > MAX_PS_STACK {
            return None;
        }
    }
    Some(())
}

/// Execute one operator.
fn run_op(op: PsOp, stack: &mut Vec<PsVal>) -> Option<()> {
    match op {
        PsOp::Abs => unary(stack, f64::abs),
        PsOp::Neg => unary(stack, |a| -a),
        PsOp::Ceiling => unary(stack, f64::ceil),
        PsOp::Floor => unary(stack, f64::floor),
        PsOp::Round => unary(stack, round_half_up),
        // The stack holds reals only, so `cvi` truncates like `truncate`
        // and `cvr` leaves its operand unchanged.
        PsOp::Truncate | PsOp::Cvi => unary(stack, f64::trunc),
        PsOp::Cvr => unary(stack, |a| a),
        PsOp::Sqrt => unary(stack, f64::sqrt),
        PsOp::Sin => unary(stack, |a| a.to_radians().sin()),
        PsOp::Cos => unary(stack, |a| a.to_radians().cos()),
        PsOp::Ln => unary(stack, f64::ln),
        PsOp::Log => unary(stack, f64::log10),
        PsOp::Add => binary(stack, |a, b| a + b),
        PsOp::Sub => binary(stack, |a, b| a - b),
        PsOp::Mul => binary(stack, |a, b| a * b),
        PsOp::Div => binary(stack, |a, b| a / b),
        PsOp::Exp => binary(stack, f64::powf),
        // `num den atan`: the angle in degrees, in [0, 360).
        PsOp::Atan => binary(stack, |num, den| {
            let deg = num.atan2(den).to_degrees();
            if deg < 0.0 { deg + 360.0 } else { deg }
        }),
        PsOp::Idiv => int_binary(stack, |a, b| (a / b).trunc()),
        PsOp::Mod => int_binary(stack, |a, b| a % b),
        PsOp::Eq => equality(stack, true),
        PsOp::Ne => equality(stack, false),
        PsOp::Gt => compare(stack, std::cmp::Ordering::is_gt),
        PsOp::Ge => compare(stack, std::cmp::Ordering::is_ge),
        PsOp::Lt => compare(stack, std::cmp::Ordering::is_lt),
        PsOp::Le => compare(stack, std::cmp::Ordering::is_le),
        PsOp::And => logic(stack, |a, b| a & b, |a, b| a & b),
        PsOp::Or => logic(stack, |a, b| a | b, |a, b| a | b),
        PsOp::Xor => logic(stack, |a, b| a ^ b, |a, b| a ^ b),
        PsOp::Not => not(stack),
        PsOp::Bitshift => bitshift(stack),
        PsOp::True => {
            stack.push(PsVal::Bool(true));
            Some(())
        }
        PsOp::False => {
            stack.push(PsVal::Bool(false));
            Some(())
        }
        PsOp::Pop => pop(stack).map(drop),
        PsOp::Dup => {
            let v = *stack.last()?;
            stack.push(v);
            Some(())
        }
        PsOp::Exch => {
            let len = stack.len();
            if len < 2 {
                return None;
            }
            stack.swap(len - 1, len - 2);
            Some(())
        }
        PsOp::Index => index(stack),
        PsOp::Copy => copy(stack),
        PsOp::Roll => roll(stack),
    }
}

/// PostScript `round`: the nearest integer, the greater of two equally near.
fn round_half_up(a: f64) -> f64 {
    let f = a.floor();
    if a - f >= 0.5 { f + 1.0 } else { f }
}

/// Pop one number, apply `f`, push the result.
fn unary(stack: &mut Vec<PsVal>, f: impl Fn(f64) -> f64) -> Option<()> {
    let a = pop_num(stack)?;
    push_num(stack, f(a))
}

/// Pop two numbers, apply `f`, push the result.
fn binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = pop_num(stack)?;
    let a = pop_num(stack)?;
    push_num(stack, f(a, b))
}

/// Pop two numbers, truncate both toward zero, apply `f`, push the result;
/// a zero divisor is an error.  Used by `idiv` and `mod`.
fn int_binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = pop_num(stack)?.trunc();
    let a = pop_num(stack)?.trunc();
    if b == 0.0 {
        return None;
    }
    push_num(stack, f(a, b))
}

/// `eq` (`want_eq`) or `ne`: operands of different types are unequal.
fn equality(stack: &mut Vec<PsVal>, want_eq: bool) -> Option<()> {
    let b = pop(stack)?;
    let a = pop(stack)?;
    #[expect(clippy::float_cmp, reason = "PostScript `eq` compares numbers exactly")]
    let eq = match (a, b) {
        (PsVal::Num(x), PsVal::Num(y)) => x == y,
        (PsVal::Bool(x), PsVal::Bool(y)) => x == y,
        _ => false,
    };
    stack.push(PsVal::Bool(eq == want_eq));
    Some(())
}

/// `gt`, `ge`, `lt`, `le` on two numbers.
fn compare(stack: &mut Vec<PsVal>, pred: impl Fn(std::cmp::Ordering) -> bool) -> Option<()> {
    let b = pop_num(stack)?;
    let a = pop_num(stack)?;
    stack.push(PsVal::Bool(pred(a.partial_cmp(&b)?)));
    Some(())
}

/// `and`, `or`, `xor`: logical on two booleans, bitwise on two integers.
fn logic(
    stack: &mut Vec<PsVal>,
    bf: impl Fn(bool, bool) -> bool,
    nf: impl Fn(i32, i32) -> i32,
) -> Option<()> {
    let b = pop(stack)?;
    let a = pop(stack)?;
    let r = match (a, b) {
        (PsVal::Bool(x), PsVal::Bool(y)) => PsVal::Bool(bf(x, y)),
        (PsVal::Num(x), PsVal::Num(y)) => PsVal::Num(f64::from(nf(to_i32(x)?, to_i32(y)?))),
        _ => return None,
    };
    stack.push(r);
    Some(())
}

/// `not`: logical complement of a boolean, bitwise complement of an integer.
fn not(stack: &mut Vec<PsVal>) -> Option<()> {
    let r = match pop(stack)? {
        PsVal::Bool(b) => PsVal::Bool(!b),
        PsVal::Num(n) => PsVal::Num(f64::from(!to_i32(n)?)),
    };
    stack.push(r);
    Some(())
}

/// `int shift bitshift`: shift the 32 bits of `int` left by `shift`, or
/// right by `-shift` when negative.  Bits shifted out are lost and the bits
/// shifted in are 0, so a shift of 32 or more places yields 0.
fn bitshift(stack: &mut Vec<PsVal>) -> Option<()> {
    let shift = to_i32(pop_num(stack)?)?;
    let bits = to_i32(pop_num(stack)?)?.cast_unsigned();
    let amount = shift.unsigned_abs();
    let r = if shift >= 0 {
        bits.checked_shl(amount)
    } else {
        bits.checked_shr(amount)
    };
    stack.push(PsVal::Num(f64::from(r.unwrap_or(0).cast_signed())));
    Some(())
}

/// `n index`: push a copy of the element `n` places below the top.
fn index(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = to_count(pop_num(stack)?)?;
    let idx = stack.len().checked_sub(n.checked_add(1)?)?;
    stack.push(stack[idx]);
    Some(())
}

/// `n copy`: duplicate the top `n` elements, preserving their order.
fn copy(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = to_count(pop_num(stack)?)?;
    let start = stack.len().checked_sub(n)?;
    stack.extend_from_within(start..);
    Some(())
}

/// `n j roll`: rotate the top `n` elements by `j` places toward the top
/// (away from it when `j` is negative).
fn roll(stack: &mut Vec<PsVal>) -> Option<()> {
    let j = to_i32(pop_num(stack)?)?;
    let n = to_count(pop_num(stack)?)?;
    if n == 0 {
        return Some(());
    }
    let start = stack.len().checked_sub(n)?;
    // n ≤ i32::MAX, so the remainder lies in [0, n).
    let shift = usize::try_from(i64::from(j).rem_euclid(i64::try_from(n).ok()?)).ok()?;
    stack[start..].rotate_right(shift);
    Some(())
}

/// A stack number as a 32-bit integer; `None` when it is not integral or
/// lies outside the `i32` range.
fn to_i32(n: f64) -> Option<i32> {
    if n.fract() != 0.0 || !(f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&n) {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n is integral and within the i32 range"
    )]
    Some(n as i32)
}

/// A stack number as a non-negative count for `index`, `copy` and `roll`.
fn to_count(n: f64) -> Option<usize> {
    usize::try_from(to_i32(n)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str, inputs: &[f64], n: usize) -> Option<Vec<f64>> {
        Program::parse(src.as_bytes())
            .expect("program parses")
            .run(inputs, n)
    }

    fn parse_err(src: &str) -> ParseError {
        Program::parse(src.as_bytes()).expect_err("program is rejected")
    }

    #[test]
    fn round_takes_the_greater_of_two_equally_near_integers() {
        assert_eq!(run("{ round }", &[2.5], 1), Some(vec![3.0]));
        assert_eq!(run("{ round }", &[-2.5], 1), Some(vec![-2.0]));
        assert_eq!(run("{ round }", &[-2.6], 1), Some(vec![-3.0]));
        assert_eq!(
            run("{ round }", &[0.499_999_999_999_999_94], 1),
            Some(vec![0.0])
        );
    }

    #[test]
    fn bitshift_fills_with_zero_bits() {
        assert_eq!(run("{ 8 2 bitshift }", &[], 1), Some(vec![32.0]));
        assert_eq!(
            run("{ -8 -1 bitshift }", &[], 1),
            Some(vec![2_147_483_644.0])
        );
        assert_eq!(run("{ 1 32 bitshift }", &[], 1), Some(vec![0.0]));
        assert_eq!(run("{ 1 -40 bitshift }", &[], 1), Some(vec![0.0]));
        assert_eq!(
            run("{ 1 31 bitshift }", &[], 1),
            Some(vec![f64::from(i32::MIN)])
        );
    }

    #[test]
    fn eq_of_different_types_is_false() {
        assert_eq!(
            run("{ 1 true eq { 1 } { 0 } ifelse }", &[], 1),
            Some(vec![0.0])
        );
        assert_eq!(
            run("{ 1 true ne { 1 } { 0 } ifelse }", &[], 1),
            Some(vec![1.0])
        );
    }

    #[test]
    fn radix_numbers_are_integers() {
        assert_eq!(run("{ 16#FF 2#101 }", &[], 2), Some(vec![255.0, 5.0]));
        assert_eq!(run("{ 16#FFFFFFFF }", &[], 1), Some(vec![-1.0]));
        assert!(matches!(parse_err("{ 37#1 }"), ParseError::UnknownWord(_)));
        assert!(matches!(parse_err("{ 16#-1 }"), ParseError::UnknownWord(_)));
    }

    #[test]
    fn names_that_rust_reads_as_numbers_are_rejected() {
        for word in ["inf", "nan", "infinity", "+inf", "NaN"] {
            assert!(
                matches!(
                    parse_err(&format!("{{ {word} }}")),
                    ParseError::UnknownWord(_)
                ),
                "{word}"
            );
        }
        assert!(matches!(
            parse_err("{ 1e400 }"),
            ParseError::NumberOutOfRange(_)
        ));
    }

    #[test]
    fn nul_separates_words_and_comments_need_not_be_utf8() {
        assert_eq!(
            run("{\0 2\0mul % caf\u{e9}\n}", &[0.25], 1),
            Some(vec![0.5])
        );
        let latin1_comment = b"{ 2 mul % caf\xe9\n }";
        let program = Program::parse(latin1_comment).expect("parses");
        assert_eq!(program.run(&[0.25], 1), Some(vec![0.5]));
    }

    #[test]
    fn procedures_must_be_conditional_operands() {
        assert_eq!(parse_err("{ { 1 } }"), ParseError::MisplacedProcedure);
        assert_eq!(parse_err("{ { 1 } 2 if }"), ParseError::MisplacedProcedure);
        assert_eq!(
            parse_err("{ { 1 } { 2 } if }"),
            ParseError::MisplacedProcedure
        );
        assert_eq!(
            parse_err("{ { 1 } ifelse }"),
            ParseError::MisplacedProcedure
        );
        assert_eq!(
            parse_err("{ {1} {2} {3} ifelse }"),
            ParseError::MisplacedProcedure
        );
        assert_eq!(parse_err("{ true if }"), ParseError::MisplacedProcedure);
    }

    #[test]
    fn program_structure_errors() {
        assert_eq!(parse_err(""), ParseError::NoProcedure);
        assert_eq!(parse_err("2 mul"), ParseError::NoProcedure);
        assert_eq!(parse_err("{ 2 mul"), ParseError::Unbalanced);
        assert_eq!(parse_err("{ 2 mul } 3"), ParseError::TrailingTokens);
        assert!(matches!(
            parse_err("{ frobnicate }"),
            ParseError::UnknownWord(w) if w == "frobnicate"
        ));
    }

    #[test]
    fn an_over_long_program_is_rejected_before_it_runs() {
        let src = format!("{{{}}}", " 1 pop".repeat(MAX_PS_TOKENS / 2));
        assert!(matches!(parse_err(&src), ParseError::TooLong(_)));
    }

    #[test]
    fn nested_conditionals_run_the_selected_branch() {
        let src = "{ dup 0.5 lt { 0.25 lt { 1 } { 2 } ifelse } { pop 3 } ifelse }";
        assert_eq!(run(src, &[0.1], 1), Some(vec![1.0]));
        assert_eq!(run(src, &[0.3], 1), Some(vec![2.0]));
        assert_eq!(run(src, &[0.9], 1), Some(vec![3.0]));
    }

    #[test]
    fn boolean_results_and_underflow_fail() {
        assert_eq!(run("{ true }", &[], 1), None);
        assert_eq!(run("{ pop }", &[], 1), None);
        assert_eq!(run("{ 1 index }", &[0.5], 1), None);
        assert_eq!(run("{ 2 copy }", &[0.5], 1), None);
    }
}
