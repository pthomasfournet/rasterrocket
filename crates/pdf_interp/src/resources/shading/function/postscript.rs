//! Type 4 (PostScript calculator) functions, PDF §7.10.5.
//!
//! A program is parsed once into a token tree ([`Program::parse`]) and run
//! any number of times ([`Program::run`]).  The interpreter implements the
//! bounded operator subset of §7.10.5 — arithmetic, boolean/relational,
//! stack, and `if`/`ifelse` conditionals.  There are no loops, defs,
//! strings, or arrays beyond the operand stack; a malformed program, stack
//! underflow or non-finite result yields `None`.

/// A parsed Type 4 program: the token tree of its outermost `{ }` block.
#[derive(Debug)]
pub(super) struct Program(Vec<PsTok>);

impl Program {
    /// Parse the program text `src`; `None` when it is not valid UTF-8 or
    /// not a single well-formed `{ }` block.
    pub(super) fn parse(src: &[u8]) -> Option<Self> {
        parse_ps_program(std::str::from_utf8(src).ok()?).map(Self)
    }

    /// Run the program on `inputs` (already clipped to `Domain`) and return
    /// the top `n` stack values, deepest first.  `None` on any execution
    /// error, on fewer than `n` results, or on a boolean or non-finite
    /// result.
    pub(super) fn run(&self, inputs: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut stack: Vec<PsVal> = inputs.iter().map(|&t| PsVal::Num(t)).collect();
        let mut steps = MAX_PS_STEPS;
        exec_ps(&self.0, &mut stack, &mut steps, 0)?;

        if stack.len() < n {
            return None;
        }
        let out: Vec<f64> = stack[stack.len() - n..]
            .iter()
            .map(|v| match v {
                PsVal::Num(x) => Some(*x),
                PsVal::Bool(_) => None,
            })
            .collect::<Option<Vec<f64>>>()?;
        // A literal such as `1e400` parses to infinity; a non-finite colour
        // component is rejected rather than passed to the rasterizer.
        if out.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(out)
    }
}

/// Cap on total operators executed by a single Type-4 invocation.  PDF
/// §7.10.5 forbids loops, so a well-formed program executes a fixed bounded
/// number of operators; this guards against pathological hand-crafted input
/// that nests `if`/`ifelse` to blow the budget.
const MAX_PS_STEPS: u32 = 100_000;

/// Cap on `{ }` procedure nesting.  Both the parser and the executor recurse
/// per nested block, so unbounded depth can overflow the native stack on a
/// hand-crafted `{{{{…}}}}` program (the step budget does not bound parse
/// recursion, which runs before any step is charged, nor the executor's
/// native frame depth on conditional branches).  Real tint transforms nest a
/// handful of levels at most; 64 is far beyond any legitimate program.
pub(super) const MAX_PS_DEPTH: usize = 64;

/// Cap on the operand stack.  A single `copy`/`dup`/`index` can request a
/// huge push while charging only one step against [`MAX_PS_STEPS`], so the
/// step budget alone does not bound memory; `{ 0 0 2000000000 copy }` would
/// otherwise allocate billions of slots.  §7.10.5 calculator programs operate
/// on a tiny working set (tint in, a few colorants out); 100k is generous.
pub(super) const MAX_PS_STACK: usize = 100_000;

/// A value on the Type-4 operand stack: a real number or a boolean.  PDF
/// §7.10.5 booleans arise from relational/boolean operators and are consumed
/// only by `if`/`ifelse`.
#[derive(Clone, Copy, Debug)]
enum PsVal {
    Num(f64),
    Bool(bool),
}

/// A parsed Type-4 token: a literal, an operator, or a `{ ... }` procedure
/// block (only ever an operand to `if`/`ifelse`).
#[derive(Clone, Debug)]
enum PsTok {
    Num(f64),
    Op(PsOp),
    Proc(Vec<Self>),
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
    If,
    Ifelse,
}

fn parse_ps_op(word: &str) -> Option<PsOp> {
    Some(match word {
        "abs" => PsOp::Abs,
        "add" => PsOp::Add,
        "atan" => PsOp::Atan,
        "ceiling" => PsOp::Ceiling,
        "cos" => PsOp::Cos,
        "cvi" => PsOp::Cvi,
        "cvr" => PsOp::Cvr,
        "div" => PsOp::Div,
        "exp" => PsOp::Exp,
        "floor" => PsOp::Floor,
        "idiv" => PsOp::Idiv,
        "ln" => PsOp::Ln,
        "log" => PsOp::Log,
        "mod" => PsOp::Mod,
        "mul" => PsOp::Mul,
        "neg" => PsOp::Neg,
        "round" => PsOp::Round,
        "sin" => PsOp::Sin,
        "sqrt" => PsOp::Sqrt,
        "sub" => PsOp::Sub,
        "truncate" => PsOp::Truncate,
        "eq" => PsOp::Eq,
        "ne" => PsOp::Ne,
        "gt" => PsOp::Gt,
        "ge" => PsOp::Ge,
        "lt" => PsOp::Lt,
        "le" => PsOp::Le,
        "and" => PsOp::And,
        "or" => PsOp::Or,
        "xor" => PsOp::Xor,
        "not" => PsOp::Not,
        "bitshift" => PsOp::Bitshift,
        "true" => PsOp::True,
        "false" => PsOp::False,
        "copy" => PsOp::Copy,
        "dup" => PsOp::Dup,
        "exch" => PsOp::Exch,
        "index" => PsOp::Index,
        "pop" => PsOp::Pop,
        "roll" => PsOp::Roll,
        "if" => PsOp::If,
        "ifelse" => PsOp::Ifelse,
        _ => return None,
    })
}

/// Split the decoded program text into whitespace/brace-delimited lexemes.
/// `{` and `}` are returned as standalone tokens; everything else is a word.
///
/// PDF §7.10.5 calculator functions are PostScript, which treats `%` as a
/// line comment to end-of-line.  Comments are stripped here so an otherwise
/// well-formed commented program is not rejected wholesale (a rejected tint
/// transform silently falls back to the gray heuristic, blanking
/// spot-colour pages).
fn ps_lex(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_comment = false;
    for ch in src.chars() {
        if in_comment {
            // A PostScript comment runs to the next end-of-line.
            if ch == '\n' || ch == '\r' {
                in_comment = false;
            }
            continue;
        }
        match ch {
            '%' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                in_comment = true;
            }
            '{' | '}' => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                out.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Parse a lexeme slice starting just after a `{` into a token list, stopping
/// at the matching `}`.  Returns the tokens and the index just past the `}`.
///
/// `depth` is the current `{ }` nesting level; recursion is refused past
/// [`MAX_PS_DEPTH`] so a `{{{{…}}}}` bomb cannot overflow the native stack
/// during parsing (it returns `None`, which the caller logs and falls back
/// on, rather than aborting the process).
fn parse_ps_block(lexemes: &[String], mut i: usize, depth: usize) -> Option<(Vec<PsTok>, usize)> {
    if depth > MAX_PS_DEPTH {
        return None;
    }
    let mut toks = Vec::new();
    while i < lexemes.len() {
        let lx = lexemes[i].as_str();
        match lx {
            "}" => return Some((toks, i + 1)),
            "{" => {
                let (inner, next) = parse_ps_block(lexemes, i + 1, depth + 1)?;
                toks.push(PsTok::Proc(inner));
                i = next;
            }
            _ => {
                if let Ok(num) = lx.parse::<f64>() {
                    toks.push(PsTok::Num(num));
                } else {
                    toks.push(PsTok::Op(parse_ps_op(lx)?));
                }
                i += 1;
            }
        }
    }
    // Reached end of input without a closing brace.
    None
}

/// Parse the full Type-4 program (a single top-level `{ ... }` block).
fn parse_ps_program(src: &str) -> Option<Vec<PsTok>> {
    let lexemes = ps_lex(src);
    let first = lexemes.first()?;
    if first != "{" {
        return None;
    }
    let (toks, end) = parse_ps_block(&lexemes, 1, 0)?;
    // The top-level block must be the entire program; trailing lexemes after
    // the matching `}` mean the input is malformed.
    if end != lexemes.len() {
        return None;
    }
    Some(toks)
}

fn ps_pop(stack: &mut Vec<PsVal>) -> Option<PsVal> {
    stack.pop()
}

fn ps_pop_num(stack: &mut Vec<PsVal>) -> Option<f64> {
    match stack.pop()? {
        PsVal::Num(n) => Some(n),
        PsVal::Bool(_) => None,
    }
}

fn ps_pop_bool(stack: &mut Vec<PsVal>) -> Option<bool> {
    match stack.pop()? {
        PsVal::Bool(b) => Some(b),
        PsVal::Num(_) => None,
    }
}

/// Execute a single operator.  `If`/`Ifelse` never reach here — [`exec_ps`]
/// resolves them by consuming the preceding `Proc` token(s) as branches.
fn run_ps_op(op: PsOp, stack: &mut Vec<PsVal>) -> Option<()> {
    match op {
        PsOp::Abs => unary(stack, f64::abs),
        PsOp::Neg => unary(stack, |a| -a),
        PsOp::Ceiling => unary(stack, f64::ceil),
        PsOp::Floor => unary(stack, f64::floor),
        PsOp::Round => unary(stack, f64::round),
        // `truncate` and `cvi` are both truncate-toward-zero; `cvr` is a
        // no-op since the stack is already real-valued.
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
        PsOp::Atan => binary(stack, |num, den| {
            let deg = num.atan2(den).to_degrees();
            if deg < 0.0 { deg + 360.0 } else { deg }
        }),
        PsOp::Idiv => int_binary(stack, |a, b| (a / b).trunc()),
        PsOp::Mod => int_binary(stack, |a, b| a % b),
        PsOp::Eq => rel_eq(stack, true),
        PsOp::Ne => rel_eq(stack, false),
        PsOp::Gt => rel_ord(stack, |o| o == std::cmp::Ordering::Greater),
        PsOp::Ge => rel_ord(stack, |o| o != std::cmp::Ordering::Less),
        PsOp::Lt => rel_ord(stack, |o| o == std::cmp::Ordering::Less),
        PsOp::Le => rel_ord(stack, |o| o != std::cmp::Ordering::Greater),
        PsOp::And => logic(stack, |a, b| a & b, |a, b| a & b),
        PsOp::Or => logic(stack, |a, b| a | b, |a, b| a | b),
        PsOp::Xor => logic(stack, |a, b| a ^ b, |a, b| a ^ b),
        PsOp::Not => ps_not(stack),
        PsOp::Bitshift => ps_bitshift(stack),
        PsOp::True => {
            stack.push(PsVal::Bool(true));
            Some(())
        }
        PsOp::False => {
            stack.push(PsVal::Bool(false));
            Some(())
        }
        PsOp::Pop => ps_pop(stack).map(|_| ()),
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
        PsOp::Index => ps_index(stack),
        PsOp::Copy => ps_copy(stack),
        PsOp::Roll => ps_roll(stack),
        // If/Ifelse are resolved in exec_ps; reaching here is malformed.
        PsOp::If | PsOp::Ifelse => None,
    }
}

/// Pop two numbers, truncate both toward zero, apply `f`, push the result.
/// Used by `idiv` and `mod`, which operate on integer values.
fn int_binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = ps_pop_num(stack)?.trunc();
    let a = ps_pop_num(stack)?.trunc();
    if b == 0.0 {
        return None;
    }
    let r = f(a, b);
    // §7.10.5 leaves a non-finite result undefined; refuse it rather than
    // let an Inf/NaN flow into the tint→colour conversion (silent-wrong).
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

/// `not`: logical complement for booleans, bitwise complement for integers.
fn ps_not(stack: &mut Vec<PsVal>) -> Option<()> {
    match ps_pop(stack)? {
        PsVal::Bool(b) => stack.push(PsVal::Bool(!b)),
        PsVal::Num(n) => stack.push(PsVal::Num(f64::from(!ps_to_i32(n)?))),
    }
    Some(())
}

/// `bitshift`: arithmetic shift of an integer by a signed shift count
/// (positive = left, negative = right).
fn ps_bitshift(stack: &mut Vec<PsVal>) -> Option<()> {
    let shift = ps_to_i32(ps_pop_num(stack)?)?;
    let val = ps_to_i32(ps_pop_num(stack)?)?;
    let r = if shift >= 0 {
        val.wrapping_shl(shift.unsigned_abs())
    } else {
        val.wrapping_shr(shift.unsigned_abs())
    };
    stack.push(PsVal::Num(f64::from(r)));
    Some(())
}

/// `n index`: push a copy of the element `n` positions below the top.
fn ps_index(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    let idx = stack.len().checked_sub(1 + n)?;
    stack.push(stack[idx]);
    Some(())
}

/// `n copy`: duplicate the top `n` elements, preserving order.
fn ps_copy(stack: &mut Vec<PsVal>) -> Option<()> {
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    let start = stack.len().checked_sub(n)?;
    for i in 0..n {
        stack.push(stack[start + i]);
    }
    Some(())
}

/// `n j roll`: circularly shift the top `n` elements by `j` (positive `j`
/// rotates toward the top of the stack).
fn ps_roll(stack: &mut Vec<PsVal>) -> Option<()> {
    let j = ps_to_i32(ps_pop_num(stack)?)?;
    let n = ps_to_usize(ps_pop_num(stack)?)?;
    if n == 0 {
        return Some(());
    }
    let start = stack.len().checked_sub(n)?;
    let slice = &mut stack[start..];
    // rem_euclid keeps the shift in [0, n) for negative j as well.  i64
    // avoids any 32-bit wrap; n came from a stack number bounded by ps_to_i32
    // so it fits i64.
    let n_i64 = i64::try_from(n).ok()?;
    let shift = usize::try_from(i64::from(j).rem_euclid(n_i64)).ok()?;
    slice.rotate_right(shift);
    Some(())
}

/// Pop one number, apply `f`, push the result.  A non-finite result (e.g.
/// `sqrt` of a negative, `ln`/`log` of a non-positive) is undefined per
/// §7.10.5; refuse it loud-graceful so no Inf/NaN reaches colour conversion.
fn unary(stack: &mut Vec<PsVal>, f: impl Fn(f64) -> f64) -> Option<()> {
    let a = ps_pop_num(stack)?;
    let r = f(a);
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

/// Pop two numbers, apply `f`, push the result.  A non-finite result (e.g.
/// `1 0 div` → Inf, `exp` overflow) is undefined per §7.10.5; refuse it
/// loud-graceful so no Inf/NaN reaches colour conversion.
fn binary(stack: &mut Vec<PsVal>, f: impl Fn(f64, f64) -> f64) -> Option<()> {
    let b = ps_pop_num(stack)?;
    let a = ps_pop_num(stack)?;
    let r = f(a, b);
    if !r.is_finite() {
        return None;
    }
    stack.push(PsVal::Num(r));
    Some(())
}

fn rel_eq(stack: &mut Vec<PsVal>, want_eq: bool) -> Option<()> {
    let b = ps_pop(stack)?;
    let a = ps_pop(stack)?;
    let eq = match (a, b) {
        (PsVal::Num(x), PsVal::Num(y)) => x == y,
        (PsVal::Bool(x), PsVal::Bool(y)) => x == y,
        _ => return None,
    };
    stack.push(PsVal::Bool(eq == want_eq));
    Some(())
}

fn rel_ord(stack: &mut Vec<PsVal>, pred: impl Fn(std::cmp::Ordering) -> bool) -> Option<()> {
    let b = ps_pop_num(stack)?;
    let a = ps_pop_num(stack)?;
    let ord = a.partial_cmp(&b)?;
    stack.push(PsVal::Bool(pred(ord)));
    Some(())
}

fn logic(
    stack: &mut Vec<PsVal>,
    bf: impl Fn(bool, bool) -> bool,
    nf: impl Fn(i32, i32) -> i32,
) -> Option<()> {
    let b = ps_pop(stack)?;
    let a = ps_pop(stack)?;
    match (a, b) {
        (PsVal::Bool(x), PsVal::Bool(y)) => stack.push(PsVal::Bool(bf(x, y))),
        (PsVal::Num(x), PsVal::Num(y)) => {
            stack.push(PsVal::Num(f64::from(nf(ps_to_i32(x)?, ps_to_i32(y)?))));
        }
        _ => return None,
    }
    Some(())
}

/// Convert a stack number to an i32 for bitwise ops; reject non-integral or
/// out-of-range values (PostScript integers are 32-bit).
fn ps_to_i32(n: f64) -> Option<i32> {
    if !n.is_finite() || n.fract() != 0.0 || n < f64::from(i32::MIN) || n > f64::from(i32::MAX) {
        return None;
    }
    // Bounds + integrality checked above.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value verified finite, integral, and within i32 range above"
    )]
    Some(n as i32)
}

/// Convert a stack number to a non-negative count for `index`/`copy`/`roll`.
/// Rejects negative, non-integral, or out-of-i32-range values.
fn ps_to_usize(n: f64) -> Option<usize> {
    let i = ps_to_i32(n)?;
    if i < 0 {
        return None;
    }
    usize::try_from(i).ok()
}

/// Walk a token list, executing it while resolving `if`/`ifelse`.  The parser
/// places procedure blocks as `Proc` tokens immediately before the `if` /
/// `ifelse` operator; this pass consumes them as the conditional branches.
///
/// `depth` mirrors the `{ }` nesting and bounds this function's own native
/// recursion: a deeply nested conditional chain (each `if` consuming only one
/// step) could otherwise recurse far enough to overflow the native stack
/// before [`MAX_PS_STEPS`] is reached.  The operand stack is also capped at
/// [`MAX_PS_STACK`] because a single `copy`/`dup`/`index` can push many
/// values while charging only one step.
fn exec_ps(toks: &[PsTok], stack: &mut Vec<PsVal>, steps: &mut u32, depth: usize) -> Option<()> {
    if depth > MAX_PS_DEPTH {
        return None;
    }
    let mut i = 0;
    while i < toks.len() {
        if *steps == 0 {
            return None;
        }
        match &toks[i] {
            PsTok::Num(n) => {
                *steps -= 1;
                stack.push(PsVal::Num(*n));
                i += 1;
            }
            PsTok::Op(PsOp::If) => {
                // Preceding token must be the procedure block.
                let proc = match toks.get(i.wrapping_sub(1)) {
                    Some(PsTok::Proc(p)) if i > 0 => p,
                    _ => return None,
                };
                *steps -= 1;
                let cond = ps_pop_bool(stack)?;
                if cond {
                    exec_ps(proc, stack, steps, depth + 1)?;
                }
                i += 1;
            }
            PsTok::Op(PsOp::Ifelse) => {
                // Preceding two tokens must be the two procedure blocks.
                let (p1, p2) = match (toks.get(i.wrapping_sub(2)), toks.get(i.wrapping_sub(1))) {
                    (Some(PsTok::Proc(a)), Some(PsTok::Proc(b))) if i >= 2 => (a, b),
                    _ => return None,
                };
                *steps -= 1;
                let cond = ps_pop_bool(stack)?;
                if cond {
                    exec_ps(p1, stack, steps, depth + 1)?;
                } else {
                    exec_ps(p2, stack, steps, depth + 1)?;
                }
                i += 1;
            }
            PsTok::Proc(_) => {
                // A procedure block is only valid as an operand to a
                // following if/ifelse; skip it here (it is consumed by the
                // If/Ifelse arms by looking back).  But verify it IS followed
                // by if/ifelse, otherwise the program is malformed.
                let followed_by_cond =
                    matches!(toks.get(i + 1), Some(PsTok::Op(PsOp::If | PsOp::Ifelse)))
                        || matches!(
                            (toks.get(i + 1), toks.get(i + 2)),
                            (Some(PsTok::Proc(_)), Some(PsTok::Op(PsOp::Ifelse)))
                        );
                if !followed_by_cond {
                    return None;
                }
                i += 1;
            }
            PsTok::Op(op) => {
                *steps -= 1;
                run_ps_op(*op, stack)?;
                i += 1;
            }
        }
        // Bound operand-stack growth.  `copy`/`dup`/`index` and bare literals
        // can grow the stack while charging at most one step, so the step
        // budget alone is not a memory bound; refuse loud-graceful past
        // MAX_PS_STACK rather than let a `copy` bomb exhaust memory.
        if stack.len() > MAX_PS_STACK {
            return None;
        }
    }
    Some(())
}
