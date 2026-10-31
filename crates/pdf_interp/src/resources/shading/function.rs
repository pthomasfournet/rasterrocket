//! PDF functions (§7.10): Type 0 (Sampled), 2 (Exponential), 3 (Stitching)
//! and 4 (PostScript calculator).
//!
//! A function object is compiled once into a [`Function`] — sample streams
//! decoded, PostScript programs parsed, every dictionary entry read — and
//! evaluated any number of times by arithmetic alone.  [`Function::load`]
//! memoises the compiled form of an indirect function object for the
//! lifetime of the document.
//!
//! # Limitations
//!
//! Type 0 (Sampled) interpolates multilinearly at every `BitsPerSample`
//! the specification allows; `Order 3` (cubic spline) is evaluated as
//! `Order 1`.
//!
//! Type 4 (PostScript calculator) implements the bounded operator subset of
//! PDF §7.10.5 (see [`postscript`]); a malformed program or stack underflow
//! evaluates to `None` so the caller falls back loudly.

use std::collections::HashMap;
use std::sync::Arc;

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;
use crate::resources::obj_to_f64;

mod postscript;
mod sampled;

use postscript::Program;
use sampled::Sampled;

/// Maximum nesting depth of compiled sub-functions (guards against crafted
/// PDFs with deeply or circularly nested Type 3 functions).
const MAX_FN_DEPTH: u8 = 10;

// ── Compiled functions ────────────────────────────────────────────────────────

/// A compiled PDF function.
///
/// The output count `n` is supplied at evaluation time by the consumer (the
/// colour space a shading or tint transform feeds), not read from `Range`.
#[derive(Debug)]
pub(crate) struct Function {
    body: Body,
    /// `Range` as flat `[lo0 hi0 lo1 hi1 …]` numbers; empty when absent.
    range: Vec<f64>,
}

#[derive(Debug)]
enum Body {
    /// An array of single-output functions, one per output (a shading's
    /// `Function` entry, §8.7.4.5.1).
    Array(Vec<Arc<Function>>),
    /// Type 2: `C0 + x^N × (C1 − C0)`.
    Exponential {
        domain: (f64, f64),
        c0: Option<Vec<f64>>,
        c1: Option<Vec<f64>>,
        exponent: f64,
    },
    /// Type 3: `Domain` split at `Bounds` into sub-domains, each mapped
    /// through its `Encode` pair onto a sub-function.
    Stitching {
        domain: (f64, f64),
        functions: Vec<Arc<Function>>,
        bounds: Vec<f64>,
        encode: Vec<f64>,
    },
    /// Type 0.
    Sampled(Sampled),
    /// Type 4; `None` when the program failed to parse.
    PostScript {
        domain: Vec<f64>,
        program: Option<Program>,
    },
    /// A function evaluated as its `C0` entry: an unknown `FunctionType`,
    /// or a sub-function past [`MAX_FN_DEPTH`].
    Constant(Option<Vec<f64>>),
    /// An object that is not a function; every evaluation yields `None`.
    Invalid,
}

impl Function {
    const INVALID: Self = Self {
        body: Body::Invalid,
        range: Vec::new(),
    };

    /// Compile the function object `obj`.  An indirect function is compiled
    /// once per document and shared; the elements of a direct array of
    /// functions are loaded the same way.
    #[must_use]
    pub(crate) fn load(doc: &Document, obj: &Object) -> Arc<Self> {
        match obj {
            Object::Reference(id) => doc
                .derived(*id, || Some(Compiler::new(doc).compile(obj, 0)))
                .unwrap_or_else(|| Arc::new(Self::INVALID)),
            Object::Array(elems) => Arc::new(Self {
                body: Body::Array(elems.iter().map(|e| Self::load(doc, e)).collect()),
                range: Vec::new(),
            }),
            direct => Arc::new(Compiler::new(doc).compile(direct, 0)),
        }
    }

    /// Evaluate at the single input `t`, returning `n` outputs clipped to
    /// `Range`.
    #[must_use]
    pub(crate) fn eval(&self, t: f64, n: usize) -> Option<Vec<f64>> {
        let out = match &self.body {
            Body::Array(fns) => {
                if fns.len() != n {
                    log::debug!(
                        "function: Function array has {} entries, {n} outputs needed",
                        fns.len()
                    );
                    return None;
                }
                return fns.iter().map(|f| f.eval(t, 1)?.first().copied()).collect();
            }
            Body::Exponential {
                domain,
                c0,
                c1,
                exponent,
            } => eval_exponential(*domain, c0.as_deref(), c1.as_deref(), *exponent, t, n),
            Body::Stitching {
                domain,
                functions,
                bounds,
                encode,
            } => {
                let (f, x) = stitch(*domain, functions, bounds, encode, t);
                f.eval(x, n)?
            }
            Body::Sampled(s) => s.eval_nd(&[t], n)?,
            Body::PostScript { domain, program } => {
                let out = program
                    .as_ref()
                    .and_then(|p| p.run(&clip_to_domains(domain, &[t]), n));
                if out.is_none() {
                    log::debug!("function: Type 4 program failed to evaluate");
                }
                out?
            }
            Body::Constant(c0) => pad(c0.as_deref()?, n),
            Body::Invalid => return None,
        };
        Some(self.clip_to_range(out))
    }

    /// Evaluate at the `m`-dimensional input point `inputs`, returning `n`
    /// outputs clipped to `Range` (PDF §7.10).
    ///
    /// A one-element input is exactly [`Self::eval`].  With several inputs
    /// only Type 0 (multilinear interpolation over the sample grid) and
    /// Type 4 (inputs pushed in order) apply; Types 2 and 3 take one input by
    /// definition (§7.10.3, §7.10.4) and yield `None`.  An array of
    /// functions evaluates element `i` for output `i`.
    #[must_use]
    pub(crate) fn eval_n(&self, inputs: &[f64], n: usize) -> Option<Vec<f64>> {
        if let [t] = inputs {
            return self.eval(*t, n);
        }
        let out = match &self.body {
            Body::Array(fns) => {
                if fns.len() != n {
                    log::debug!(
                        "function: Function array has {} entries, {n} outputs needed",
                        fns.len()
                    );
                    return None;
                }
                return fns
                    .iter()
                    .map(|f| match f.body {
                        Body::Array(_) => None,
                        _ => f.eval_n(inputs, 1)?.first().copied(),
                    })
                    .collect();
            }
            Body::Sampled(s) => s.eval_nd(inputs, n)?,
            Body::PostScript { domain, program } => {
                program.as_ref()?.run(&clip_to_domains(domain, inputs), n)?
            }
            _ => {
                log::debug!(
                    "function: function takes one input, {} supplied",
                    inputs.len()
                );
                return None;
            }
        };
        Some(self.clip_to_range(out))
    }

    /// Clip output `ch` to `Range` pair `ch` (PDF §7.10.2).  A missing pair
    /// or an empty interval leaves the output unconstrained.
    fn clip_to_range(&self, mut out: Vec<f64>) -> Vec<f64> {
        for (v, &[lo, hi]) in out.iter_mut().zip(self.range.as_chunks::<2>().0) {
            if hi > lo {
                *v = v.clamp(lo, hi);
            }
        }
        out
    }
}

/// One compilation of a top-level function object.
///
/// Sub-functions named by reference are compiled once per compilation and
/// shared, so an object that several `Functions` entries name — directly or
/// through further sub-functions — costs one compilation rather than one
/// per path to it.  The document cache is not consulted below the top
/// level: a function that refers to itself would re-enter its own cache
/// cell.  A cycle is instead cut at [`MAX_FN_DEPTH`], and every later
/// reference to an object on the cycle reuses the result first recorded
/// for it.
struct Compiler<'d> {
    doc: &'d Document,
    compiled: HashMap<ObjectId, Arc<Function>>,
}

impl<'d> Compiler<'d> {
    fn new(doc: &'d Document) -> Self {
        Self {
            doc,
            compiled: HashMap::new(),
        }
    }

    /// The sub-function `obj` at nesting `depth`, shared when `obj` is a
    /// reference compiled before.
    fn sub(&mut self, obj: &Object, depth: u8) -> Arc<Function> {
        let Object::Reference(id) = obj else {
            return Arc::new(self.compile(obj, depth));
        };
        if let Some(f) = self.compiled.get(id) {
            return Arc::clone(f);
        }
        let f = Arc::new(self.compile(obj, depth));
        Arc::clone(self.compiled.entry(*id).or_insert(f))
    }

    /// Compile `obj` at sub-function nesting `depth`.
    fn compile(&mut self, obj: &Object, depth: u8) -> Function {
        let doc = self.doc;
        let referent;
        let target = match obj {
            Object::Reference(id) => {
                let Ok(o) = doc.get_object(*id) else {
                    return Function::INVALID;
                };
                referent = o;
                referent.as_ref()
            }
            direct => direct,
        };
        let (dict, stream) = match target {
            Object::Array(elems) => {
                if depth >= MAX_FN_DEPTH {
                    return Function::INVALID;
                }
                return Function {
                    body: Body::Array(elems.iter().map(|e| self.sub(e, depth + 1)).collect()),
                    range: Vec::new(),
                };
            }
            Object::Dictionary(d) => (d, None),
            Object::Stream(s) => (&s.dict, Some(s)),
            _ => return Function::INVALID,
        };
        let range = numbers(dict, b"Range");
        if depth >= MAX_FN_DEPTH {
            log::warn!(
                "function: nesting depth {depth} exceeds limit {MAX_FN_DEPTH} — \
                 evaluating as C0"
            );
            return Function {
                body: Body::Constant(read_color(dict, b"C0")),
                range,
            };
        }
        let stream_bytes = || {
            stream?
                .decompressed_content()
                .map_err(|e| log::debug!("function: stream decode failed: {e}"))
                .ok()
        };
        let body = match dict.get_i64(b"FunctionType") {
            None => Body::Invalid,
            Some(0) => Body::Sampled(Sampled::compile(dict, stream_bytes())),
            Some(2) => Body::Exponential {
                domain: read_fn_domain(dict),
                c0: read_color(dict, b"C0"),
                c1: read_color(dict, b"C1"),
                exponent: dict.get(b"N").and_then(obj_to_f64).unwrap_or(1.0),
            },
            Some(3) => match dict.get(b"Functions").and_then(Object::as_array) {
                Some(fns) if !fns.is_empty() => {
                    let domain = read_fn_domain(dict);
                    let bounds = numbers(dict, b"Bounds");
                    // §7.10.4: k sub-functions take k − 1 Bounds.  Otherwise
                    // the first sub-function covers the Domain unchanged.
                    let (fns, bounds, encode) = if bounds.len() + 1 == fns.len() {
                        (fns, bounds, numbers(dict, b"Encode"))
                    } else {
                        log::warn!(
                            "function: Type 3 function has {} sub-functions but {} Bounds \
                             values — evaluating the first sub-function",
                            fns.len(),
                            bounds.len(),
                        );
                        (&fns[..1], Vec::new(), vec![domain.0, domain.1])
                    };
                    Body::Stitching {
                        domain,
                        functions: fns.iter().map(|f| self.sub(f, depth + 1)).collect(),
                        bounds,
                        encode,
                    }
                }
                _ => Body::Invalid,
            },
            Some(4) => Body::PostScript {
                domain: numbers(dict, b"Domain"),
                program: stream_bytes().and_then(|src| {
                    Program::parse(&src)
                        .map_err(|e| log::warn!("function: Type 4 program rejected: {e}"))
                        .ok()
                }),
            },
            Some(other) => {
                log::debug!("function: FunctionType {other} not implemented — evaluating as C0");
                Body::Constant(read_color(dict, b"C0"))
            }
        };
        Function { body, range }
    }
}

/// Type 2 (Exponential, PDF §7.10.3): `C0 + x^N × (C1 − C0)` with `x` the
/// input clipped to `Domain`, `C0` default 0 and `C1` default 1 per output.
/// A power with no finite real value (a negative `x` with a non-integer
/// `N`, or `x = 0` with a negative `N`) evaluates to `C0`.
fn eval_exponential(
    (d0, d1): (f64, f64),
    c0: Option<&[f64]>,
    c1: Option<&[f64]>,
    exponent: f64,
    t: f64,
    n: usize,
) -> Vec<f64> {
    let c0 = c0.map_or_else(|| vec![0.0; n], |c| pad(c, n));
    let c1 = c1.map_or_else(|| vec![1.0; n], |c| pad(c, n));
    let x = if d1 > d0 { t.clamp(d0, d1) } else { d0 };
    let weight = x.powf(exponent);
    if !weight.is_finite() {
        return c0;
    }
    c0.iter()
        .zip(&c1)
        .map(|(&a, &b)| weight.mul_add(b - a, a))
        .collect()
}

/// Type 3 (Stitching): select the sub-function whose sub-domain holds `t`
/// (clipped to `Domain`) and map `t` through that sub-domain's `Encode`
/// pair.  `functions` is non-empty and `bounds` holds one value fewer, as
/// compilation ensures.
fn stitch<'f>(
    (d0, d1): (f64, f64),
    functions: &'f [Arc<Function>],
    bounds: &[f64],
    encode: &[f64],
    t: f64,
) -> (&'f Function, f64) {
    let t = if d1 > d0 { t.clamp(d0, d1) } else { d0 };
    let k = functions.len();
    // Sub-domain i is [lower(i), upper(i)]; the outer bounds are the Domain.
    let lower = |i: usize| {
        i.checked_sub(1)
            .and_then(|j| bounds.get(j))
            .copied()
            .unwrap_or(d0)
    };
    let upper = |i: usize| bounds.get(i).copied().unwrap_or(d1);
    let idx = if t >= d1 {
        k - 1
    } else {
        (0..k).find(|&i| t < upper(i)).unwrap_or(k - 1)
    };
    let (in_lo, in_hi) = (lower(idx), upper(idx));
    let (out_lo, out_hi) = match encode.get(2 * idx..2 * idx + 2) {
        Some(&[lo, hi]) => (lo, hi),
        _ => (0.0, 1.0),
    };
    let x = if (in_hi - in_lo).abs() < f64::EPSILON {
        out_lo
    } else {
        ((t - in_lo) / (in_hi - in_lo)).mul_add(out_hi - out_lo, out_lo)
    };
    (&functions[idx], x)
}

/// Clip input `i` to `Domain` interval `i` (default `[0, 1]`); an empty
/// interval leaves the input as given.
fn clip_to_domains(domain: &[f64], inputs: &[f64]) -> Vec<f64> {
    inputs
        .iter()
        .zip(domain_pairs(domain, inputs.len()))
        .map(|(&t, (d0, d1))| if d1 > d0 { t.clamp(d0, d1) } else { t })
        .collect()
}

/// `values` padded with its last element, or truncated, to `n` entries;
/// `values` is non-empty.
fn pad(values: &[f64], n: usize) -> Vec<f64> {
    let mut v = values.to_vec();
    if let Some(&last) = values.last() {
        v.resize(n, last);
    }
    v
}

/// A colour array (`C0` or `C1`); `None` when absent or holding no number.
fn read_color(dict: &Dictionary, key: &[u8]) -> Option<Vec<f64>> {
    Some(numbers(dict, key)).filter(|v| !v.is_empty())
}

/// The numbers of the array under `key`, non-numeric elements skipped;
/// empty when the key is absent or not a direct array.
fn numbers(dict: &Dictionary, key: &[u8]) -> Vec<f64> {
    dict.get(key)
        .and_then(Object::as_array)
        .map(|a| a.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default()
}

/// The first `m` intervals of the flat `Domain` numbers `domain`; an
/// interval the array does not supply defaults to `[0, 1]`.
fn domain_pairs(domain: &[f64], m: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
    (0..m).map(|i| match domain.get(2 * i..2 * i + 2) {
        Some(&[d0, d1]) => (d0, d1),
        _ => (0.0, 1.0),
    })
}

/// Read the `Domain` key (1D) from a function or shading dict; defaults to
/// `[0, 1]`.
pub(super) fn read_fn_domain(dict: &Dictionary) -> (f64, f64) {
    domain_pairs(&numbers(dict, b"Domain"), 1)
        .next()
        .unwrap_or((0.0, 1.0))
}

/// Read a fixed-length array of `f64` values from a dictionary key.
pub(super) fn read_f64_array(dict: &Dictionary, key: &[u8], expected: usize) -> Option<Vec<f64>> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < expected {
        return None;
    }
    let vals: Vec<f64> = arr.iter().take(expected).filter_map(obj_to_f64).collect();
    if vals.len() < expected {
        return None;
    }
    Some(vals)
}

#[cfg(test)]
mod tests {
    use super::postscript::{MAX_PS_DEPTH, MAX_PS_STACK};
    use super::*;

    fn eval_function(doc: &Document, fn_obj: &Object, t: f64, n: usize) -> Option<Vec<f64>> {
        Function::load(doc, fn_obj).eval(t, n)
    }

    fn make_exp_dict(c0: f32, c1: f32, exponent: f32) -> pdf::Dictionary {
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(2));
        dict.set("C0", pdf::Object::Array(vec![pdf::Object::Real(c0)]));
        dict.set("C1", pdf::Object::Array(vec![pdf::Object::Real(c1)]));
        dict.set("N", pdf::Object::Real(exponent));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict
    }

    #[test]
    fn exponential_linear_midpoint() {
        let dict = make_exp_dict(0.0, 1.0, 1.0);
        let result = eval_function(&empty_doc(), &pdf::Object::Dictionary(dict), 0.5, 1)
            .expect("Type 2 evaluates");
        assert!(
            (result[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn exponential_at_t0_gives_c0() {
        let dict = make_exp_dict(0.2, 1.0, 1.0);
        let result = eval_function(&empty_doc(), &pdf::Object::Dictionary(dict), 0.0, 1)
            .expect("Type 2 evaluates");
        assert!((result[0] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn exponential_at_t1_gives_c1() {
        let dict = make_exp_dict(0.0, 0.8, 1.0);
        let result = eval_function(&empty_doc(), &pdf::Object::Dictionary(dict), 1.0, 1)
            .expect("Type 2 evaluates");
        assert!((result[0] - 0.8).abs() < 1e-5);
    }

    use crate::test_helpers::empty_doc;

    /// A Type 2 function `C0 [0] C1 [1]` with exponent `n` over `domain`.
    fn exp_over(domain: [f32; 2], n: f32) -> pdf::Object {
        let mut dict = make_exp_dict(0.0, 1.0, n);
        dict.set(
            "Domain",
            pdf::Object::Array(domain.iter().map(|&d| pdf::Object::Real(d)).collect()),
        );
        pdf::Object::Dictionary(dict)
    }

    /// PDF §7.10.3: `y = C0 + x^N × (C1 − C0)` on the input clipped to
    /// `Domain`, not on the input rescaled onto `[0, 1]`.
    #[test]
    fn exponential_exponentiates_the_domain_clipped_input() {
        let doc = empty_doc();
        let eval = |f: &pdf::Object, t| eval_function(&doc, f, t, 1).expect("Type 2")[0];
        let linear = exp_over([0.0, 2.0], 1.0);
        assert!((eval(&linear, 1.0) - 1.0).abs() < 1e-9);
        assert!((eval(&linear, 0.5) - 0.5).abs() < 1e-9);
        assert!((eval(&linear, 3.0) - 2.0).abs() < 1e-9, "clipped to Domain");
        let square = exp_over([0.0, 2.0], 2.0);
        assert!((eval(&square, 1.0) - 1.0).abs() < 1e-9);
        let symmetric = exp_over([-1.0, 1.0], 2.0);
        assert!((eval(&symmetric, -1.0) - 1.0).abs() < 1e-9);
        assert!((eval(&symmetric, 0.5) - 0.25).abs() < 1e-9);
    }

    /// A power with no finite real value evaluates to `C0`; a negative
    /// exponent is otherwise honoured.
    #[test]
    fn exponential_undefined_powers_are_c0() {
        let doc = empty_doc();
        let eval = |f: &pdf::Object, t| eval_function(&doc, f, t, 1).expect("Type 2")[0];
        assert!(eval(&exp_over([-1.0, 1.0], 0.5), -0.25).abs() < 1e-9);
        let reciprocal = exp_over([0.0, 2.0], -1.0);
        assert!(eval(&reciprocal, 0.0).abs() < 1e-9);
        assert!((eval(&reciprocal, 2.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn function_array_evaluates_one_function_per_channel() {
        let fns: Vec<pdf::Object> = [(0.0, 0.8), (0.2, 0.6), (0.0, 0.0), (1.0, 0.0)]
            .into_iter()
            .map(|(c0, c1)| pdf::Object::Dictionary(make_exp_dict(c0, c1, 1.0)))
            .collect();
        let doc = empty_doc();
        let out = eval_function(&doc, &pdf::Object::Array(fns), 0.5, 4).expect("4 channels");
        let want = [0.4, 0.4, 0.0, 0.5];
        for (got, want) in out.iter().zip(want) {
            assert!((got - want).abs() < 1e-5, "got {out:?}, want {want:?}");
        }
    }

    #[test]
    fn function_array_length_mismatch_is_unresolved() {
        let fns = vec![pdf::Object::Dictionary(make_exp_dict(0.0, 1.0, 1.0)); 3];
        let doc = empty_doc();
        assert!(eval_function(&doc, &pdf::Object::Array(fns), 0.5, 4).is_none());
    }

    #[test]
    fn eval_stitching_wrong_bounds_count_falls_back() {
        // PDF §7.10.3: a Type 3 stitching function with k sub-functions must
        // carry k−1 Bounds values.  Two sub-functions therefore need exactly
        // one bound; supplying zero is malformed.  eval_stitching_depth must
        // fall back to evaluating the first sub-function rather than panicking
        // on the missing breakpoint.
        let sub_a = make_exp_dict(0.2, 0.8, 1.0); // sub_a(0.5) = 0.5
        // sub_b chosen so sub_b(0.5) = 0.95, distinct from sub_a(0.5) = 0.5;
        // if the fallback used the wrong sub-function the assertion below
        // would fire.
        let sub_b = make_exp_dict(0.9, 1.0, 1.0);
        let mut stitching = pdf::Dictionary::new();
        stitching.set("FunctionType", pdf::Object::Integer(3));
        stitching.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        stitching.set(
            "Functions",
            pdf::Object::Array(vec![
                pdf::Object::Dictionary(sub_a),
                pdf::Object::Dictionary(sub_b),
            ]),
        );
        // Bounds intentionally empty — should have 1 entry for 2 sub-functions.
        stitching.set("Bounds", pdf::Object::Array(vec![]));
        stitching.set(
            "Encode",
            pdf::Object::Array(vec![
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
            ]),
        );

        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(stitching);

        // Must not panic; must use sub_a (the first sub-function) as fallback.
        let result =
            eval_function(&doc, &fn_obj, 0.5, 1).expect("fallback should produce a result");
        // sub_a at t=0.5 is 0.2 + 0.5 × (0.8 − 0.2) = 0.5.
        assert!(
            (result[0] - 0.5).abs() < 1e-5,
            "expected fallback to evaluate sub_a at t=0.5 → 0.5, got {}",
            result[0]
        );
    }

    #[test]
    fn indirect_function_is_compiled_once_per_document() {
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            "<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length 9 >>\nstream\n{ 2 mul }\nendstream",
        )]);
        let a = Function::load(&doc, &Object::Reference((4, 0)));
        let b = Function::load(&doc, &Object::Reference((4, 0)));
        assert!(
            Arc::ptr_eq(&a, &b),
            "second load must reuse the compiled function"
        );
        let out = a.eval(0.25, 1).expect("{ 2 mul } evaluates");
        assert!((out[0] - 0.5).abs() < 1e-12, "got {out:?}");
    }

    #[test]
    fn self_referencing_stitching_is_depth_bounded() {
        // 4 0 R stitches to itself: compilation recurses until the nesting
        // limit and evaluates the innermost level as its C0.
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            "<< /FunctionType 3 /Domain [0 1] /Functions [4 0 R] /Bounds [] /Encode [0 1] /C0 [0.25] >>",
        )]);
        let f = Function::load(&doc, &Object::Reference((4, 0)));
        let out = f.eval(0.5, 1).expect("innermost level evaluates as C0");
        assert!((out[0] - 0.25).abs() < 1e-12, "got {out:?}");
    }

    /// A stitching function naming itself in each of 16 sub-functions would
    /// compile 16^10 nodes if each path to it were compiled separately.
    #[test]
    fn repeated_sub_function_references_compile_once() {
        let refs = "4 0 R ".repeat(16);
        let bounds = (1..16)
            .map(|i| (f64::from(i) / 16.0).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let encode = "0 1 ".repeat(16);
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            &format!(
                "<< /FunctionType 3 /Domain [0 1] /Functions [{refs}] /Bounds [{bounds}] \
                 /Encode [{encode}] /C0 [0.25] >>"
            ),
        )]);
        let f = Function::load(&doc, &Object::Reference((4, 0)));
        let out = f.eval(0.5, 1).expect("innermost level evaluates as C0");
        assert!((out[0] - 0.25).abs() < 1e-12, "got {out:?}");
    }

    // ── Type 0 (Sampled) ──────────────────────────────────────────────────────

    /// Build a Type 0 sampled function as a Stream object with the given
    /// 8-bit sample bytes.  Domain = [0, 1], Range = [0, 1], 1 output channel
    /// unless `n` is given.
    fn make_sampled_8bit_stream(samples: &[u8]) -> pdf::Object {
        let len = i64::try_from(samples.len()).expect("test fixture < i64::MAX");
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(0));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        dict.set("Size", pdf::Object::Array(vec![pdf::Object::Integer(len)]));
        dict.set("BitsPerSample", pdf::Object::Integer(8));
        // Length is required for stream parsing in real PDFs; harmless to
        // include here since we hand the Stream directly to eval_function.
        dict.set("Length", pdf::Object::Integer(len));
        pdf::Object::Stream(pdf::Stream::new(dict, samples.to_vec()))
    }

    #[test]
    fn sampled_reads_sample_stream() {
        // Non-linear sample table [0, 0, 0, 0, 255]: ~0 for t < 0.75 and
        // ~1 at t = 1.
        let stream = make_sampled_8bit_stream(&[0, 0, 0, 0, 255]);
        let doc = empty_doc();

        let lo = eval_function(&doc, &stream, 0.0, 1).expect("eval at t=0");
        assert!(
            lo[0].abs() < 1e-5,
            "t=0 should sample idx 0 → 0/255 = 0.0, got {}",
            lo[0]
        );

        let mid = eval_function(&doc, &stream, 0.5, 1).expect("eval at t=0.5");
        assert!(
            mid[0].abs() < 1e-5,
            "t=0.5 should sample idx 2 → 0/255 = 0.0, got {}",
            mid[0]
        );

        let hi = eval_function(&doc, &stream, 1.0, 1).expect("eval at t=1");
        assert!(
            (hi[0] - 1.0).abs() < 1e-5,
            "t=1 should sample idx 4 → 255/255 = 1.0, got {}",
            hi[0]
        );
    }

    #[test]
    fn sampled_interpolates_between_samples() {
        // Samples [0, 100, 200] at indices [0, 1, 2].  t=0.25 maps to idx 0.5
        // (encode default = [0, n−1] = [0, 2], so t × 2 = 0.5).  Linear interp
        // between idx 0 (=0) and idx 1 (=100) at frac=0.5 → 50/255.
        let stream = make_sampled_8bit_stream(&[0, 100, 200]);
        let doc = empty_doc();

        let result = eval_function(&doc, &stream, 0.25, 1).expect("eval at t=0.25");
        let expected = 50.0 / 255.0;
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "expected linear interpolation 50/255 ≈ {expected}, got {}",
            result[0]
        );
    }

    /// A three-sample peak `[0, max, 0]` over `Domain [0 1]` as a
    /// one-input Type 0 function with the given depth, `Order` and `Range`.
    fn peak_function(bps: i64, samples: Vec<u8>, order: i64, range: [f32; 2]) -> pdf::Object {
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(0));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Integer(0), pdf::Object::Integer(1)]),
        );
        dict.set(
            "Range",
            pdf::Object::Array(range.iter().map(|&v| pdf::Object::Real(v)).collect()),
        );
        dict.set("Size", pdf::Object::Array(vec![pdf::Object::Integer(3)]));
        dict.set("BitsPerSample", pdf::Object::Integer(bps));
        dict.set("Order", pdf::Object::Integer(order));
        pdf::Object::Stream(pdf::Stream::new(dict, samples))
    }

    fn assert_peak(f: &pdf::Object, lo: f64, hi: f64) {
        let doc = empty_doc();
        for (t, want) in [
            (0.0, lo),
            (0.25, f64::midpoint(lo, hi)),
            (0.5, hi),
            (1.0, lo),
        ] {
            let got = eval_function(&doc, f, t, 1).expect("sampled function evaluates")[0];
            assert!((got - want).abs() < 1e-6, "t={t}: got {got}, want {want}");
        }
    }

    #[test]
    fn sampled_reads_every_bit_depth() {
        assert_peak(
            &peak_function(1, vec![0b0100_0000], 1, [0.0, 1.0]),
            0.0,
            1.0,
        );
        assert_peak(
            &peak_function(2, vec![0b0011_0000], 1, [0.0, 1.0]),
            0.0,
            1.0,
        );
        assert_peak(&peak_function(4, vec![0x0F, 0x00], 1, [0.0, 1.0]), 0.0, 1.0);
        assert_peak(
            &peak_function(12, vec![0x00, 0x0F, 0xFF, 0x00, 0x00], 1, [0.0, 1.0]),
            0.0,
            1.0,
        );
        assert_peak(
            &peak_function(24, [[0; 3], [0xFF; 3], [0; 3]].concat(), 1, [0.0, 1.0]),
            0.0,
            1.0,
        );
        assert_peak(
            &peak_function(32, [[0; 4], [0xFF; 4], [0; 4]].concat(), 1, [0.0, 1.0]),
            0.0,
            1.0,
        );
    }

    #[test]
    fn sampled_order_3_interpolates_the_samples() {
        assert_peak(&peak_function(8, vec![0, 255, 0], 3, [0.0, 1.0]), 0.0, 1.0);
    }

    #[test]
    fn sampled_decode_defaults_to_range() {
        assert_peak(&peak_function(8, vec![0, 255, 0], 1, [0.2, 0.6]), 0.2, 0.6);
    }

    #[test]
    fn sampled_input_count_must_match_size() {
        // Two grid axes, one input supplied.
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(0));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
                pdf::Object::Real(0.0),
                pdf::Object::Real(1.0),
            ]),
        );
        dict.set(
            "Size",
            pdf::Object::Array(vec![pdf::Object::Integer(2), pdf::Object::Integer(2)]),
        );
        dict.set("BitsPerSample", pdf::Object::Integer(8));
        let stream = pdf::Object::Stream(pdf::Stream::new(dict, vec![0, 64, 128, 255]));
        assert!(eval_function(&empty_doc(), &stream, 0.5, 1).is_none());
    }

    // ── Range clip ────────────────────────────────────────────────────────────

    #[test]
    fn range_clip_clamps_above_max() {
        // Exponential 0 → 2 evaluated at t=1 would normally return 2.0; the
        // Range entry [0, 1] should clip it to 1.0.
        let mut dict = make_exp_dict(0.0, 2.0, 1.0);
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(dict);
        let result = eval_function(&doc, &fn_obj, 1.0, 1).expect("eval at t=1");
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "Range clip should cap at 1.0; got {}",
            result[0]
        );
    }

    // ── Type 4 (PostScript calculator) ────────────────────────────────────────

    /// Build a Type 4 function as a Stream whose content is `program`.
    /// Domain = [0, 1]; `range_pairs` flattened into the `Range` array.
    fn make_ps_stream(program: &str, range_pairs: &[(f64, f64)]) -> pdf::Object {
        let bytes = program.as_bytes().to_vec();
        let len = i64::try_from(bytes.len()).expect("test fixture < i64::MAX");
        let mut dict = pdf::Dictionary::new();
        dict.set("FunctionType", pdf::Object::Integer(4));
        dict.set(
            "Domain",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let mut range = Vec::new();
        for (lo, hi) in range_pairs {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "test fixture values are small and exactly representable in f32"
            )]
            {
                range.push(pdf::Object::Real(*lo as f32));
                range.push(pdf::Object::Real(*hi as f32));
            }
        }
        dict.set("Range", pdf::Object::Array(range));
        dict.set("Length", pdf::Object::Integer(len));
        pdf::Object::Stream(pdf::Stream::new(dict, bytes))
    }

    #[test]
    fn ps_simple_arithmetic() {
        // { 2 mul } at input 0.5 → 1.0
        let f = make_ps_stream("{ 2 mul }", &[(0.0, 10.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.5, 1).expect("eval { 2 mul }");
        assert!((r[0] - 1.0).abs() < 1e-9, "expected 1.0, got {}", r[0]);
    }

    #[test]
    fn ps_ifelse_branches() {
        // { dup 0.5 lt { pop 0 } { pop 1 } ifelse }
        let prog = "{ dup 0.5 lt { pop 0 } { pop 1 } ifelse }";
        let f = make_ps_stream(prog, &[(0.0, 1.0)]);
        let doc = empty_doc();
        let lo = eval_function(&doc, &f, 0.3, 1).expect("eval at 0.3");
        assert!(lo[0].abs() < 1e-9, "0.3 < 0.5 → 0, got {}", lo[0]);
        let hi = eval_function(&doc, &f, 0.7, 1).expect("eval at 0.7");
        assert!((hi[0] - 1.0).abs() < 1e-9, "0.7 ≥ 0.5 → 1, got {}", hi[0]);
    }

    #[test]
    fn ps_separation_to_cmyk_shape() {
        // obj-12-style: 1 input tint → 4 CMYK outputs.  This program maps a
        // "Black" separation tint t to DeviceCMYK [0 0 0 t]: push three zeros
        // then duplicate-free copy the tint through.
        let prog = "{ 0 0 0 4 -1 roll }";
        let f = make_ps_stream(prog, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.6, 4).expect("eval 1→4");
        assert_eq!(r.len(), 4);
        assert!(r[0].abs() < 1e-9 && r[1].abs() < 1e-9 && r[2].abs() < 1e-9);
        assert!(
            (r[3] - 0.6).abs() < 1e-9,
            "K channel should be tint, got {}",
            r[3]
        );
    }

    #[test]
    fn ps_stack_operators() {
        // exch: { 1 exch sub } at t=0.25 → 0.75
        let f = make_ps_stream("{ 1 exch sub }", &[(0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.25, 1).expect("exch");
        assert!((r[0] - 0.75).abs() < 1e-9, "expected 0.75, got {}", r[0]);

        // index: { 2 3 1 index } leaves [2 3 2]; pop top two via pop pop → 2.
        let f2 = make_ps_stream("{ pop 2 3 1 index add add }", &[(0.0, 100.0)]);
        let r2 = eval_function(&doc, &f2, 0.0, 1).expect("index");
        // 2 + 3 + 2 = 7
        assert!((r2[0] - 7.0).abs() < 1e-9, "expected 7, got {}", r2[0]);

        // copy: { pop 1 2 2 copy add add add } → 1 2 1 2 → 1+2+1+2 = 6
        let f3 = make_ps_stream("{ pop 1 2 2 copy add add add }", &[(0.0, 100.0)]);
        let r3 = eval_function(&doc, &f3, 0.0, 1).expect("copy");
        assert!((r3[0] - 6.0).abs() < 1e-9, "expected 6, got {}", r3[0]);

        // roll: { pop 1 2 3 3 1 roll add add } → roll right by 1 → 3 1 2 → sum 6
        let f4 = make_ps_stream("{ pop 1 2 3 3 1 roll add add }", &[(0.0, 100.0)]);
        let r4 = eval_function(&doc, &f4, 0.0, 1).expect("roll");
        assert!((r4[0] - 6.0).abs() < 1e-9, "expected 6, got {}", r4[0]);
    }

    #[test]
    fn ps_malformed_returns_none() {
        let doc = empty_doc();
        // Missing closing brace.
        let bad = make_ps_stream("{ 2 mul", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad, 0.5, 1).is_none());
        // Unknown operator.
        let bad2 = make_ps_stream("{ frobnicate }", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad2, 0.5, 1).is_none());
        // Stack underflow (add with one operand).
        let bad3 = make_ps_stream("{ add }", &[(0.0, 1.0)]);
        assert!(eval_function(&doc, &bad3, 0.5, 1).is_none());
        // Not enough outputs requested vs produced.
        let bad4 = make_ps_stream("{ }", &[(0.0, 1.0), (0.0, 1.0)]);
        assert!(eval_function(&doc, &bad4, 0.5, 2).is_none());
    }

    #[test]
    fn ps_recursion_bomb_is_bounded() {
        // Deeply nested `{{{{…}}}}` must terminate loud-graceful (None), not
        // overflow the native stack during parse or exec.  Spawn on a thread
        // with a small stack so an unbounded recursion would actually abort
        // the test rather than silently pass on a large default stack.
        let depth = MAX_PS_DEPTH + 5_000;
        let mut prog = String::with_capacity(depth * 2 + 8);
        prog.push('{');
        for _ in 0..depth {
            prog.push('{');
        }
        for _ in 0..depth {
            prog.push('}');
        }
        prog.push_str(" if }");
        let handle = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let doc = empty_doc();
                let f = make_ps_stream(&prog, &[(0.0, 1.0)]);
                eval_function(&doc, &f, 0.5, 1)
            })
            .expect("spawn bounded-stack thread");
        let result = handle.join().expect("recursion-bomb must not overflow");
        assert!(
            result.is_none(),
            "over-deep program must fail loud-graceful"
        );
    }

    #[test]
    fn ps_stack_bomb_is_bounded() {
        // `n copy` duplicates the top n elements; feeding it the exact
        // current stack length doubles the stack with a single step.  A
        // chain of doublings reaches MAX_PS_STACK in a handful of ops while
        // charging almost no step budget, so only the operand-stack cap
        // stops it.  Build the doubling chain statically (length is known
        // after each `copy`): start at 1, then `1 copy`→2, `2 copy`→4, …
        let mut prog = String::from("{");
        let mut len: usize = 1; // the single input tint already on the stack
        while len <= MAX_PS_STACK {
            prog.push(' ');
            prog.push_str(&len.to_string());
            prog.push_str(" copy");
            len *= 2;
        }
        prog.push_str(" }");
        let doc = empty_doc();
        let f = make_ps_stream(&prog, &[(0.0, 1.0)]);
        assert!(
            eval_function(&doc, &f, 0.5, 1).is_none(),
            "operand-stack bomb must fail loud-graceful, not OOM"
        );
    }

    #[test]
    fn ps_div_zero_and_domain_errors_fail_graceful() {
        let doc = empty_doc();
        // 1 0 div → +Inf: must not leak Inf into the colour.
        let f = make_ps_stream("{ pop 1 0 div }", &[(-1e9, 1e9)]);
        assert!(
            eval_function(&doc, &f, 0.5, 1).is_none(),
            "div-by-zero must fail loud-graceful, not push Inf"
        );
        // -1 sqrt → NaN.
        let f2 = make_ps_stream("{ pop -1 sqrt }", &[(-1e9, 1e9)]);
        assert!(
            eval_function(&doc, &f2, 0.5, 1).is_none(),
            "sqrt(-1) → None"
        );
        // 0 ln → -Inf.
        let f3 = make_ps_stream("{ pop 0 ln }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f3, 0.5, 1).is_none(), "ln(0) → None");
        // idiv by zero.
        let f4 = make_ps_stream("{ pop 5 0 idiv }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f4, 0.5, 1).is_none(), "idiv 0 → None");
        // A literal that parses to Inf must not reach the output.
        let f5 = make_ps_stream("{ pop 1e400 }", &[(-1e9, 1e9)]);
        assert!(eval_function(&doc, &f5, 0.5, 1).is_none(), "1e400 → None");
    }

    #[test]
    fn ps_comments_are_stripped() {
        // A PostScript `%` comment to end-of-line must not break parsing.
        let prog = "{ % map tint through\n 2 mul % double it\n }";
        let f = make_ps_stream(prog, &[(0.0, 10.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 0.5, 1).expect("commented program evaluates");
        assert!((r[0] - 1.0).abs() < 1e-9, "expected 1.0, got {}", r[0]);
    }

    #[test]
    fn ps_sot_apprenti_tint_program() {
        // The exact obj-12 tint transform from sot-apprenti-1774:
        // Separation /Black → DeviceCMYK, mapping tint t to [0 0 0 t].
        // A regression here re-blanks the page.
        let prog = "{dup 0 mul exch dup 0 mul exch dup 0 mul exch 1 mul }";
        let f = make_ps_stream(prog, &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]);
        let doc = empty_doc();
        let r = eval_function(&doc, &f, 1.0, 4).expect("sot-apprenti tint at t=1");
        assert_eq!(r.len(), 4);
        assert!(
            r[0].abs() < 1e-9 && r[1].abs() < 1e-9 && r[2].abs() < 1e-9,
            "C/M/Y must be 0, got {r:?}"
        );
        assert!(
            (r[3] - 1.0).abs() < 1e-9,
            "K must equal the tint (1.0), got {}",
            r[3]
        );
    }

    #[test]
    fn range_clip_clamps_below_min() {
        // C0 = -0.5 would normally pass through at t=0; Range [0, 1] should
        // clip it to 0.0.
        let mut dict = make_exp_dict(-0.5, 1.0, 1.0);
        dict.set(
            "Range",
            pdf::Object::Array(vec![pdf::Object::Real(0.0), pdf::Object::Real(1.0)]),
        );
        let doc = empty_doc();
        let fn_obj = pdf::Object::Dictionary(dict);
        let result = eval_function(&doc, &fn_obj, 0.0, 1).expect("eval at t=0");
        assert!(
            result[0].abs() < 1e-5,
            "Range clip should floor at 0.0; got {}",
            result[0]
        );
    }
}
