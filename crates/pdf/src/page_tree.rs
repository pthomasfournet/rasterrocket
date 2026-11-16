//! Logarithmic page-tree descent.
//!
//! Unlike `Document::get_pages()` (which walks the whole tree), this module
//! descends from the catalog's `/Pages` root to a single requested page index
//! using each interior node's `/Count` to pick the correct `/Kids` branch.
//! O(log pages) lookups for documents whose page-tree depth is shallow
//! (typically 4–5 levels for 100k-page docs).

use std::collections::HashSet;

use crate::{
    dictionary::Dictionary,
    document::Document,
    error::PdfError,
    object::{Object, ObjectId},
};

/// Maximum page-tree depth.  Bounds descent against pathological / cyclic PDFs.
const MAX_DEPTH: usize = 64;

/// Resolve a 0-based page index to its `ObjectId`, walking only the
/// root-to-leaf path.
///
/// Returns `PdfError::PageOutOfRange { page, total }` if `idx` is
/// outside `[0, total)` where `total` is [`Document::page_count_fast`].
///
/// The descent trusts each interior node's `/Count`.  Where the counts do
/// not match the tree (a node without `/Count`, a `/Count` past the
/// leaves, a `/Kids` cycle), the page is taken from the document-order walk
/// of [`Document::get_pages`], memoised on first use.
///
/// Cycle protection: tracks visited object numbers in a `HashSet<u32>`.
/// Depth protection: bounded to `MAX_DEPTH` levels.
///
/// # Errors
/// [`PdfError::PageOutOfRange`] for an index past the page count; the
/// errors of [`Document::get_object`] for the nodes visited; and
/// [`PdfError::BadObject`] or [`PdfError::MissingKey`] when a `/Kids` entry
/// is malformed, or when the counts disagree with the tree and the walk has
/// no page `idx` either.
//
// The descent does not consult linearization hints: `LinearizationHints`
// holds only the `/Linearized` dictionary entries, and the page offset
// hint table, which would map idx to a byte offset, is not parsed.
pub fn descend_to_page_index(doc: &Document, idx: u32) -> Result<ObjectId, PdfError> {
    let total = doc.page_count_fast();
    if idx >= total {
        return Err(PdfError::PageOutOfRange { page: idx, total });
    }
    match descend_by_count(doc, idx) {
        Ok(id) => Ok(id),
        Err(Descent::Miscounted(err)) => {
            log::debug!("page index {idx}: {err}; taking the page from the page-tree walk");
            doc.walked_pages().get(idx as usize).copied().ok_or(err)
        }
        Err(Descent::Failed(err)) => Err(err),
    }
}

/// Why [`descend_by_count`] found no page.
pub(crate) enum Descent {
    /// The `/Count` entries do not describe the tree: an interior node
    /// without `/Count`, a `/Count` past the leaves below it, a cycle or a
    /// path deeper than [`MAX_DEPTH`].  A walk of the tree decides.
    Miscounted(PdfError),
    /// A node could not be read, or a `/Kids` entry is malformed.
    Failed(PdfError),
}

impl From<PdfError> for Descent {
    fn from(err: PdfError) -> Self {
        Self::Failed(err)
    }
}

/// The page at 0-based index `idx` by descent from the `/Pages` root,
/// choosing each branch by its `/Count`.
pub(crate) fn descend_by_count(doc: &Document, idx: u32) -> Result<ObjectId, Descent> {
    let pages_root = doc.pages_root_id()?;
    let mut visited: HashSet<u32> = HashSet::new();
    let mut current = pages_root;
    let mut remaining = idx;

    for _ in 0..MAX_DEPTH {
        if !visited.insert(current.0) {
            return Err(Descent::Miscounted(PdfError::BadObject {
                id: current.0,
                detail: "cyclic /Kids reference during descent".into(),
            }));
        }
        let node_obj = doc.get_dict_arc(current)?;
        let node_dict = dict_of(&node_obj, current)?;
        if !is_interior(node_dict) {
            return Ok(current);
        }

        let kids = resolve_kids(doc, current, node_dict)?;

        // Flat-tree fast path: when /Count exactly matches kids.len(), every
        // kid holds exactly one page.  Index directly instead of summing the
        // siblings' /Count entries; only the selected kid is probed, since
        // it may be an interior node with a single leaf below it.
        let parent_count = node_dict.get(b"Count").and_then(Object::as_u32);
        if parent_count == Some(u32::try_from(kids.len()).unwrap_or(u32::MAX)) {
            let kid = kids.get(remaining as usize).copied().ok_or_else(|| {
                Descent::Miscounted(PdfError::BadObject {
                    id: current.0,
                    detail: format!(
                        "flat /Pages: remaining={remaining} >= kids.len()={}",
                        kids.len()
                    ),
                })
            })?;
            let kid_obj = doc.get_dict_arc(kid)?;
            if !is_interior(dict_of(&kid_obj, kid)?) {
                return Ok(kid);
            }
            current = kid;
            remaining = 0;
            continue;
        }

        // Mixed tree — sum /Count across siblings.
        let mut descended = false;
        for kid_ref in &kids {
            let kid_obj = doc.get_dict_arc(*kid_ref)?;
            let kid_dict = dict_of(&kid_obj, *kid_ref)?;
            let kid_count = kid_subtree_count(*kid_ref, kid_dict)?;
            if remaining < kid_count {
                // Found the kid containing the requested page.  If the kid
                // itself is a leaf, return immediately to avoid the
                // top-of-loop re-fetch on the next iteration.
                if !is_interior(kid_dict) {
                    return Ok(*kid_ref);
                }
                current = *kid_ref;
                descended = true;
                break;
            }
            remaining -= kid_count;
        }
        if !descended {
            return Err(Descent::Miscounted(PdfError::BadObject {
                id: current.0,
                detail: format!(
                    "ran out of /Kids before reaching requested page (remaining={remaining})",
                ),
            }));
        }
    }
    Err(Descent::Miscounted(PdfError::BadObject {
        id: current.0,
        detail: "page-tree descent exceeded depth limit".into(),
    }))
}

/// The dictionary of an object returned by [`Document::get_dict_arc`], which
/// is always a dictionary or a stream.
fn dict_of(obj: &Object, id: ObjectId) -> Result<&Dictionary, PdfError> {
    obj.as_dict().ok_or_else(|| PdfError::BadObject {
        id: id.0,
        detail: format!("expected Dictionary or Stream, got {}", obj.enum_variant()),
    })
}

/// Whether a page-tree node is an interior `/Pages` node rather than a page
/// object: typed `/Pages`, or, when `/Type` is neither `/Page` nor `/Pages`
/// (absent or wrong), carrying a `/Kids` entry.  A node typed `/Page` is a
/// page.  Every page-tree walker classifies nodes by this one rule.
pub(crate) fn is_interior(dict: &Dictionary) -> bool {
    match dict.get(b"Type").and_then(Object::as_name) {
        Some(b"Page") => false,
        Some(b"Pages") => true,
        _ => dict.get(b"Kids").is_some(),
    }
}

/// Read a kid's subtree page count: 1 for a page, `/Count` for an
/// interior node (see [`is_interior`]).  [`Descent::Miscounted`] if an
/// interior node lacks a valid `/Count`.
fn kid_subtree_count(kid_id: ObjectId, kid_dict: &Dictionary) -> Result<u32, Descent> {
    if !is_interior(kid_dict) {
        return Ok(1);
    }
    kid_dict
        .get(b"Count")
        .and_then(Object::as_u32)
        .ok_or_else(|| {
            Descent::Miscounted(PdfError::BadObject {
                id: kid_id.0,
                detail: "interior page-tree node missing /Count".into(),
            })
        })
}

/// The ids of the `/Kids` of node `node_id` (dictionary `node_dict`), one
/// per element: the referenced object for a reference, the number
/// [`Document::page_tree_node`] gives a direct dictionary, and `None` for
/// any other element.  `/Kids` is an array or a reference to one; the
/// object holding the array names the site of its direct elements.
///
/// # Errors
/// [`PdfError::MissingKey`] without `/Kids`, the errors of
/// [`Document::get_object`] for a referenced array, and
/// [`PdfError::BadObject`] when `/Kids` is not an array.
pub(crate) fn kid_ids(
    doc: &Document,
    node_id: ObjectId,
    node_dict: &Dictionary,
) -> Result<Vec<Option<ObjectId>>, PdfError> {
    let kids_obj = node_dict.get(b"Kids").ok_or(PdfError::MissingKey("Kids"))?;
    let referenced;
    let (holder, kids) = match kids_obj {
        Object::Array(a) => (node_id.0, a.as_slice()),
        Object::Reference(rid) => {
            referenced = doc.get_object(*rid)?;
            let kids = referenced.as_array().ok_or_else(|| PdfError::BadObject {
                id: node_id.0,
                detail: "/Kids reference does not resolve to an array".into(),
            })?;
            (rid.0, kids)
        }
        _ => {
            return Err(PdfError::BadObject {
                id: node_id.0,
                detail: "/Kids is neither an array nor a reference".into(),
            });
        }
    };
    Ok(kids
        .iter()
        .zip(0u32..)
        .map(|(kid, slot)| doc.page_tree_node(kid, holder, slot, Some(node_id)))
        .collect())
}

/// [`kid_ids`] with every element a page-tree node.
///
/// Rejects an element that is neither a reference nor a dictionary hard
/// rather than filtering it out.  The descent indexes into the result by
/// page index, so a filtered list breaks the page-index invariant: a
/// `/Kids [3 0 R 4 0 R 99]` with `/Count 3` would silently misroute
/// page 2 to a `BadObject` error rather than the `99` slot the source
/// document intended.  Hard-fail surfaces the corruption to the caller.
fn resolve_kids(
    doc: &Document,
    node_id: ObjectId,
    node_dict: &Dictionary,
) -> Result<Vec<ObjectId>, PdfError> {
    let kids = kid_ids(doc, node_id, node_dict)?;
    let total = kids.len();
    let nodes: Vec<ObjectId> = kids.into_iter().flatten().collect();
    if nodes.len() != total {
        return Err(PdfError::BadObject {
            id: node_id.0,
            detail: format!(
                "/Kids contains {} entries that are neither references nor dictionaries \
                 (have {} of {total} as nodes)",
                total - nodes.len(),
                nodes.len(),
            ),
        });
    }
    Ok(nodes)
}

// ── Page iterator ─────────────────────────────────────────────────────────────

/// Most node visits plus `/Kids` entries one page-tree walk processes.  A
/// kid referenced from several `/Kids` arrays is visited once per
/// reference, so a tree of shared nodes describes exponentially many
/// pages in linear space; the walk ends at this bound instead.
pub(crate) const MAX_PAGE_TREE_WORK: usize = 1 << 22;

/// The document-order walk of [`Document::get_pages`].
pub(crate) struct PageIter<'a> {
    doc: &'a Document,
    /// Interior nodes on the traversal path, each with its resolved `/Kids`.
    stack: Vec<PageTreeFrame>,
    /// Object numbers of the nodes in `stack`; a kid already present is a
    /// cyclic `/Kids` reference and is skipped.
    on_stack: HashSet<u32>,
    /// The `/Pages` root, visited on the first call to `next`.
    root: Option<ObjectId>,
    page_num: u32,
    /// Work left of [`MAX_PAGE_TREE_WORK`].
    budget: usize,
}

/// One interior page-tree node: its `/Kids` references, resolved once, and
/// the index of the next kid to visit.
struct PageTreeFrame {
    id: ObjectId,
    kids: Vec<ObjectId>,
    next: usize,
}

impl<'a> PageIter<'a> {
    pub(crate) fn new(doc: &'a Document) -> Self {
        // A catalog without a page tree root yields no pages; logged so
        // corruption doesn't go silent.
        let root = doc
            .pages_root_id()
            .inspect_err(|e| log::warn!("PageIter: no page tree root, no pages yielded: {e}"))
            .ok();
        Self {
            doc,
            stack: Vec::new(),
            on_stack: HashSet::new(),
            root,
            page_num: 0,
            budget: MAX_PAGE_TREE_WORK,
        }
    }

    /// Take `work` units from the budget; on exhaustion the walk ends.
    fn charge(&mut self, work: usize) -> bool {
        if let Some(left) = self.budget.checked_sub(work) {
            self.budget = left;
            return true;
        }
        log::warn!(
            "page-tree walk stopped after {MAX_PAGE_TREE_WORK} node visits and /Kids entries \
             ({} pages found); the tree repeats nodes past that bound",
            self.page_num
        );
        self.budget = 0;
        self.root = None;
        self.stack.clear();
        self.on_stack.clear();
        false
    }

    /// Visit node `id`: a page object (see [`is_interior`])
    /// is returned as the next page; an interior node with a `/Kids` array
    /// (direct, or an indirect reference to one) is pushed.  Anything else is
    /// skipped.
    fn visit(&mut self, id: ObjectId) -> Option<(u32, ObjectId)> {
        if !self.charge(1) {
            return None;
        }
        let obj = self.doc.get_object(id).ok()?;
        let dict = obj.as_dict()?;
        if !is_interior(dict) {
            self.page_num += 1;
            return Some((self.page_num, id));
        }
        let kids: Vec<ObjectId> = kid_ids(self.doc, id, dict)
            .ok()?
            .into_iter()
            .flatten()
            .collect();
        if !self.charge(kids.len()) {
            return None;
        }
        let _ = self.on_stack.insert(id.0);
        self.stack.push(PageTreeFrame { id, kids, next: 0 });
        None
    }
}

impl Iterator for PageIter<'_> {
    type Item = (u32, ObjectId);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(root) = self.root.take()
            && let Some(page) = self.visit(root)
        {
            return Some(page);
        }
        loop {
            let frame = self.stack.last_mut()?;
            let Some(&kid) = frame.kids.get(frame.next) else {
                let id = frame.id;
                let _ = self.stack.pop();
                let _ = self.on_stack.remove(&id.0);
                continue;
            };
            frame.next += 1;
            if self.on_stack.contains(&kid.0) {
                log::warn!("PageIter: cyclic /Kids reference to obj {} skipped", kid.0);
                continue;
            }
            if let Some(page) = self.visit(kid) {
                return Some(page);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-page PDF.  Verifies that `descend_to_page_index` returns the
    /// correct page object for both indices.  The eager `get_pages()` and
    /// the lazy `descend_to_page_index` must agree.
    fn two_page_pdf() -> Vec<u8> {
        // %PDF-1.4\n = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n" = 47 bytes → obj2 at 56
        // "2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>\nendobj\n" = 61 bytes → obj3 at 117
        // "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → obj4 at 186
        // "4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → xref at 255
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 5\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000117 00000 n\r\n\
0000000186 00000 n\r\n\
trailer\n<</Size 5 /Root 1 0 R>>\n\
startxref\n255\n%%EOF"
            .to_vec()
    }

    #[test]
    fn descend_finds_first_page() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let id = descend_to_page_index(&doc, 0).expect("page 0");
        assert_eq!(id.0, 3);
    }

    #[test]
    fn descend_finds_second_page() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let id = descend_to_page_index(&doc, 1).expect("page 1");
        assert_eq!(id.0, 4);
    }

    #[test]
    fn descend_rejects_out_of_range() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let err = descend_to_page_index(&doc, 2).expect_err("page 2 must error");
        match err {
            PdfError::PageOutOfRange { page, total } => {
                assert_eq!(page, 2);
                assert_eq!(total, 2);
            }
            other => panic!("expected PageOutOfRange, got {other:?}"),
        }
    }

    /// `u32::MAX` is a representative pathological index — the function must
    /// reject it cleanly with `PageOutOfRange`, not panic on overflow during
    /// `remaining` arithmetic and not silently truncate.
    #[test]
    fn descend_rejects_u32_max_index() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let err = descend_to_page_index(&doc, u32::MAX).expect_err("u32::MAX must error");
        match err {
            PdfError::PageOutOfRange { page, total } => {
                assert_eq!(page, u32::MAX);
                assert_eq!(total, 2);
            }
            other => panic!("expected PageOutOfRange, got {other:?}"),
        }
    }

    /// Verify `descend_to_page_index` agrees with the eager `get_pages()` iterator
    /// across the same fixture.
    #[test]
    fn descend_agrees_with_eager_walk() {
        let doc = Document::from_bytes_owned(two_page_pdf()).unwrap();
        let eager: Vec<ObjectId> = doc.get_pages().map(|(_, id)| id).collect();
        for (idx, expected_id) in eager.iter().enumerate() {
            let got = descend_to_page_index(&doc, u32::try_from(idx).unwrap()).expect("descend");
            assert_eq!(got, *expected_id, "mismatch at idx {idx}");
        }
    }

    /// Three-level page tree (catalog → root /Pages → 2 interior nodes →
    /// 4 leaves).  Forces the descent to actually iterate `kids` past
    /// index 0 and decrement `remaining` between siblings; the two-leaf
    /// flat tree above never exercises that branch.
    ///
    ///   1 = Catalog → /Pages 2 0 R
    ///   2 = /Pages /Kids [3 0 R 4 0 R] /Count 4
    ///   3 = /Pages /Kids [5 0 R 6 0 R] /Count 2 /Parent 2 0 R
    ///   4 = /Pages /Kids [7 0 R 8 0 R] /Count 2 /Parent 2 0 R
    ///   5–8 = /Page leaves
    fn three_level_pdf() -> Vec<u8> {
        // Byte offsets (computed exactly by summing segment lengths):
        //   header "%PDF-1.4\n"                                             9 bytes  → obj1 at 9
        //   obj1 (catalog)                                                 47 bytes  → obj2 at 56
        //   obj2 (root /Pages, /Count 4)                                   61 bytes  → obj3 at 117
        //   obj3 (interior /Pages, /Count 2, kids 5+6)                     75 bytes  → obj4 at 192
        //   obj4 (interior /Pages, /Count 2, kids 7+8)                     75 bytes  → obj5 at 267
        //   obj5..obj8 (leaves, Parent 3 / 3 / 4 / 4)                      69 bytes  → next at 336/405/474
        // xref at 543
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 4 0 R] /Count 4>>\nendobj\n\
3 0 obj\n<</Type /Pages /Kids [5 0 R 6 0 R] /Count 2 /Parent 2 0 R>>\nendobj\n\
4 0 obj\n<</Type /Pages /Kids [7 0 R 8 0 R] /Count 2 /Parent 2 0 R>>\nendobj\n\
5 0 obj\n<</Type /Page /Parent 3 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
6 0 obj\n<</Type /Page /Parent 3 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
7 0 obj\n<</Type /Page /Parent 4 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
8 0 obj\n<</Type /Page /Parent 4 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 9\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000117 00000 n\r\n\
0000000192 00000 n\r\n\
0000000267 00000 n\r\n\
0000000336 00000 n\r\n\
0000000405 00000 n\r\n\
0000000474 00000 n\r\n\
trailer\n<</Size 9 /Root 1 0 R>>\n\
startxref\n543\n%%EOF"
            .to_vec()
    }

    /// Hand-built two-page PDF with a deliberately malformed /Kids array
    /// containing an inline integer (`99`) where a reference should be.
    /// `resolve_kids` must return `BadObject` rather than silently dropping
    /// the bad entry — silent filtering would misroute page 2 (the slot the
    /// integer occupies) to a misleading "ran out of /Kids" error.
    fn malformed_kids_pdf() -> Vec<u8> {
        // Byte offsets (computed exactly):
        //   header                                                              9 bytes  → obj1 at 9
        //   obj1 catalog                                                       47 bytes  → obj2 at 56
        //   obj2 /Pages /Kids [3 0 R 99 4 0 R] /Count 3                        64 bytes  → obj3 at 120
        //   obj3 leaf                                                          69 bytes  → obj4 at 189
        //   obj4 leaf                                                          69 bytes  → xref at 258
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R 99 4 0 R] /Count 3>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
4 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 5\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000120 00000 n\r\n\
0000000189 00000 n\r\n\
trailer\n<</Size 5 /Root 1 0 R>>\n\
startxref\n258\n%%EOF"
            .to_vec()
    }

    #[test]
    fn descend_rejects_non_reference_kids() {
        let doc = Document::from_bytes_owned(malformed_kids_pdf()).expect("open");
        let err = descend_to_page_index(&doc, 0).expect_err("malformed /Kids must error");
        match err {
            PdfError::BadObject { detail, .. } => {
                assert!(
                    detail.contains("neither references nor dictionaries"),
                    "expected a non-node /Kids error, got: {detail}"
                );
            }
            other => panic!("expected BadObject, got {other:?}"),
        }
    }

    #[test]
    fn descend_three_level_tree() {
        let doc = Document::from_bytes_owned(three_level_pdf()).expect("open");
        // page 0 → leaf object 5 (first kid of first interior)
        // page 1 → leaf object 6 (second kid of first interior — exercises
        //          the inner kid loop's increment past the first kid)
        // page 2 → leaf object 7 (first kid of second interior — exercises
        //          remaining decrement across an interior boundary)
        // page 3 → leaf object 8 (second kid of second interior)
        for (idx, expected_obj) in [(0u32, 5u32), (1, 6), (2, 7), (3, 8)] {
            let id = descend_to_page_index(&doc, idx)
                .unwrap_or_else(|e| panic!("descend idx={idx}: {e:?}"));
            assert_eq!(id.0, expected_obj, "idx={idx} mapped to wrong leaf");
        }
        // Also confirm /Pages /Count = 4 is read directly via fast path.
        assert_eq!(doc.page_count_fast(), 4);
    }

    /// A PDF whose objects 1, 2, … are `bodies`, with a classic xref table.
    fn assemble(bodies: &[&str]) -> Vec<u8> {
        use std::fmt::Write as _;

        let mut out = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, body) in bodies.iter().enumerate() {
            offsets.push(out.len());
            write!(out, "{} 0 obj\n{body}\nendobj\n", i + 1).unwrap();
        }
        let xref = out.len();
        write!(out, "xref\n0 {}\n0000000000 65535 f\r\n", bodies.len() + 1).unwrap();
        for off in offsets {
            write!(out, "{off:010} 00000 n\r\n").unwrap();
        }
        write!(
            out,
            "trailer\n<</Size {} /Root 1 0 R>>\nstartxref\n{xref}\n%%EOF",
            bodies.len() + 1
        )
        .unwrap();
        out.into_bytes()
    }

    /// Root `/Pages` with `root_count` → [untyped page 3, `/Pages` 4 →
    /// [page 5, page 6]]: the root's `/Count` differs from its kid count, so
    /// the descent sums the kids' counts.
    fn untyped_leaf_pdf(root_count: &str) -> Vec<u8> {
        assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            &format!("<</Type /Pages /Kids [3 0 R 4 0 R] {root_count}>>"),
            "<</Parent 2 0 R>>",
            "<</Type /Pages /Parent 2 0 R /Kids [5 0 R 6 0 R] /Count 2>>",
            "<</Type /Page /Parent 4 0 R>>",
            "<</Type /Page /Parent 4 0 R>>",
        ])
    }

    /// A node with neither `/Type` nor `/Kids` is a page to the descent and
    /// to the eager walk alike.
    #[test]
    fn untyped_leaf_is_a_page_to_every_walker() {
        let doc = Document::from_bytes_owned(untyped_leaf_pdf("/Count 3")).unwrap();
        let descended: Vec<u32> = (0..3)
            .map(|i| descend_to_page_index(&doc, i).expect("page").0)
            .collect();
        assert_eq!(descended, [3, 5, 6]);
        let walked: Vec<u32> = doc.get_pages().map(|(_, id)| id.0).collect();
        assert_eq!(walked, [3, 5, 6]);

        // Without a root `/Count` the page count comes from the eager walk.
        let doc = Document::from_bytes_owned(untyped_leaf_pdf("")).unwrap();
        assert_eq!(doc.page_count(), 3);
    }

    /// The pages of `doc` by descent, in index order.
    fn descended(doc: &Document) -> Vec<u32> {
        (0..doc.page_count_fast())
            .map(|i| descend_to_page_index(doc, i).expect("page").0)
            .collect()
    }

    /// A root `/Count` past the leaves gives way to the walked count; a
    /// `/Count` the tree bears out is kept.
    #[test]
    fn count_past_the_leaves_takes_the_walked_count() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            "<</Type /Pages /Kids [3 0 R] /Count 3>>",
            "<</Type /Page /Parent 2 0 R>>",
        ]))
        .unwrap();
        assert_eq!(doc.page_count_fast(), 1);
        assert_eq!(descended(&doc), [3]);
        assert!(matches!(
            descend_to_page_index(&doc, 1),
            Err(PdfError::PageOutOfRange { page: 1, total: 1 })
        ));

        let doc = Document::from_bytes_owned(three_level_pdf()).unwrap();
        assert_eq!(doc.page_count_fast(), 4);
    }

    /// Without `/Count` anywhere, a kid listed twice is a page twice, and
    /// every index resolves through the walk.
    #[test]
    fn tree_without_counts_resolves_repeated_kids_by_walk() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            "<</Type /Pages /Kids [3 0 R 3 0 R]>>",
            "<</Type /Pages /Parent 2 0 R /Kids [4 0 R 4 0 R 4 0 R]>>",
            "<</Type /Page /Parent 3 0 R>>",
        ]))
        .unwrap();
        assert_eq!(doc.page_count_fast(), 6);
        assert_eq!(descended(&doc), [4; 6]);
    }

    /// An interior node without `/Count` below a root that has one is
    /// counted by the walk.
    #[test]
    fn interior_node_without_count_resolves_by_walk() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            "<</Type /Pages /Kids [3 0 R 4 0 R] /Count 3>>",
            "<</Type /Pages /Parent 2 0 R /Kids [5 0 R 6 0 R]>>",
            "<</Type /Page /Parent 2 0 R>>",
            "<</Type /Page /Parent 3 0 R>>",
            "<</Type /Page /Parent 3 0 R>>",
        ]))
        .unwrap();
        assert_eq!(doc.page_count_fast(), 3);
        assert_eq!(descended(&doc), [5, 6, 4]);
    }

    /// A chain of nodes each listing the next twice describes 2^40 pages in
    /// 42 objects; the walk stops at its work bound.
    #[test]
    fn shared_node_page_tree_walk_is_bounded() {
        let mut bodies = vec!["<</Type /Catalog /Pages 2 0 R>>".to_owned()];
        for num in 2..42 {
            bodies.push(format!(
                "<</Type /Pages /Kids [{next} 0 R {next} 0 R]>>",
                next = num + 1
            ));
        }
        bodies.push("<</Type /Page>>".to_owned());
        let bodies: Vec<&str> = bodies.iter().map(String::as_str).collect();
        let doc = Document::from_bytes_owned(assemble(&bodies)).unwrap();
        let start = std::time::Instant::now();
        let count = doc.page_count_fast();
        assert!(count > 0 && (count as usize) < MAX_PAGE_TREE_WORK);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(20),
            "page-tree walk must be bounded, took {:?}",
            start.elapsed()
        );
    }

    /// `/Type` decides over `/Kids`: a `/Page` with a stray `/Kids` is a page,
    /// and an untyped node with `/Kids` is interior.
    #[test]
    fn type_decides_before_kids() {
        let node = |src: &str| {
            let mut pos = 0;
            match crate::object::parse_object(src.as_bytes(), &mut pos) {
                Some(Object::Dictionary(d)) => d,
                other => panic!("{other:?}"),
            }
        };
        assert!(!is_interior(&node("<</Type /Page /Kids []>>")));
        assert!(is_interior(&node("<</Type /Pages>>")));
        assert!(is_interior(&node("<</Kids []>>")));
        assert!(!is_interior(&node("<</Type /Annot>>")));
        assert!(is_interior(&node("<</Type /Annot /Kids []>>")));
    }

    /// The ids of the walk, in document order.
    fn walked(doc: &Document) -> Vec<ObjectId> {
        doc.get_pages().map(|(_, id)| id).collect()
    }

    /// A catalog whose `/Pages` is a direct dictionary without `/Kids` has
    /// that dictionary as its one page.
    #[test]
    fn direct_pages_root_is_a_page() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages <</MediaBox [0 0 200 100]>>>>",
        ]))
        .unwrap();
        assert_eq!(doc.page_count_fast(), 1);
        let id = descend_to_page_index(&doc, 0).expect("page 0");
        assert!(id.0 > crate::object::MAX_OBJECT_NUMBER);
        assert_eq!(walked(&doc), [id]);
        assert!(!doc.is_undefined(id.0));
        assert!(doc.is_undefined(id.0 + 1));
        assert!(doc.get_object((id.0 + 1, 0)).is_err());
        let media_box = doc.get_inherited_page_attr(id, b"MediaBox");
        assert_eq!(
            media_box
                .as_ref()
                .and_then(Object::as_array)
                .map(<[_]>::len),
            Some(4)
        );
    }

    /// Direct dictionaries in `/Kids`, under a direct root and under an
    /// indirect node, are pages and interior nodes to both walkers, with
    /// the same id on every walk, and inherit through the node holding
    /// them.
    #[test]
    fn direct_kids_are_page_tree_nodes() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages <</Type /Pages /Count 4 /Rotate 90 /Kids [\
             <</Type /Page>> 2 0 R \
             <</Type /Pages /Count 2 /Kids [<</Type /Page /Rotate 180>> 3 0 R]>>]>>>>",
            "<</Type /Page>>",
            "<</Type /Pages /Count 1 /Rotate 270 /Kids [<</Type /Page>>]>>",
        ]))
        .unwrap();
        assert_eq!(doc.page_count_fast(), 4);
        let pages = walked(&doc);
        assert_eq!(pages.len(), 4);
        assert_eq!(walked(&doc), pages);
        let descended: Vec<ObjectId> = (0..4)
            .map(|i| descend_to_page_index(&doc, i).expect("page"))
            .collect();
        assert_eq!(descended, pages);
        assert_eq!(pages[1], (2, 0));
        let direct = [pages[0], pages[2], pages[3]];
        assert!(
            direct
                .iter()
                .all(|id| id.0 > crate::object::MAX_OBJECT_NUMBER)
        );
        let rotate = |id| {
            doc.get_inherited_page_attr(id, b"Rotate")
                .and_then(|o| o.as_u32())
        };
        assert_eq!(direct.map(rotate), [Some(90), Some(180), Some(270)]);
        assert_eq!(rotate(pages[1]), None);
    }

    /// A direct node in a shared `/Kids` array that lists the array again
    /// is the same node at every visit: the cycle is cut and the walk ends.
    #[test]
    fn direct_kid_listing_its_own_array_is_a_cycle() {
        let doc = Document::from_bytes_owned(assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            "<</Type /Pages /Kids 3 0 R>>",
            "[<</Type /Pages /Kids 3 0 R>> <</Type /Page>>]",
        ]))
        .unwrap();
        let pages = walked(&doc);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0], pages[1]);
        assert_eq!(doc.page_count_fast(), 2);
        assert_eq!(descended(&doc).len(), 2);
    }
}
