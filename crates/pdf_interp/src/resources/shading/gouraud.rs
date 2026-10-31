//! Gouraud-shaded triangle meshes: free-form (Type 4, PDF §8.7.4.5.5) and
//! lattice-form (Type 5, §8.7.4.5.6).

use pdf::Dictionary;

use super::mesh::{CornerSource, MeshVertex, TriangleSink, emit_triangle};
use super::stream::{BitReader, VertexFormat, parse_bits_per_flag};
use crate::resources::dict_ext::DictExt;

/// Decode a Type 4 (free-form Gouraud triangle mesh) stream into `out`.
///
/// Each vertex record (PDF §8.7.4.5.5) is a `BitsPerFlag`-bit edge flag
/// followed by the vertex, and starts on a byte boundary.  The flag of a
/// vertex that opens a triangle selects how it forms one:
/// - 0 → it and the next two vertices form a new triangle A, B, C (the next
///   two vertices' flags are ignored);
/// - 1 → it extends edge (B, C) of the previous triangle: triangle (B, C, D);
/// - 2 → it extends edge (C, A) of the previous triangle: triangle (C, A, D).
pub(super) fn decode_free_form<S: CornerSource>(
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    format: &VertexFormat<'_>,
    out: &mut TriangleSink,
) {
    let Some(bpf) = parse_bits_per_flag(sh, 4) else {
        return;
    };
    let mut reader = BitReader::new(data);
    let mut fields = vec![0.0; source.components()];
    // The most recently formed triangle, which flags 1 and 2 extend.
    let mut prev: Option<[MeshVertex<S::Corner>; 3]> = None;
    // The vertices of a flag-0 triangle read so far.
    let mut opening: Vec<MeshVertex<S::Corner>> = Vec::with_capacity(3);

    while !out.is_full()
        && let Some(flag) = reader.read_bits(bpf)
    {
        let Some(v) = format.read_vertex(&mut reader, source, &mut fields) else {
            break;
        };
        reader.align();

        let tri = if opening.is_empty() {
            match (flag, prev) {
                (0, _) => {
                    opening.push(v);
                    None
                }
                (1, Some(p)) => Some([p[1], p[2], v]),
                (2, Some(p)) => Some([p[2], p[0], v]),
                (f, _) => {
                    log::debug!("shading/type4: flag {f} cannot open a triangle — skipping vertex");
                    None
                }
            }
        } else {
            opening.push(v);
            <[_; 3]>::try_from(opening.as_slice())
                .ok()
                .inspect(|_| opening.clear())
        };

        if let Some(t) = tri {
            prev = Some(t);
            emit_triangle(source, t, out);
        }
    }
}

/// Decode a Type 5 (lattice-form Gouraud mesh) stream into `out`.
///
/// The vertices form rows of `VerticesPerRow`, in row-major order, each
/// vertex record starting on a byte boundary (PDF §8.7.4.5.6).  Each cell
/// of two adjacent rows splits along its diagonal into the triangles
/// (top-left, top-right, bottom-left) and (top-right, bottom-right,
/// bottom-left).
pub(super) fn decode_lattice<S: CornerSource>(
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    format: &VertexFormat<'_>,
    out: &mut TriangleSink,
) {
    let Some(verts_per_row) = sh.get_i64(b"VerticesPerRow") else {
        log::warn!("shading/type5: VerticesPerRow is missing — skipping the shading");
        return;
    };
    let Some(vpr) = usize::try_from(verts_per_row).ok().filter(|&v| v >= 2) else {
        log::warn!(
            "shading/type5: VerticesPerRow {verts_per_row} is below 2 — skipping the shading"
        );
        return;
    };
    let mut reader = BitReader::new(data);
    let mut fields = vec![0.0; source.components()];
    // Every vertex record occupies at least one byte, so no row holds more
    // vertices than the stream has bytes.
    let row_capacity = vpr.min(data.len());
    let mut prev_row: Vec<MeshVertex<S::Corner>> = Vec::new();

    'rows: while !out.is_full() {
        let mut row = Vec::with_capacity(row_capacity);
        for _ in 0..vpr {
            let Some(v) = format.read_vertex(&mut reader, source, &mut fields) else {
                break 'rows;
            };
            reader.align();
            row.push(v);
        }
        for (top, bottom) in prev_row.windows(2).zip(row.windows(2)) {
            emit_triangle(source, [top[0], top[1], bottom[0]], out);
            emit_triangle(source, [top[1], bottom[1], bottom[0]], out);
        }
        prev_row = row;
    }
}

#[cfg(test)]
mod tests {
    use super::super::mesh::DirectColors;
    use super::super::stream::default_decode;
    use super::super::{ColorSpace, ShadingCs};
    use super::*;
    use crate::test_helpers::empty_doc;
    use pdf::Object;

    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    fn gray_mesh_dict(extra: &[(&str, i64)]) -> Dictionary {
        let mut sh = Dictionary::new();
        sh.set("BitsPerCoordinate", Object::Integer(8));
        sh.set("BitsPerComponent", Object::Integer(8));
        for &(k, v) in extra {
            sh.set(k, Object::Integer(v));
        }
        sh
    }

    /// Decode `data` as a gray mesh of type `shading_type` (4 or 5).
    fn decode(
        sh: &Dictionary,
        data: &[u8],
        shading_type: u8,
    ) -> Vec<[raster::shading::gouraud::GouraudVertex; 3]> {
        let doc = empty_doc();
        let source = DirectColors {
            cs: ShadingCs {
                doc: &doc,
                space: &ColorSpace::DeviceGray,
            },
        };
        let decode = default_decode(1);
        let format =
            VertexFormat::parse(sh, shading_type, &decode, &IDENTITY, 1.0).expect("format");
        let mut sink = TriangleSink::new();
        if shading_type == 4 {
            decode_free_form(sh, data, &source, &format, &mut sink);
        } else {
            decode_lattice(sh, data, &source, &format, &mut sink);
        }
        sink.finish()
    }

    /// `VerticesPerRow` sizes no allocation beyond what the stream holds.
    #[test]
    fn huge_vertices_per_row_reads_the_stream_it_has() {
        let sh = gray_mesh_dict(&[("VerticesPerRow", 1 << 60)]);
        let tris = decode(&sh, &[0, 0, 0, 255, 0, 128, 0, 255, 255], 5);
        assert!(tris.is_empty(), "one partial row forms no triangle");
    }

    #[test]
    fn a_two_by_two_lattice_forms_two_triangles() {
        let sh = gray_mesh_dict(&[("VerticesPerRow", 2)]);
        let data = [0, 0, 0, 255, 0, 85, 0, 255, 170, 255, 255, 255];
        let tris = decode(&sh, &data, 5);
        assert_eq!(tris.len(), 2);
        assert_eq!(tris[0].map(|v| v.color[0]), [0, 85, 170]);
        assert_eq!(tris[1].map(|v| v.color[0]), [85, 255, 170]);
    }

    #[test]
    fn free_form_flags_extend_the_previous_triangle() {
        let sh = gray_mesh_dict(&[("BitsPerFlag", 8)]);
        // Flag 0 opens A B C; flag 1 extends (B, C); flag 2 extends (C, A).
        let data = [
            0, 0, 0, 10, // A
            0, 255, 0, 20, // B (flag ignored)
            0, 0, 255, 30, // C (flag ignored)
            1, 255, 255, 40, // D: (B, C, D)
            2, 128, 128, 50, // E: (D, B, E), edge (C, A) of (B, C, D)
        ];
        let tris = decode(&sh, &data, 4);
        let colors: Vec<[u8; 3]> = tris.iter().map(|t| t.map(|v| v.color[0])).collect();
        assert_eq!(colors, [[10, 20, 30], [20, 30, 40], [40, 20, 50]]);
    }
}
