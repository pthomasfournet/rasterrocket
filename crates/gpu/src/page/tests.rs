use super::*;

const EMPTY: CoverageRows<'static> = CoverageRows {
    y_min: 0,
    row_starts: &[0],
    spans: &[],
    bytes: &[],
};

/// An empty 8-bit page of `width × height` white pixels.
fn page8(width: u32, height: u32) -> PageCommands {
    PageCommands::new(width, height, SampleDepth::Eight, [255; 3])
}

#[test]
fn bins_list_each_command_in_its_tiles_in_paint_order() {
    let mut page = page8(40, 20);
    let b = |x0, y0, x1, y1| PixelBounds { x0, y0, x1, y1 };
    for bounds in [b(0, 0, 39, 19), b(20, 2, 25, 3), b(5, 17, 100, 100)] {
        page.push_coverage(EMPTY, bounds, [0; 3], 255)
            .expect("push");
    }
    let bins = page.bin().expect("bin");
    assert_eq!(bins.items.len(), 6 + 1 + 3);
    assert_eq!((bins.tiles_x, bins.tiles_y), (3, 2));
    let tile = |t: usize| {
        bins.items[bins.offsets[t] as usize..bins.offsets[t + 1] as usize]
            .iter()
            .map(|i| i[0])
            .collect::<Vec<_>>()
    };
    assert_eq!(tile(0), [0]);
    assert_eq!(tile(1), [0, 1]);
    assert_eq!(tile(2), [0]);
    assert_eq!(tile(3), [0, 2]);
    assert_eq!(tile(4), [0, 2]);
    assert_eq!(tile(5), [0, 2]);
}

#[test]
fn bounds_off_the_page_push_nothing() {
    let mut page = page8(10, 10);
    let bounds = PixelBounds {
        x0: 10,
        y0: 0,
        x1: 20,
        y1: 5,
    };
    page.push_coverage(EMPTY, bounds, [0; 3], 255)
        .expect("push");
    assert!(page.is_empty());
}

#[test]
fn tables_are_rebased_onto_the_page() {
    let mut page = page8(16, 16);
    let table = CoverageRows {
        y_min: 3,
        row_starts: &[0, 2, 3],
        spans: &[[1, 2, 0], [4, 9, -1], [2, 2, 2]],
        bytes: &[10, 20, 30],
    };
    let bounds = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 15,
        y1: 15,
    };
    page.push_coverage(table, bounds, [0; 3], 255)
        .expect("push");
    page.push_coverage(table, bounds, [0; 3], 255)
        .expect("push");
    assert_eq!(page.row_starts(), [0, 2, 3, 3, 5, 6]);
    assert_eq!(page.spans()[3..], [[1, 2, 3], [4, 9, -1], [2, 2, 5]]);
    let (words, bases) = page.table_words();
    let base = bases.coverage;
    assert_eq!(base, 18);
    assert_eq!(
        words[18..],
        [
            u32::from_le_bytes([10, 20, 30, 10]),
            u32::from_le_bytes([20, 30, 0, 0])
        ]
    );
}

#[test]
#[should_panic(expected = "malformed")]
fn a_span_past_its_bytes_is_refused() {
    let mut page = page8(16, 16);
    let table = CoverageRows {
        y_min: 0,
        row_starts: &[0, 1],
        spans: &[[0, 3, 1]],
        bytes: &[1, 2, 3, 4],
    };
    let bounds = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 0,
    };
    let _ = page.push_coverage(table, bounds, [0; 3], 255);
}

#[test]
fn opaque_runs_cover_whole_pixels_and_merge_overlaps() {
    let mut page = page8(32, 32);
    let mut runs = vec![(3, 7, 9), (2, 1, 4), (2, 3, 6), (2, 10, 10)];
    page.push_opaque_runs(&mut runs, [1, 2, 3]).expect("push");
    assert_eq!(page.len(), 1);
    let c = page.cmds()[0];
    assert_eq!((c.x0, c.y0, c.x1, c.y1), (1, 2, 10, 3));
    assert_eq!((c.table_y0, c.n_rows, c.a_input), (2, 2, 255));
    assert_eq!(page.row_starts(), [0, 2, 3]);
    assert_eq!(page.spans(), [[1, 6, -1], [10, 10, -1], [7, 9, -1]]);
}

#[test]
fn opaque_runs_are_clipped_to_the_page() {
    let mut page = page8(32, 8);
    let mut runs = vec![
        (2, -1_000_000_000, 3),
        (2, 30, 1_000_000_000),
        (-5, 0, 4),
        (9, 0, 4),
        (3, 40, 50),
        (4, i32::MIN, i32::MAX),
    ];
    page.push_opaque_runs(&mut runs, [0; 3]).expect("push");
    let c = page.cmds()[0];
    assert_eq!((c.x0, c.y0, c.x1, c.y1), (0, 2, 31, 4));
    assert_eq!(page.row_starts(), [0, 2, 2, 3]);
    assert_eq!(page.spans(), [[0, 3, -1], [30, 31, -1], [0, 31, -1]]);

    let mut off_page = vec![(-1, 0, 5), (2, 32, 40)];
    page.push_opaque_runs(&mut off_page, [0; 3]).expect("push");
    assert_eq!(page.len(), 1);
}

#[test]
fn a_push_past_the_byte_limit_fails_and_leaves_the_page_unchanged() {
    let n = PAGE_COMMANDS_MAX_BYTES / size_of::<[i32; 3]>();
    let spans = vec![[0, 0, -1]; n];
    let row_starts = [0, u32::try_from(n).expect("small")];
    let table = CoverageRows {
        y_min: 0,
        row_starts: &row_starts,
        spans: &spans,
        bytes: &[],
    };
    let bounds = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };
    let mut page = page8(16, 16);
    let before = page.bytes();
    assert_eq!(
        page.push_coverage(table, bounds, [0; 3], 255),
        Err(PageCommandsError::TooLarge)
    );
    assert!(page.is_empty());
    assert_eq!(page.bytes(), before);
    let fits = CoverageRows {
        spans: &spans[..n / 2],
        row_starts: &[0, u32::try_from(n / 2).expect("small")],
        ..table
    };
    page.push_coverage(fits, bounds, [0; 3], 255)
        .expect("half the limit fits");
    assert!(page.bytes() <= PAGE_COMMANDS_MAX_BYTES);
}

#[test]
fn opaque_runs_paint_at_the_largest_sample_of_the_depth() {
    let mut page = PageCommands::new(8, 8, SampleDepth::Sixteen, [65_535; 3]);
    page.push_opaque_runs(&mut [(1, 0, 3)], [1, 300, 65_535])
        .expect("push");
    let c = page.cmds()[0];
    assert_eq!((c.rg, c.b, c.a_input), (1 | 300 << 16, 65_535, 65_535));
    assert_eq!(page.paper(), [65_535; 3]);
}

#[test]
#[should_panic(expected = "exceeds 8 bits")]
fn a_sample_past_the_depth_is_refused() {
    let _ = page8(4, 4).push_coverage(
        EMPTY,
        PixelBounds {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 3,
        },
        [256, 0, 0],
        255,
    );
}
