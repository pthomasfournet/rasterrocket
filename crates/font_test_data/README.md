# font-test-data

> **Vendored test data.** This directory is a copy of upstream's
> [`font-test-data`](https://github.com/googlefonts/fontations/tree/main/font-test-data)
> crate at fontations commit `1d1672367c7909013b4e366085d68d5ad612ee67`
> (2026-11-02, crate version 0.10.1), the base of the local `read-fonts` and
> `skrifa` forks. It
> exists so that the test suites of the local `read-fonts` and `skrifa` forks
> (`crates/read_fonts`, `crates/skrifa`) and of `rasterrocket-font` run
> against the fonts upstream wrote them for; it is a dev-dependency only and
> never part of a build. The crate is not published (`publish = false`).
> Its changes from upstream are listed at the top of its `Cargo.toml`.
>
> Everything below is upstream's README, kept as written.

Test files to support tests, split out due to use by multiple crates.

See the README in the test_data directory for information on adding and
rebuilding test cases.
