# font-test-data

> **Vendored test data.** This directory is an unmodified copy of upstream's
> [`font-test-data`](https://github.com/googlefonts/fontations/tree/main/font-test-data)
> crate at fontations commit `b373cdbc6043a791fb23b3e0f413649458f5e63a`
> (2026-07-04, crate version 0.7.0), the base of the local `skrifa` fork. It
> exists so that the test suites of the local `read-fonts` and `skrifa` forks
> (`crates/read_fonts`, `crates/skrifa`) and of `rasterrocket-font` run
> against the fonts upstream wrote them for; it is a dev-dependency only and
> never part of a build. The crate is not published (`publish = false`).
>
> Everything below is upstream's README, kept as written.

Test files to support tests, split out due to use by multiple crates.

See the README in the test_data directory for information on adding and
rebuilding test cases.
