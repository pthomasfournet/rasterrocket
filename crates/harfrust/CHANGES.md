# Changes from upstream

This crate is a local fork of `harfrust`, vendored from harfrust commit
`60b28ea22b5261710018d69c168a762bcb28794c` (2026-07-03, the 0.12.0 release).
It is not published and changes are not sent upstream.

0.12.0 is the release built on read-fonts 0.41, the base of the read-fonts
fork in `crates/read_fonts`. Moving to a later release goes with rebasing that
fork.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.99), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields are resolved to values; `publish =
  false`; `read-fonts` is the fork in `crates/read_fonts`. The benchmarks
  and their dev-dependencies are dropped.

- Upstream's test suite (`harfrust/tests`, about 6 000 shaping cases with
  their fonts) is not vendored. It passed in full on this fork's sources
  linked against the read-fonts fork when the fork was made.

## Lints under the 1.99 minimum toolchain

Behaviour is unchanged by each of these.

- `hb/face.rs`: `shape`, re-exported only under `experimental_font_api`,
  is compiled only under that feature.
- `hb/glyph_metrics.rs`: two side-bearing lookups return through `?`.
- `hb/unicode.rs`: `compose` returns through `?` and initialises its result
  in one expression; the Hangul composition tests use `is_multiple_of`.
