# Changes from upstream

This crate is a local fork of `hayro-jpeg2000`, vendored from upstream commit
`79926198e557e06fea8a8edcb0bc577eb0130ec5` (2026-10-23). It is not published
and changes are not sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `Cargo.toml`: workspace-inherited fields resolved to values; `publish =
  false`; the `image` integration is no longer a default feature (the test
  suite enables it with `--features image`).
