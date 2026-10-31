# Predefined CMaps

The 59 CMaps ISO 32000-2 Table 116 lists as predefined for the Adobe-GB1,
Adobe-CNS1, Adobe-Japan1 and Adobe-Korea1 character collections
(Identity-H and Identity-V need no file).  Every CMap one of them reaches
through `usecmap` is itself in that list.  Each is the Adobe resource
file, zlib-compressed; `src/resources/cmap_predefined.rs` embeds it, and
inflates and parses it on first use.

They come from Adobe's `cmap-resources` repository
(github.com/adobe-type-tools/cmap-resources), copyright 1990-2023 Adobe,
under the BSD 3-Clause licence in `LICENSE.md`.  `scripts/vendor_cmaps.py`
regenerates them from a checkout.
