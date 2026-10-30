# Predefined CMaps

The CMaps ISO 32000-2 Table 116 lists as predefined for the Adobe-GB1,
Adobe-CNS1, Adobe-Japan1 and Adobe-Korea1 character collections (59 files;
Identity-H and Identity-V need none), plus every CMap they reach through
`usecmap`.  Each is the Adobe resource file, zlib-compressed, and is
inflated and parsed on first use.

They come from Adobe's `cmap-resources` repository
(github.com/adobe-type-tools/cmap-resources), copyright 1990-2023 Adobe,
under the BSD 3-Clause licence in `LICENSE.md`.  `scripts/vendor_cmaps.py`
regenerates them from a checkout.
