# Bundled substitute fonts

`foxit/` holds the fourteen base-14 substitute faces compiled into the
renderer (`Sans`, `Serif`, `Fixed` in regular, bold, italic and bold italic,
plus `Symbol` and `Dingbats`), one bare CFF program each.  They are drawn
for a font without an embedded program when no metric-compatible face is
installed, and always for the Symbol and ZapfDingbats families.

They are the Foxit fonts shipped with PDFium, copyright 2014 Foxit Software
Inc. and the PDFium Authors, under the BSD 3-Clause licence in
`foxit/LICENSE`.  `scripts/extract_foxit_fonts.py` regenerates them from a
PDFium checkout (`core/fxge/fontdata/chromefontdata/`).  PDFium's two
multiple-master faces are not bundled.
