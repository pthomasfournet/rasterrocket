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

# Bundled fallback fonts

`noto/` holds thirteen Noto faces compiled into the renderer behind the
`bundled-fonts` Cargo feature (on by default in `rasterrocket` and
`rasterrocket-cli`).  A character that neither a font's substitute nor any
installed face covers is drawn from them, the last resort before `.notdef`;
an installed face that covers the character is always preferred.

| File | Source face (version) | Code ranges |
|---|---|---|
| `NotoSans-GreekCyrillic.ttf.z` | Noto Sans Regular (2.004) | Greek, Greek Extended, Cyrillic and its supplements and extensions |
| `NotoSerif-GreekCyrillic.ttf.z` | Noto Serif Regular (2.004) | as above |
| `NotoSansHebrew.ttf.z` | Noto Sans Hebrew Regular (3.000) | Hebrew, Hebrew presentation forms |
| `NotoSerifHebrew.ttf.z` | Noto Serif Hebrew Regular (2.000) | as above |
| `NotoSansArabic.ttf.z` | Noto Sans Arabic Regular (2.005) | Arabic, Arabic Supplement, Extended-A, presentation forms A and B |
| `NotoNaskhArabic.ttf.z` | Noto Naskh Arabic Regular (2.004) | as above |
| `NotoSansDevanagari.ttf.z` | Noto Sans Devanagari Regular (2.001) | Devanagari, Devanagari Extended |
| `NotoSerifDevanagari.ttf.z` | Noto Serif Devanagari Regular (2.001) | as above |
| `NotoSansThai.ttf.z` | Noto Sans Thai Regular (2.000) | Thai |
| `NotoSerifThai.ttf.z` | Noto Serif Thai Regular (2.000) | Thai |
| `NotoSansMath.ttf.z` | Noto Sans Math Regular (2.001) | letterlike symbols, number forms, arrows, mathematical operators and symbols, technical, geometric shapes, miscellaneous symbols, dingbats |
| `NotoSansSymbols.ttf.z` | Noto Sans Symbols Regular (2.001) | as above |
| `NotoSansSymbols2.ttf.z` | Noto Sans Symbols 2 Regular (2.003) | as above |

The source files are those of the Debian/Ubuntu package `fonts-noto-core`
20201225-2build1, built from the Noto Project
(<https://github.com/notofonts/noto-fonts>).  Copyright 2015–2020 Google
LLC / Google Inc., licensed under the SIL Open Font License 1.1, whose text,
as distributed with Noto, is in `noto/OFL.txt`.  The faces declare no
Reserved Font Name.

`scripts/build_bundled_fonts.py` regenerates them with fontTools 4.61.1:

    python3 scripts/build_bundled_fonts.py --src /usr/share/fonts/truetype/noto

It subsets each face to the exact ranges listed in the script, without
layout tables or hinting, keeps the `.notdef` outline, the source's
timestamp and the copyright, licence and naming records (`name` IDs 0–6,
13, 14), and stores the result zlib-compressed at level 9.  The output is
byte-for-byte reproducible from the same sources.
