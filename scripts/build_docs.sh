#!/usr/bin/env bash
# Build the project docs to PDF.
#
#   RESULTS.tex  -> RESULTS.pdf        (pdflatex; the formulas version)
#   METHODS.md   -> METHODS.pdf        (pandoc/xelatex)
#   RESULTS.md   -> RESULTS_brief.pdf  (pandoc/xelatex)
#
# Helvetica Neue lacks a few glyphs the markdown uses. xelatex DROPS missing glyphs
# with a warning rather than an error, so "->" and "sigma" would silently vanish from
# the PDF. `newunicodechar` would fix it but is absent from TeX Live basic, so we
# substitute ASCII at build time instead -- safe inside code spans too, where math
# mode would render literally. The .md sources are never modified.
set -euo pipefail
cd "$(dirname "$0")/.."

FONT_ARGS=(--pdf-engine=xelatex -V geometry:margin=1in
           -V mainfont="Helvetica Neue" -V monofont="Menlo" -V fontsize=10pt --toc)

sanitize() {   # stdin -> stdout, glyph-safe
  sed -e 's/→/->/g' -e 's/←/<-/g' -e 's/≤/<=/g' -e 's/≥/>=/g' \
      -e 's/≫/>>/g'  -e 's/≪/<</g' -e 's/≈/~/g'  -e 's/≠/!=/g' \
      -e 's/σ/sigma/g' -e 's/μ/mu/g' -e 's/κ/kappa/g' -e 's/Δ/delta/g'
}

build_md() {   # $1 = source .md, $2 = output .pdf
  local tmp log missing
  tmp="$(mktemp -t docbuild).md"; log="$(mktemp -t docbuild).log"
  sanitize < "$1" > "$tmp"
  # capture stderr from the REAL build. Re-running to /dev/null to count warnings
  # fails (xelatex cannot write there), which yields zero matches and reports clean
  # while glyphs are silently dropped -- exactly the bug this check exists to catch.
  pandoc "$tmp" -o "$2" "${FONT_ARGS[@]}" 2>"$log"
  missing="$(grep -c 'Missing character' "$log" || true)"
  echo "  $2  (missing glyphs: ${missing})"
  if [ "$missing" -ne 0 ]; then
    grep -o 'no . (U+[0-9A-F]*)' "$log" | sort -u | sed 's/^/    /'
    echo "    ERROR: glyphs would be dropped -- add them to sanitize()"
    rm -f "$tmp" "$log"; exit 1
  fi
  rm -f "$tmp" "$log"
}

echo "building docs:"
pdflatex -interaction=nonstopmode -halt-on-error RESULTS.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error RESULTS.tex >/dev/null   # 2x: refs/TOC
echo "  RESULTS.pdf  ($(pdfinfo RESULTS.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
build_md METHODS.md METHODS.pdf
build_md RESULTS.md RESULTS_brief.pdf
rm -f RESULTS.aux RESULTS.log RESULTS.out RESULTS.toc
echo "done."
