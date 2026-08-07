#!/usr/bin/env bash
# Build the project documentation.
#
#   REPORT.tex -> REPORT.pdf   -- the single combined methods + results document.
#
# Methods and results are deliberately interwoven: each design choice is followed by
# the measurement that justified it. They were previously split across METHODS.md and
# RESULTS.md, which drifted apart within a day; that split is retired. Both remain in
# git history if the earlier prose is ever wanted.
#
# Run twice: the table of contents needs a second pass to resolve.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "building docs:"
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex >/dev/null

miss=$(grep -ci 'missing character\|undefined control sequence' REPORT.log || true)
pages=$(pdfinfo REPORT.pdf 2>/dev/null | awk '/^Pages/{print $2}')
echo "  REPORT.pdf  (${pages} pages, ${miss} glyph/macro problems)"
rm -f REPORT.aux REPORT.log REPORT.out REPORT.toc
[ "$miss" -eq 0 ] || { echo "  ERROR: see REPORT.log"; exit 1; }
echo "done."
