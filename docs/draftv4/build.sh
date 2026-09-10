#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs main.tex
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
else
  printf 'Install Tectonic or XeLaTeX with IEEEtran and xeCJK.\n' >&2
  exit 1
fi
