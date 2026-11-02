#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs main.tex
elif [[ -x "$HOME/.local/bin/tectonic" ]]; then
  "$HOME/.local/bin/tectonic" --keep-logs main.tex
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
elif command -v conda >/dev/null 2>&1 && conda run -n latex tectonic --version >/dev/null 2>&1; then
  conda run --no-capture-output -n latex tectonic --keep-logs main.tex
else
  printf 'Tectonic or XeLaTeX with IEEEtran and xeCJK is required.\n' >&2
  exit 1
fi
