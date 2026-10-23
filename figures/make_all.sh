#!/bin/bash
# Regenerates every figure PDF from saved results. Run from the repo root: bash figures/make_all.sh
set -e
PY=${PY:-python}
for f in figures/fig*.py; do $PY -I "$f"; done
