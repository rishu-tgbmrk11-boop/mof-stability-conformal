#!/bin/bash
# Regenerates every figure PDF (double-column figN_*.pdf and single-column figN_*_1col.pdf) and figures/captions.md from saved results.
# Run from the repo root: bash figures/make_all.sh   (env var PY selects the interpreter; fig1 and fig6 need data/processed/phase5/ from scripts/p5_03 and p5_05)
set -e
PY=${PY:-python}
for f in figures/fig*.py; do $PY -I "$f"; done
$PY -I figures/write_captions.py > /dev/null
echo "wrote figures/captions.md"
