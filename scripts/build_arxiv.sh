#!/bin/bash
# Build reports/factworld.pdf from reports/factworld.tex (tectonic; self-contained LaTeX).
set -eu
cd "$(dirname "$0")/../reports"
cp -f ../docs/benchmark/fig_bench_headline.png .
tectonic factworld.tex
rm -f fig_bench_headline.png
echo "wrote reports/factworld.pdf"
