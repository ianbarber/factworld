#!/bin/bash
# Build reports/arxiv/factworld.pdf from factworld.tex (tectonic; self-contained LaTeX).
set -eu
cd "$(dirname "$0")/../reports/arxiv"
cp -f ../../docs/benchmark/fig_bench_headline.png .
tectonic factworld.tex
echo "wrote reports/arxiv/factworld.pdf"
