#!/usr/bin/env bash
# MOPD v2 re-pin pipeline (issue #11) — GPU-serialized, per REPRODUCE.md.
# binding domain = CANONICAL['binding_v2'] (uniform last-write sampler); recall unchanged.
# Stages: teachers -> mopd(pg) -> mopd(kl) -> evaluate (single seed) -> seeds 0 1 2 (headline).
# Each stage writes its <name>.md table next to the script (crash-safe); we snapshot those
# plus stage logs under results/mopd_v2/.
set -uo pipefail
cd /home/ianbarber/Projects/factworld
PY=.venv-train/bin/python
OUT=results/mopd_v2
MOPD=experiments/mopd

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_stage() {
  local name="$1"; shift
  echo "[$(stamp)] START $name: $*" >> "$OUT/pipeline.log"
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "[$(stamp)] DONE  $name" >> "$OUT/pipeline.log"
  else
    echo "[$(stamp)] FAIL  $name (exit $?) — aborting pipeline" >> "$OUT/pipeline.log"
    exit 1
  fi
}

echo "[$(stamp)] pipeline start (pid $$)" >> "$OUT/pipeline.log"

run_stage stage2_teachers "$PY" "$MOPD/hf_teachers.py" --steps 300
cp "$MOPD/hf_teachers.md" "$OUT/hf_teachers.md"

run_stage stage3_mopd_pg "$PY" "$MOPD/hf_mopd.py" --loss pg
run_stage stage3_mopd_kl "$PY" "$MOPD/hf_mopd.py" --loss kl
cp "$MOPD/hf_mopd.md" "$OUT/hf_mopd.md"

run_stage stage4_evaluate "$PY" "$MOPD/hf_evaluate.py"
cp "$MOPD/hf_evaluate.md" "$OUT/hf_evaluate.md"

run_stage stage5_seeds "$PY" "$MOPD/hf_seeds.py" --seeds 0 1 2
cp "$MOPD/hf_seeds.md" "$OUT/hf_seeds.md"

echo "[$(stamp)] pipeline COMPLETE" >> "$OUT/pipeline.log"
