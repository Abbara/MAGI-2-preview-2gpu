#!/usr/bin/env bash
# Capability evaluation: 15 clips at 1080p (13 text-to-video from prompts/eval/, 2 image-to-video
# from the upstream demo assets), full Wan2.2 decoder, default 100 draft + 5 refiner steps.
# One process per clip so a failure doesn't lose the rest; finished clips are skipped on re-run.
# Each clip's 540p draft is saved to output/eval/cache/<name>/ for later re-refining.
# ~42 min per clip, ~10.5 h total.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=output/eval
mkdir -p "$OUT"

run() {  # run <name> <entry.py args...>
  local name="$1"; shift
  if [[ -s "$OUT/$name/sample_000.mp4" ]]; then echo "[skip] $name (exists)"; return 0; fi
  echo "[$(date +%T)] start $name"
  local t0=$SECONDS
  MAGI2_TURBO_VAE=0 MAGI2_PREVIEW_CACHE_DIR="$OUT/cache/$name" \
    scripts/run_local_2gpu.sh --resolution 1080p --seed 42 --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  echo "[$(date +%T)] done  $name rc=$? in $(( (SECONDS - t0) / 60 )) min"
}

for f in prompts/eval/[0-9][0-9]_*.json; do
  run "$(basename "$f" .json)" --prompt-file "$f"
done
run 14_i2v_anime_forest --image assets/sample_000.jpeg --prompt-file assets/sample_000.txt
run 15_i2v_clay_room    --image assets/sample_001.jpeg --prompt-file assets/sample_001.txt
echo "[$(date +%T)] all done"
