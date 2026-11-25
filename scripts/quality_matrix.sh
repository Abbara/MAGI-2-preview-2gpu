#!/usr/bin/env bash
# Quality experiment queue for MAGI-2 on 2x RTX PRO 6000. Runs sequentially;
# each run gets its own folder under output/quality/. Logs in output/quality/*.log.
#
#   A  structured fox prompt, seed 42, 540p  (preview cached -> reused by C/D)
#   C  refiner steps 5 / 8 / 12 from A's cached preview (1080p, turbo decoder)
#   D  refiner 5 steps from A's preview with the full Wan2.2 decoder
#   B  seeds 43, 44 (540p)
#   E  guidance low (VG 3.5 / AG 5) and high (VG 7 / AG 9), seed 42 (540p)
#
# Defaults for reference: 100 preview steps, VG 5.0, AG 7.0, 5 refiner steps, turbo decoder.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=output/quality
mkdir -p "$OUT"
PROMPT=prompts/fox_structured.json
RUN=scripts/run_local_2gpu.sh

run() {  # run <name> <env...> -- <args...>
  local name="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  if [[ -s "$OUT/$name/sample_000.mp4" ]]; then echo "[skip] $name (exists)"; return 0; fi
  echo "[$(date +%T)] start $name"
  local t0=$SECONDS
  env "${envs[@]}" "$RUN" --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] done  $name rc=$rc in $(( (SECONDS - t0) / 60 )) min"
}

# A: baseline structured prompt, cache the preview for C/D.
run A_struct_s42 MAGI2_PREVIEW_CACHE_DIR=$OUT/cache_s42 -- \
  --resolution 540p --seed 42 --prompt-file "$PROMPT"

CACHE=$OUT/cache_s42/preview_000.pt
if [[ -s "$CACHE" ]]; then
  for steps in 5 8 12; do
    run C_refiner_${steps}steps MAGI2_PREVIEW_CACHE_LOAD=$CACHE -- \
      --resolution 1080p --seed 42 --prompt-file "$PROMPT" --refiner-num-inference-steps "$steps"
  done
  run D_refiner_5steps_fullvae MAGI2_PREVIEW_CACHE_LOAD=$CACHE MAGI2_TURBO_VAE=0 -- \
    --resolution 1080p --seed 42 --prompt-file "$PROMPT" --refiner-num-inference-steps 5
else
  echo "[warn] no preview cache at $CACHE; skipping C/D"
fi

for seed in 43 44; do
  run B_struct_s${seed} MAGI2_PREVIEW_CACHE_DIR=$OUT/cache_s${seed} -- \
    --resolution 540p --seed "$seed" --prompt-file "$PROMPT"
done

run E_guid_low  MAGI2_VG=3.5 MAGI2_AG=5 -- --resolution 540p --seed 42 --prompt-file "$PROMPT"
run E_guid_high MAGI2_VG=7   MAGI2_AG=9 -- --resolution 540p --seed 42 --prompt-file "$PROMPT"
echo "[$(date +%T)] all done"
