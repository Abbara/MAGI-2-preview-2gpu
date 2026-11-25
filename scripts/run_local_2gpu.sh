#!/usr/bin/env bash
# Run MAGI-2-preview on 2x RTX PRO 6000 Blackwell (96GB, sm_120) inside the
# official sandai/magi-2-preview image, with local changes:
#   - FA2 attention instead of Hopper-only FA3
#   - FP8 expert weights (MAGI2_MOE_FP8=1), streamed straight to GPU
#   - cp=2 / ep=2 configs under configs/local/
#   - NCCL_P2P_DISABLE=1: PCIe P2P between the two cards hangs NCCL on this
#     host (IOMMU active, no iommu=pt); host-memory transport gives ~28 GB/s.
#
# Usage:
#   scripts/run_local_2gpu.sh [--resolution 540p|1080p] [entry.py args...]
# Output goes to output/runs/<timestamp>/ unless --output <repo-relative dir> is given.
# Quality knobs (env): MAGI2_VG / MAGI2_AG (video/audio guidance), MAGI2_TURBO_VAE=0
# (full Wan2.2 decoder), MAGI2_PREVIEW_CACHE_DIR=<dir> (save 540p previews),
# MAGI2_PREVIEW_CACHE_LOAD=<file> (refiner-only run from a saved preview).
# Paths are relative to the repo root.
# Smoke test without the preview/refiner/text-encoder weights:
#   SKIP_LOAD_MODEL=1 MAGI2_FAKE_TEXT_ENCODER=1 scripts/run_local_2gpu.sh --resolution 540p --num-inference-steps 2
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${MAGI2_CKPT_ROOT_HOST:-$(dirname "$REPO")/ckpt}"
CACHE="${MAGI2_CACHE_HOST:-$(dirname "$REPO")/cache}"
IMAGE="${MAGI2_IMAGE:-sandai/magi-2-preview:latest}"
mkdir -p "$CACHE/home" "$CACHE/magi_compiler" "$CACHE/magi_attention" "$REPO/output"

RESOLUTION=540p
OUTPUT="output/runs/$(date +%Y%m%d_%H%M%S)"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$RESOLUTION" in
  1080p) CONFIG=configs/local/2gpu_refiner.json
         # Free preview experts while the refiner runs; re-stream per clip.
         : "${MAGI2_PREVIEW_RELEASE_EXPERTS:=1}" ;;
  *)     CONFIG=configs/local/2gpu_preview.json ;;
esac
[[ " ${ARGS[*]-} " == *" --prompt"* || " ${ARGS[*]-} " == *" --samples"* ]] \
  || ARGS+=(--prompt "A red fox trots through fresh snow at dawn, breath steaming, soft crunching footsteps.")

mkdir -p "$REPO/$OUTPUT"
echo "[run_local_2gpu] output -> $REPO/$OUTPUT" >&2
exec docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v "$REPO":/workspace/magi2 -v "$CKPT":/ckpt:ro \
  -v "$CACHE/home":/tmp/home -v "$CACHE/magi_compiler":/tmp/magi_compiler_cache \
  -v "$CACHE/magi_attention":/tmp/magi_attention \
  -w /workspace/magi2 \
  -e HOME=/tmp/home -e MAGI2_CKPT_ROOT=/ckpt -e PYTHONPATH=/workspace/magi2 \
  -e MAGI2_MOE_FP8="${MAGI2_MOE_FP8:-1}" \
  -e MAGI2_PREVIEW_OFFLOAD_MODE="${MAGI2_PREVIEW_OFFLOAD_MODE:-gpu}" \
  -e MAGI2_TEXT_ENC_OFFLOAD_MODE="${MAGI2_TEXT_ENC_OFFLOAD_MODE:-cpu}" \
  -e MAGI2_REFINER_OFFLOAD_MODE="${MAGI2_REFINER_OFFLOAD_MODE:-roundtrip}" \
  -e MAGI2_VAE_OFFLOAD_MODE="${MAGI2_VAE_OFFLOAD_MODE:-gpu}" \
  -e MAGI2_PREVIEW_RELEASE_EXPERTS="${MAGI2_PREVIEW_RELEASE_EXPERTS:-0}" \
  -e SKIP_LOAD_MODEL="${SKIP_LOAD_MODEL:-0}" \
  -e MAGI2_FAKE_TEXT_ENCODER="${MAGI2_FAKE_TEXT_ENCODER:-0}" \
  -e MAGI2_ATTN_BACKEND="${MAGI2_ATTN_BACKEND:-}" \
  -e MAGI2_REFINER_ATTN="${MAGI2_REFINER_ATTN:-}" \
  -e MAGI2_DUMP_ATTN_RANGES="${MAGI2_DUMP_ATTN_RANGES:-}" \
  -e MAGI2_VG="${MAGI2_VG:-}" -e MAGI2_AG="${MAGI2_AG:-}" \
  -e MAGI2_TURBO_VAE="${MAGI2_TURBO_VAE:-}" \
  -e MAGI2_PREVIEW_CACHE_DIR="${MAGI2_PREVIEW_CACHE_DIR:-}" \
  -e MAGI2_PREVIEW_CACHE_LOAD="${MAGI2_PREVIEW_CACHE_LOAD:-}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e NCCL_ALGO='^NVLS' \
  -e NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  "$IMAGE" \
  torchrun --nnodes=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 \
    inference/pipeline/entry.py --config "$CONFIG" --resolution "$RESOLUTION" \
    --seconds 10 --output "$OUTPUT" "${ARGS[@]}"
