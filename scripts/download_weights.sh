#!/usr/bin/env bash
# Download the MAGI-2-preview weights (~307 GB) from Hugging Face at a pinned
# revision, then verify them. Safe to re-run: completed files are skipped and
# interrupted downloads resume.
#
# Usage:
#   scripts/download_weights.sh [--dest DIR] [--only comp1,comp2] [--sha256]
#
# Components (sizes at the pinned revision):
#   preview               228.1 GB  MAGI-2 preview DiT (114B MoE), 56 shards
#   text_encoder           55.6 GB  Qwen3.5-27B text encoder, 11 shards
#   refiner                13.5 GB  MAGI-2 refiner (540p -> 1080p), 3 shards
#   stable-audio-open-1.0   4.9 GB  audio VAE (decodes the generated audio)
#   vae                     2.8 GB  Wan2.2 video VAE
#   turbo_vae               1.9 GB  distilled fast video decoder
#
# DIR defaults to the sibling folder ../ckpt (next to this repo), which is where
# scripts/run_local_2gpu.sh looks unless MAGI2_CKPT_ROOT_HOST is set.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${MAGI2_CKPT_ROOT_HOST:-$(dirname "$REPO")/ckpt}"
REVISION="${MAGI2_HF_REVISION:-2dea51b64db47ee5b4402d36fd90829a0c58913b}"
ONLY="preview,refiner,text_encoder,vae,turbo_vae,stable-audio-open-1.0"
SHA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --sha256) SHA=(--sha256); shift ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI ('hf') is required:  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "$DEST"
avail_gb=$(df -P -BG "$DEST" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Destination: $DEST  (free: ${avail_gb} GB, full set needs ~307 GB)"
if [[ "$ONLY" == *preview* && "$avail_gb" -lt 310 ]]; then
  echo "WARNING: less than 310 GB free; the download may not fit." >&2
fi

includes=()
IFS=',' read -ra comps <<< "$ONLY"
for c in "${comps[@]}"; do includes+=(--include "$c/*"); done

echo "Downloading sand-ai/MAGI-2-preview@${REVISION:0:10}: ${ONLY}"
hf download sand-ai/MAGI-2-preview --revision "$REVISION" --local-dir "$DEST" "${includes[@]}"

python3 "$REPO/scripts/verify_weights.py" --ckpt "$DEST" --revision "$REVISION" --only "$ONLY" "${SHA[@]}"
