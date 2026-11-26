#!/usr/bin/env bash
# Preflight check: can this machine run MAGI-2-preview with this repo?
# Run it before downloading the ~307 GB of weights. Nothing is modified.
#
# Usage:
#   scripts/check_system.sh            # host checks (seconds)
#   scripts/check_system.sh --gpu      # + GPU-to-GPU (NCCL) test inside the Docker image (~1 min)
#
# Env: MAGI2_CKPT_ROOT_HOST (weights dir, default ../ckpt), MAGI2_IMAGE (Docker image).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${MAGI2_CKPT_ROOT_HOST:-$(dirname "$REPO")/ckpt}"
source "$REPO/scripts/image.env"
IMAGE="${MAGI2_IMAGE:-$MAGI2_DEFAULT_IMAGE}"
GPU_TEST=0
[[ "${1:-}" == "--gpu" ]] && GPU_TEST=1

fails=0; warns=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fails=$((fails + 1)); }

echo "== GPUs"
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi not found: install the NVIDIA driver"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits)
  if (( ${#gpus[@]} < 2 )); then
    fail "${#gpus[@]} GPU(s) found; 2 are required"
  else
    ok "${#gpus[@]} GPUs found (the first 2 are used)"
  fi
  for g in "${gpus[@]:0:2}"; do
    IFS=',' read -r idx name mem cc <<< "$g"
    name="${name# }"; mem="${mem// /}"; cc="${cc// /}"
    if (( mem < 94000 )); then
      fail "GPU$idx $name: $((mem / 1024)) GiB; at least 96 GB per GPU is needed (peak ~88 GB)"
    else
      ok "GPU$idx $name: $((mem / 1024)) GiB, compute capability $cc"
    fi
    case "$cc" in
      12.0) ;;
      9.0)  warn "GPU$idx is Hopper (sm_90): should take upstream's FA3 path, but untested with this repo" ;;
      10.*) warn "GPU$idx is sm_${cc/./}: Blackwell datacenter, untested with this repo" ;;
      *)    warn "GPU$idx compute capability $cc is untested" ;;
    esac
  done
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -2 | awk '$1 > 2000' | wc -l)
  (( busy > 0 )) && warn "$busy GPU(s) already have >2 GB in use (another process?); a run needs nearly all of it"
fi

echo "== System memory"
mem_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if (( mem_gb < 110 )); then
  fail "${mem_gb} GiB RAM; ~128 GB is needed (text encoder runs on the CPU, ~56 GB)"
else
  ok "${mem_gb} GiB RAM"
fi
(( avail_gb < 80 )) && warn "only ${avail_gb} GiB RAM available right now; close other big processes before running"

echo "== Disk"
d="$CKPT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
disk_gb=$(df -P -BG "$d" | awk 'NR==2 {gsub("G","",$4); print $4}')
if [[ -d "$CKPT/preview" ]]; then
  ok "weights folder exists at $CKPT (verify with: python3 scripts/verify_weights.py --ckpt $CKPT)"
elif (( disk_gb < 310 )); then
  fail "${disk_gb} GB free at $d; the weights need ~307 GB (or set MAGI2_CKPT_ROOT_HOST to another disk)"
else
  ok "${disk_gb} GB free at $d for the weights"
fi
docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
if [[ -n "$docker_root" && -d "$docker_root" ]]; then
  dgb=$(df -P -BG "$docker_root" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
  [[ -n "$dgb" ]] && (( dgb < 50 )) && warn "${dgb} GB free for Docker at $docker_root; the image needs ~42 GB unpacked"
fi

echo "== Docker"
if ! command -v docker >/dev/null; then
  fail "docker not found: install Docker Engine"
elif ! docker info >/dev/null 2>&1; then
  fail "cannot talk to the Docker daemon as $(id -un): start it, or add yourself to the 'docker' group"
else
  ok "Docker $(docker version -f '{{.Server.Version}}' 2>/dev/null) reachable"
  if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || command -v nvidia-ctk >/dev/null; then
    ok "NVIDIA Container Toolkit present"
  else
    fail "NVIDIA Container Toolkit not found: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
  fi
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "image $IMAGE is pulled"
  else
    warn "image not pulled yet:  docker pull $IMAGE   (~13 GB download)"
  fi
fi

echo "== Python / Hugging Face CLI (for the weight download)"
command -v python3 >/dev/null && ok "python3 $(python3 -c 'import sys; print(sys.version.split()[0])')" \
  || fail "python3 not found"
command -v hf >/dev/null && ok "hf CLI found" \
  || warn "hf CLI not found:  pip install -U 'huggingface_hub[cli]'"

echo "== PCIe peer-to-peer"
if [[ -d /sys/kernel/iommu_groups ]] && (( $(ls /sys/kernel/iommu_groups | wc -l) > 0 )) \
   && ! grep -qw 'iommu=pt' /proc/cmdline; then
  ok "IOMMU active without iommu=pt: GPU P2P likely hangs; the launcher already disables it (NCCL_P2P_DISABLE=1)"
else
  ok "no IOMMU translation in the way; run with --gpu to see if P2P works (then NCCL_P2P_DISABLE=0 is faster)"
fi

if (( GPU_TEST )); then
  echo "== GPU-to-GPU communication test (inside $IMAGE)"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    fail "pull the image first: docker pull $IMAGE"
  else
    test_py='
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
x = torch.ones(256 << 20, device="cuda")  # 1 GiB
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 10
if r == 0: print(f"RESULT {x.numel() * 4 / dt / 1e9:.1f} GB/s algbw")
dist.destroy_process_group()'
    for p2p in 1 0; do
      label=$([[ $p2p == 1 ]] && echo "P2P disabled (launcher default)" || echo "P2P enabled")
      cname="magi2-nccl-check-$$-$p2p"
      out=$(timeout 120 docker run --rm --name "$cname" --gpus all --ipc=host -e NCCL_P2P_DISABLE=$p2p -e PYTHONUNBUFFERED=1 \
        "$IMAGE" bash -c "cat > /tmp/t.py <<'EOF'
$test_py
EOF
torchrun --nproc_per_node=2 /tmp/t.py" 2>&1)
      rc=$?
      res=$(grep -o 'RESULT.*' <<< "$out" | sed 's/RESULT //')
      if [[ -n "$res" ]]; then
        ok "$label: $res"
      elif (( rc == 124 )); then
        if (( p2p == 1 )); then fail "$label: hung (timed out after 120 s)"
        else ok "$label: hangs (expected on IOMMU hosts; keep the default NCCL_P2P_DISABLE=1)"; fi
        docker kill "$cname" >/dev/null 2>&1
      else
        if (( p2p == 1 )); then fail "$label: failed (rc=$rc):"; else warn "$label: failed (rc=$rc):"; fi
        tail -5 <<< "$out" | sed 's/^/          /'
      fi
    done
  fi
fi

echo
if (( fails )); then
  echo "$fails problem(s), $warns warning(s): fix the FAIL items before downloading the weights."
  exit 1
fi
echo "Ready ($warns warning(s)). Next: scripts/download_weights.sh"
