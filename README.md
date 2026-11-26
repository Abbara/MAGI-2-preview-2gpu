# MAGI-2-preview on 2× RTX PRO 6000 Blackwell

A port of [Sand AI's MAGI-2-preview](https://github.com/SandAI-org/MAGI-2-preview) that runs on a
**single workstation with two 96 GB Blackwell GPUs** instead of the reference 8× H100/H200.
It generates 10-second clips with synchronized sound, from text or from text + image, at
896×512 (540p) or 1920×1088 (1080p).

MAGI-2-preview is a 114B-parameter mixture-of-experts audio-video model (6B active per token).
The upstream code needs FA3 and MagiAttention kernels that exist only for Hopper (sm_90), and it
assumes 8 GPUs with 80 GB each plus enough host RAM to park whole models.
This fork changes the parts that block other hardware and leaves the model and sampling the same.

The original upstream README (model description, prompt-enhancement details) is kept at
[`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).

---

## Contents

- [What's different from upstream](#whats-different-from-upstream)
- [Requirements](#requirements)
- [Setup](#setup)
  - [1. Clone](#1-clone)
  - [2. Check the machine](#2-check-the-machine)
  - [3. Docker image](#3-docker-image)
  - [4. Model weights (~307 GB)](#4-model-weights-307-gb)
  - [5. Smoke test (no weights needed)](#5-smoke-test-no-weights-needed)
- [Generating videos](#generating-videos)
  - [Text-to-video](#text-to-video)
  - [Image + text to video](#image--text-to-video)
  - [Prompts: why structure matters](#prompts-why-structure-matters)
  - [Draft at 540p, finish at 1080p](#draft-at-540p-finish-at-1080p)
  - [Options](#options)
- [Quality settings: what we measured](#quality-settings-what-we-measured)
- [Performance](#performance)
- [How it fits in 2× 96 GB](#how-it-fits-in-2-96-gb)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Licenses and credits](#licenses-and-credits)

---

## What's different from upstream

| Area | Upstream | This fork | Where |
|---|---|---|---|
| Main-model attention | FA3 (Hopper only) | FA2 with the same sink correction on non-Hopper GPUs; FA3 is still used on Hopper. Matches a reference to 0.0017 max abs error | `inference/model/magi2_preview.py` |
| Refiner attention | MagiAttention FFA (sm_90 only). Its fallback made ~390k FA2 calls per layer at 1080p, so one step took over 40 min | Block-sparse PyTorch `flex_attention`, with the mask built once per clip from the same (q, k) range pairs. Exact (0.0004 max abs error); 0.66 s per attention call | `inference/model/flex_block_attn.py`, `magi2_refiner.py` |
| Hopper detection | `major >= 9`, which sent Blackwell to sm_90-only kernels | `major == 9` | `magi2_refiner.py` |
| MoE expert weights | bf16, ~217 GB in total | **FP8 (e4m3)** with a per-(expert, output-channel) scale, converted back to bf16 one layer at a time. Expert memory is halved | `magi2_preview.py` (`MAGI2_MOE_FP8`) |
| Weight loading | Build on CPU, then load the whole state dict into RAM | Build on the GPU and stream tensor by tensor from disk, quantizing on the GPU. Loads 228 GB in ~35 s and never needs it in RAM | `inference/infra/checkpoint/` |
| MoE Triton tiles | Need 120 KB shared memory per block | Smaller tiles for sm_120 (99 KB limit), ~195 TFLOP/s | `inference/flash_mh_moe/triton/mh_moe_fwd.py` |
| 1080p memory | Park models in host RAM between stages | Free the preview's expert weights (~51 GB/GPU) while the refiner runs, then re-stream them for the next clip | `inference_engine.py` (`MAGI2_PREVIEW_RELEASE_EXPERTS`) |
| Parallelism | cp = ep = 8 | cp = ep = 2 | `configs/local/` |
| Video decoder | Turbo (distilled) decoder only; the full Wan2.2 decode path was removed | Full Wan2.2 decode restored; default at 1080p (sharper, see [measurements](#quality-settings-what-we-measured)), turbo at 540p | `inference/model/vae2_2.py` (`MAGI2_TURBO_VAE`) |
| Workflow | – | Save 540p previews and finish them at 1080p later; guidance overrides; a fake text encoder for smoke tests; one launcher script | `scripts/`, `inference_engine.py`, `entry.py` |

The sampler, schedulers, configs (apart from the parallel sizes) and default hyper-parameters are the same as upstream.

---

## Requirements

**Tested on:** 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, sm_120), AMD Threadripper
PRO 9955WX (16 cores), 122 GB RAM, NVMe SSD, Ubuntu with driver 595.99, Docker + NVIDIA Container Toolkit.

| Resource | Needed | Why |
|---|---|---|
| GPUs | 2 × ≥ 96 GB | ~65 GB of weights per GPU after FP8, plus activations (peak ~88 GB) |
| GPU architecture | Blackwell (sm_100/sm_120) or Hopper | Other architectures are untested (FA2 and flex_attention exist for sm_80+, but the MoE tile sizes are tuned for these) |
| System RAM | ~90 GB free (≥ 128 GB total recommended) | The 27B text encoder runs **on the CPU** (~56 GB), plus refiner CPU copies (2 × 14 GB) |
| CPU | Many cores help | Text encoding runs on the CPU once per prompt |
| Disk | ~307 GB for weights + ~42 GB for the Docker image (12.7 GB download) | Fast NVMe recommended: experts are re-streamed between 1080p clips |
| Software | Linux, Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), NVIDIA driver with CUDA 13 support (≥ 580), Python 3 on the host, `pip install -U "huggingface_hub[cli]"` | Everything else is inside the upstream Docker image |

Close other GPU-heavy apps first. For example, ComfyUI keeps unloaded models in system RAM, ~60 GB in our case.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Abbara/MAGI-2-preview-2gpu MAGI-2-Preview/code
cd MAGI-2-Preview/code
```

By default the scripts expect this layout, with the weights **next to** the repo rather than inside it:

```
MAGI-2-Preview/
├── code/     ← this repository
├── ckpt/     ← model weights (scripts/download_weights.sh)
└── cache/    ← compile caches + logs (created automatically)
```

To put the weights elsewhere, set `MAGI2_CKPT_ROOT_HOST=/path/to/ckpt` for both the download and the run scripts.

### 2. Check the machine

Before downloading anything big, run the preflight check. It changes nothing; it reports the GPUs
(count, memory, architecture), RAM, free disk, Docker and the NVIDIA Container Toolkit, and whether
the IOMMU is likely to break GPU-to-GPU transfers:

```bash
scripts/check_system.sh          # a few seconds
scripts/check_system.sh --gpu    # after pulling the image: also tests GPU-to-GPU (NCCL) transfers, ~1 min
```

Fix every `FAIL` line before going on. `WARN` lines are advisory.

### 3. Docker image

All Python dependencies (PyTorch 2.9 / CUDA 13, FA2, Triton, MagiCompiler, ffmpeg, …) come from the
upstream image. The code is **not** baked into it; the launcher mounts this repo at run time.

```bash
docker pull sandai/magi-2-preview@sha256:a81785240796338b9552bade5e62721aabace4a80dd1b30a9bdaa2651821b020
# 12.7 GB download, 41.7 GB on disk
```

That digest (build `2712106`, 2026-08-05) is the build this fork was tested with, and the launcher
uses it by default (set in `scripts/image.env`). To try a newer upstream build instead:
`docker pull sandai/magi-2-preview:latest` and run with `MAGI2_IMAGE=sandai/magi-2-preview:latest`.

### 4. Model weights (~307 GB)

Weights are **not** stored in this repository. They come from the Hugging Face repo
[`sand-ai/MAGI-2-preview`](https://huggingface.co/sand-ai/MAGI-2-preview), pinned to revision
`2dea51b64db47ee5b4402d36fd90829a0c58913b` (the version this fork was tested with):

```bash
pip install -U "huggingface_hub[cli]"
scripts/download_weights.sh                 # all components into ../ckpt, then verify
```

| Component | Size | What it is | Original source |
|---|---|---|---|
| `preview/` | 228.1 GB (56 shards) | MAGI-2 preview DiT, 114B MoE: generates video + audio at 540p | Sand AI |
| `text_encoder/` | 55.6 GB (11 shards) | Qwen3.5-27B, used as the text encoder | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |
| `refiner/` | 13.5 GB (3 shards) | MAGI-2 refiner: 540p → 1080p, 12.5 → 25 fps | Sand AI |
| `stable-audio-open-1.0/` | 4.9 GB | Audio VAE that decodes the generated audio latents | Stability AI, Stable Audio Open 1.0 |
| `vae/` | 2.8 GB | Wan2.2 video VAE (encoder for image input; full decoder, default at 1080p) | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| `turbo_vae/` | 1.9 GB | Distilled fast video decoder (default at 540p) | Sand AI |

All six are mirrored in the single `sand-ai/MAGI-2-preview` repo, so one download covers everything.

**Useful variants**

```bash
scripts/download_weights.sh --only vae,turbo_vae,stable-audio-open-1.0   # ~10 GB: enough for the smoke test
scripts/download_weights.sh --dest /data/magi2-ckpt                      # custom location (then export MAGI2_CKPT_ROOT_HOST)
scripts/download_weights.sh --sha256                                     # also SHA-256-verify every file (slow)
python3 scripts/verify_weights.py --ckpt ../ckpt                         # verify an existing download only
```

The download resumes when re-run. Verification checks every file against the Hub listing for the
pinned revision: presence, exact size, and an intact safetensors layout (with `--sha256`, a full hash).

**Doing it by hand** (equivalent):

```bash
hf download sand-ai/MAGI-2-preview \
  --revision 2dea51b64db47ee5b4402d36fd90829a0c58913b \
  --local-dir ../ckpt \
  --include "preview/*" "refiner/*" "text_encoder/*" "vae/*" "turbo_vae/*" "stable-audio-open-1.0/*"
```

On a fast connection the full set took about 25 minutes. For faster downloads, `pip install hf_transfer`
and `export HF_HUB_ENABLE_HF_TRANSFER=1`. If the Hub ever requires it, log in with `hf auth login` or set `HF_TOKEN`.

### 5. Smoke test (no weights needed)

Before or during the big download, check that the GPUs, kernels and multi-GPU setup work. This uses
synthetic preview/refiner weights and a fake text encoder, and only needs the ~10 GB set above:

```bash
SKIP_LOAD_MODEL=1 MAGI2_FAKE_TEXT_ENCODER=1 \
  scripts/run_local_2gpu.sh --resolution 540p --num-inference-steps 2
```

It should finish in about 5 minutes and write a noise video, with sound, to `output/runs/<timestamp>/sample_000.mp4`.
If it hangs at the first step, see [NCCL hangs](#nccl-hangs-on-the-first-collective-one-gpu-at-100-the-other-idle).

---

## Generating videos

Everything goes through `scripts/run_local_2gpu.sh`, which starts the Docker image on both GPUs.
Each run writes to `output/runs/<timestamp>/` (or `--output <dir>`), one `sample_NNN.mp4` per prompt.
**Paths you pass (prompt files, images, output dirs) must be inside the repo**, because only the repo is
mounted in the container. Give them relative to the repo root.

### Text-to-video

```bash
# 540p (896×512, 12.5 fps): ~30 min
scripts/run_local_2gpu.sh --resolution 540p --prompt-file prompts/fox_structured.json

# 1080p (1920×1088, 25 fps): ~40-45 min
scripts/run_local_2gpu.sh --resolution 1080p --prompt-file prompts/fox_structured.json

# Several prompts in one run (the model loads once)
scripts/run_local_2gpu.sh --resolution 540p --prompt "first prompt" "second prompt"

# Different seed (the default is 42; consecutive prompts use seed, seed+1, …)
scripts/run_local_2gpu.sh --resolution 540p --seed 7 --prompt-file prompts/fox_structured.json
```

Clips are always 10 seconds long; that's the only duration the model supports.

### Image + text to video

Put the image somewhere inside the repo (e.g. `inputs/`) and refer to it as `<Figure 1>` in the prompt.
The tag is added automatically if missing:

```bash
mkdir -p inputs && cp ~/photo.jpg inputs/
scripts/run_local_2gpu.sh --resolution 1080p \
  --image inputs/photo.jpg \
  --prompt "The woman in <Figure 1> turns toward the camera and smiles."
```

For a batch of mixed T2V/I2V jobs, use a samples file like [`assets/demo_samples.json`](assets/demo_samples.json):
`scripts/run_local_2gpu.sh --resolution 1080p --samples assets/demo_samples.json`.

> **Audio input is not supported.** The model generates its own soundtrack (speech, effects,
> ambience). This release can't lip-sync to a recording you provide.

### Prompts: why structure matters

**This is the single biggest lever on quality, and on audio in particular.** MAGI-2 was trained on
long, structured captions that describe the scene, subjects, camera, a timeline of actions and a
timeline of sounds. Short prompts give decent video but under-use the model. In our tests, a
one-sentence prompt produced a good-looking clip with an **almost silent** soundtrack (−71 dB),
while a structured caption produced footsteps, wind and breathing at the right moments (−41 dB mean).

You can write prompts in that structure in three ways:

1. **Write the JSON by hand.** Copy [`prompts/fox_structured.json`](prompts/fox_structured.json) or
   [`assets/sample_enhanced_t2v.json`](assets/sample_enhanced_t2v.json) and edit it. Pass it with
   `--prompt-file`; JSON is rendered to the model's markdown format automatically. The important audio
   fields are `audio_baseline.ambience`, per-segment `audio.special_audio_events` (timestamped
   sounds with `visual_sync`) and `audio.dialogue_lines` (speaker, text, delivery).
2. **Let an LLM do it (prompt enhancement).** Set `API_KEY` (and optionally `DEFAULT_MODEL`,
   `DEFAULT_BASE_URL`) in [`inference/prompt_enhancement/enhancer.py`](inference/prompt_enhancement/enhancer.py).
   Any OpenAI-compatible endpoint works; the default points at Google's Gemini endpoint. Short
   prompts are then expanded automatically using the templates in
   `inference/prompt_enhancement/prompts/`. The container needs network access for this.
3. **Ask any capable chat model** to fill in the template from your idea, using
   [`inference/prompt_enhancement/prompts/t2v.md`](inference/prompt_enhancement/prompts/t2v.md)
   (or `i2v.md`) as the instructions.

### Draft at 540p, finish at 1080p

The 1080p refiner adds detail but doesn't change content, so a cheap workflow is: generate a few
540p drafts (different seeds or prompt variants), pick the best, and refine only that one. Previews
can be saved and refined later without regenerating them:

```bash
# 1) drafts: each saves output/previews/preview_NNN.pt (video+audio latents and text features)
MAGI2_PREVIEW_CACHE_DIR=output/previews \
  scripts/run_local_2gpu.sh --resolution 540p --seed 1 --prompt-file prompts/fox_structured.json

# 2) refine the one you like (skips text encoding and the 540p stage)
MAGI2_PREVIEW_CACHE_LOAD=output/previews/preview_000.pt \
  scripts/run_local_2gpu.sh --resolution 1080p
```

The audio track comes from the 540p stage, so a refined clip has the same sound as its draft.
Step 2 uses the cached text features, so it needs no prompt. It refines one cached preview per run.

### Options

Command-line options (passed through to `inference/pipeline/entry.py`):

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | `540p` | `272p`, `540p` (896×512) or `1080p` (1920×1088, uses the refiner) |
| `--prompt "…" ["…"]` / `--prompt-file f [f]` / `--samples f.json` | – | What to generate (text, files with text or JSON, or a batch file) |
| `--image img [img]` | – | First-frame / reference image(s) for image-to-video |
| `--seed` | 42 | Random seed |
| `--num-inference-steps` | 100 | 540p denoising steps (~17.4 s each) |
| `--refiner-num-inference-steps` | 5 | 1080p refiner steps |
| `--output` | `output/runs/<timestamp>` | Output directory (relative to the repo) |

Environment variables read by the launcher:

| Variable | Default | Meaning |
|---|---|---|
| `MAGI2_VG`, `MAGI2_AG` | 5.0, 7.0 (from config) | Video / audio text-guidance scale |
| `MAGI2_TURBO_VAE` | 1 at 540p, 0 at 1080p | `1` = distilled turbo decoder, `0` = full Wan2.2 VAE decoder |
| `MAGI2_PREVIEW_CACHE_DIR` / `MAGI2_PREVIEW_CACHE_LOAD` | – | Save 540p previews / refine a saved one (see above) |
| `MAGI2_CKPT_ROOT_HOST` | `../ckpt` | Host path of the weights |
| `MAGI2_IMAGE` | tested digest (`scripts/image.env`) | Docker image |
| `MAGI2_MOE_FP8` | 1 | FP8 expert storage. `0` (bf16) does **not** fit on 2 × 96 GB |
| `MAGI2_PREVIEW_RELEASE_EXPERTS` | 1 at 1080p | Free preview experts during the refiner |
| `MAGI2_TEXT_ENC_OFFLOAD_MODE` | `cpu` | Where the text encoder runs (`cpu` / `gpu` / `roundtrip`) |
| `NCCL_P2P_DISABLE` | 1 | See [NCCL hangs](#nccl-hangs-on-the-first-collective-one-gpu-at-100-the-other-idle) |
| `MAGI2_ATTN_BACKEND` | auto | `fa2` / `fa3`: force the main-model attention backend |
| `MAGI2_REFINER_ATTN` | `flex` | `loop` = upstream's per-range FA2 fallback (very slow; for debugging) |
| `MAGI2_MOE_BLOCK_CONFIG` | auto | Override MoE Triton tiles: `BLOCK_T,BLOCK_DH,BLOCK_DE,STAGES,WARPS` |
| `SKIP_LOAD_MODEL`, `MAGI2_FAKE_TEXT_ENCODER` | 0 | Synthetic weights / random text features for smoke tests |

---

## Quality settings: what we measured

All runs used `prompts/fox_structured.json` (`scripts/quality_matrix.sh` reproduces them; outputs go to
`output/quality/`). Sharpness is the mean variance of the Laplacian, one frame per second; audio level
is ffmpeg `volumedetect` mean volume.

**1080p refiner, all from the same saved 540p draft (seed 42):**

| Refiner steps | Decoder | Run time¹ | Sharpness | PSNR vs 5-step turbo |
|---|---|---|---|---|
| 5 | **full Wan2.2** | **12 min** | **43.7** | 46.1 dB |
| 5 | turbo | 15 min² | 38.3 | – |
| 8 | turbo | 16 min | 38.3 | 48.2 dB |
| 12 | turbo | 20 min | 39.0 | 46.4 dB |

¹ Refine-only runs from a cached draft (load, refine, decode). ² Includes a one-time decoder compile.

- **The full decoder is the one setting that adds visible detail** (about 14% sharper fur and snow texture)
  with no time cost, so it is now the 1080p default.
- **More refiner steps don't help.** PSNR above 46 dB means visually near-identical clips; each step
  costs about 50 s. Keep the default of 5.

**540p draft stage (100 steps, ~32 min each):**

| Run | Video / audio guidance | Audio mean / peak | Notes |
|---|---|---|---|
| Seed 42 | 5 / 7 (default) | −43.4 / −22.2 dB | baseline |
| Seed 43 | 5 / 7 | −40.4 / −18.3 dB | different framing, paw prints in the snow |
| Seed 44 | 5 / 7 | −41.5 / −22.8 dB | denser forest, taller sapling |
| Seed 42, low guidance | 3.5 / 5 | −56.0 / −36.0 dB | nearly silent |
| Seed 42, high guidance | 7 / 9 | −36.6 / −12.0 dB | loudest, clearest sound; harder contrast in the fur |

- **Guidance mostly changes the audio.** Low guidance nearly mutes it; raise `MAGI2_AG` if the sound is too quiet.
- **Seeds change the composition, not the quality.** Every seed played out all the prompt's beats.
  Draft a few seeds at 540p, then refine the best one ([workflow above](#draft-at-540p-finish-at-1080p)).

---

## Performance

Measured on the reference machine (2× RTX PRO 6000, 16-core Threadripper, NVMe), with both GPUs power-limited to 300 W (their default is 600 W), so expect shorter times at the stock limit:

| Stage | Time |
|---|---|
| Stream 228 GB of preview weights to both GPUs (FP8-quantizing on the fly) | ~35 s (weights partly in the page cache) |
| 540p generation, 100 steps | ~29 min (17.4 s/step; the first step includes a ~1-2 min compile) |
| 1080p refiner (5 steps, ~50 s/step) + decode | ~5 min |
| **One 1080p clip, end to end** | **~42 min** |
| One 540p clip, end to end | ~32 min |

| Memory (per GPU) | |
|---|---|
| After loading | 64.5 GB |
| Peak during 540p denoise | ~82 GB reserved |
| Refiner stage (preview experts released) | ~30 GB + activations, ~90 GB reserved peak |

---


### Capability sweep

`scripts/eval_batch.sh` renders 15 different 1080p clips (prompts in `prompts/eval/`, built by `make_prompts.py`): speech, two-person dialogue, Mandarin, singing, piano, a kickflip, liquids, fire, a rainy crowd, chalkboard text, a drone shot, cooking, a thunderstorm, and two image-to-video samples. Each clip takes 42–50 min at 300 W, about 11 h in total. Finished clips are skipped, so the script can be re-run after an interruption.

## How it fits in 2× 96 GB

- **Expert parallelism (ep = 2):** each GPU holds half of the 256 experts in every MoE layer (6 of 12
  expert heads). The loader reads only that half from disk.
- **FP8 experts:** expert weights are 217 GB of the 228 GB model. In FP8 with fp32 scales, that's
  ~54 GB per GPU instead of ~109 GB. Each layer is converted back to bf16 just before its MoE
  kernel (~7 ms per tensor). On random weights the MoE output differs from bf16 by ~4% relative
  error. In practice clips show no visible artifacts, but it is an approximation of the original model.
- **Context parallelism (cp = 2, Ulysses):** the token sequence is split across both GPUs for attention.
- **Text encoder on CPU:** Qwen3.5-27B runs once per prompt on the CPU, on rank 0 only.
- **1080p:** the preview and refiner don't fit on the GPU together, and the preview can't be parked
  in RAM (2 × 65 GB). So after the 540p stage the expert storage is freed (`untyped_storage().resize_(0)`,
  which keeps the tensors and compiled graphs valid), the refiner runs, and the experts are
  re-streamed from disk before the next clip.

---

## Troubleshooting

### NCCL hangs on the first collective (one GPU at "100%", the other idle)

Peer-to-peer PCIe transfers between the GPUs silently hang on some workstation platforms, usually
when the IOMMU is in translated mode. `nvidia-smi topo -p2p r` may still report `OK`. Symptom: after
10 minutes, `Watchdog caught collective operation timeout … OpType=BROADCAST`. The launcher sets
`NCCL_P2P_DISABLE=1` (transfers go through host memory, ~28-30 GB/s, enough for this workload).
To fix it at the platform level, add `iommu=pt` to the kernel command line (GRUB) and reboot, then
try `NCCL_P2P_DISABLE=0`. `scripts/check_system.sh --gpu` tests both modes (a hang with P2P enabled
is reported as expected on IOMMU hosts).

### Out of memory

- GPU: make sure nothing else is using the GPUs (`nvidia-smi`). Peak use is ~88 GB of 96 GB.
- GPU, at the very end of a 1080p clip (during video decoding): the full Wan2.2 decoder, the 1080p
  default, briefly reserves ~86 GB per GPU. On cards with less headroom (e.g. 94 GB), switch to the
  lighter turbo decoder with `MAGI2_TURBO_VAE=1`; it is slightly less sharp
  (see [measurements](#quality-settings-what-we-measured)).
- Host RAM: the text encoder needs ~56 GB on the CPU. Stop apps that cache models in RAM (e.g. ComfyUI).

### `OutOfResources: shared memory` in a Triton kernel

Your GPU has less shared memory per block than the tiles assume. Use smaller MoE tiles via
`MAGI2_MOE_BLOCK_CONFIG` (e.g. `64,64,32,2,4`).

### Silent or near-silent audio

Use a structured prompt (see [Prompts](#prompts-why-structure-matters)). This was the cause in every case we've seen.

### First run is slow to start

Kernels are compiled on first use and cached in `../cache/`. Later runs start faster.

### `AssertionError: expected size … stride …` from a compiled graph

A stale compiled graph from a run with different input shapes. The launcher keeps a separate compile
cache per resolution and mode (`../cache/magi_compiler/<key>/`), so this shouldn't happen any more.
If it does, delete `../cache/magi_compiler/` and rerun.

---

## Limitations

- 10-second clips only; fixed resolution presets (272p / 540p / 1080p).
- No audio input (no lip-sync to a provided voice track).
- FP8 experts are an approximation of the bf16 model.
- Tested only on 2× RTX PRO 6000 Blackwell (sm_120). Hopper should take the original FA3 path; other GPUs are untested.
- Prompt enhancement needs an external LLM API key.

---

## Repository layout

```
configs/local/2gpu_*.json          cp=2 / ep=2 overlays on the upstream configs
scripts/run_local_2gpu.sh          launcher (Docker, both GPUs, all env knobs)
scripts/image.env                  pinned Docker image digest used by the scripts
scripts/check_system.sh            preflight: GPUs, RAM, disk, Docker, NVIDIA toolkit, NCCL
scripts/download_weights.sh        fetch + verify weights (pinned HF revision)
scripts/verify_weights.py          verify an existing download (size, layout, optional SHA-256)
scripts/quality_matrix.sh          the prompt / seed / guidance / refiner / decoder comparison we ran
prompts/fox_structured.json        example structured T2V prompt
inference/model/flex_block_attn.py block-sparse refiner attention (flex_attention)
docs/UPSTREAM_README.md            original upstream README
```

Not in git (see `.gitignore`): weights (`*.safetensors`, `*.pt`, `*.ckpt`, `*.pth`, …), `output*/`,
generated media, logs and caches.

---

## Licenses and credits

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)), as in upstream
  [SandAI-org/MAGI-2-preview](https://github.com/SandAI-org/MAGI-2-preview), on which this fork is based
  (commit `f68a0f9`). The modifications listed above are released under the same license.
- **Weights** are not part of this repository and keep their own licenses. Check each before use,
  especially for commercial work:
  - MAGI-2-preview (preview, refiner, turbo decoder): see the
    [Hugging Face model card](https://huggingface.co/sand-ai/MAGI-2-preview) (Apache 2.0 at the time of writing).
  - Qwen3.5-27B: see [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B).
  - Wan2.2 VAE: see [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
  - Stable Audio Open 1.0 (audio VAE): distributed by Stability AI under its own community license; review its terms.
- Built on [MagiCompiler](https://github.com/SandAI-org/MagiCompiler),
  [FlashAttention](https://github.com/Dao-AILab/flash-attention), PyTorch `flex_attention` and Triton.
- Please only generate likenesses of real people with their consent.
