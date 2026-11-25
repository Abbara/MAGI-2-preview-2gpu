#!/usr/bin/env python3
"""Verify a local MAGI-2-preview checkpoint directory against Hugging Face.

Compares every file of the pinned revision with the local copy:
  * default: file exists, exact byte size matches, and every .safetensors file
    has a parseable header whose data region ends exactly at end-of-file
    (catches truncated / partial downloads in seconds);
  * --sha256: additionally hashes LFS files and compares with the Hub's LFS
    SHA-256 (reads all ~307 GB; takes a while).

Standard library only, so it runs on any host Python 3.8+.

Usage:
  python3 scripts/verify_weights.py --ckpt ../ckpt [--revision SHA] [--only preview,vae] [--sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

REPO_ID = "sand-ai/MAGI-2-preview"
DEFAULT_REVISION = "2dea51b64db47ee5b4402d36fd90829a0c58913b"
COMPONENTS = ["preview", "refiner", "text_encoder", "vae", "turbo_vae", "stable-audio-open-1.0"]


def hub_listing(revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{REPO_ID}/tree/{revision}?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "magi2-verify-weights"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return [e for e in json.load(resp) if e.get("type") == "file"]


def check_safetensors(path: str) -> str | None:
    """Return an error string if the safetensors layout is inconsistent."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return "file shorter than safetensors header"
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or 8 + n > size:
            return f"bad header length {n}"
        try:
            header = json.loads(f.read(n))
        except json.JSONDecodeError as exc:
            return f"unparseable header: {exc}"
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    if 8 + n + end != size:
        return f"data region ends at {8 + n + end}, file is {size} bytes"
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="local checkpoint root (contains preview/, refiner/, ...)")
    ap.add_argument("--revision", default=os.environ.get("MAGI2_HF_REVISION", DEFAULT_REVISION))
    ap.add_argument("--only", default=",".join(COMPONENTS), help="comma-separated components to check")
    ap.add_argument("--sha256", action="store_true", help="also verify SHA-256 of LFS files (slow)")
    args = ap.parse_args()

    wanted = [c.strip() for c in args.only.split(",") if c.strip()]
    unknown = sorted(set(wanted) - set(COMPONENTS))
    if unknown:
        print(f"unknown component(s): {unknown}; choose from {COMPONENTS}", file=sys.stderr)
        return 2

    files = [e for e in hub_listing(args.revision) if e["path"].split("/")[0] in wanted]
    total = sum(e["size"] for e in files)
    print(f"Checking {len(files)} files ({total / 1e9:.1f} GB) of {REPO_ID}@{args.revision[:10]} in {args.ckpt}")

    problems = 0
    for e in sorted(files, key=lambda e: e["path"]):
        rel, expected = e["path"], e["size"]
        path = os.path.join(args.ckpt, rel)
        err = None
        if not os.path.isfile(path):
            err = "missing"
        elif os.path.getsize(path) != expected:
            err = f"size {os.path.getsize(path)} != expected {expected}"
        elif rel.endswith(".safetensors"):
            err = check_safetensors(path)
        if err is None and args.sha256 and e.get("lfs"):
            print(f"  hashing {rel} ...", flush=True)
            if sha256_file(path) != e["lfs"]["oid"]:
                err = "sha256 mismatch"
        if err:
            problems += 1
            print(f"  FAIL {rel}: {err}")

    if problems:
        print(f"{problems} problem(s). Re-run scripts/download_weights.sh to repair (it resumes).")
        return 1
    print("All files present and consistent" + (" (sha256 verified)." if args.sha256 else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
