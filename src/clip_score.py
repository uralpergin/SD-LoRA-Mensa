#!/usr/bin/env python3
"""
Compute CLIP image‑text similarity scores for a folder of generations that use a learned concept token.

CHANGES (2025‑08‑03)
--------------------
* **Adds `--token-name` flag** – required when the `.pt` file is a *bare* tensor
  that doesn’t carry the placeholder string inside.
* `load_concept_embedding()` now handles three layouts:
  1. Automatic1111 textual‑inversion (`{"string_to_param": {"<token>": tensor}}`)
  2. Diffusers native (`{"<token>": tensor}`)
  3. **Bare `torch.Tensor`** (pass the token via `--token-name`).
* **NEW:** results now save to `<experiment>/eval/clip_scores/clip_scores.csv` so
  evaluation artefacts live in their own folder.

Example
~~~~~~~
```bash
python src/clip_score.py --token-emb /work/dlclarge2/matusd-lora/mensa-lora/experiments/base/lora_weights/final/token_emb.pt 
--experiment /work/dlclarge2/matusd-lora/mensa-lora/experiments/base 
--token-name "<mensafood>" 
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1.  Work around GGML import (set *before* transformers import)
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_NO_GGML_IMPORTS", "1")

from packaging.version import parse as parse_version  # type: ignore
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# 2.  Sanity‑check tokenizers (old wheels explode on import)
# ---------------------------------------------------------------------------
try:
    import tokenizers  # type: ignore
except ImportError:
    sys.exit("[FATAL] Missing 'tokenizers'. Install with: pip install tokenizers")

if parse_version(tokenizers.__version__) < parse_version("0.14.0"):
    sys.exit(
        f"[FATAL] tokenizers {tokenizers.__version__} is too old.\n"
        "        Upgrade with: pip install -U 'tokenizers>=0.21,<0.23'"
    )

# ---------------------------------------------------------------------------
# 3.  Helper functions
# ---------------------------------------------------------------------------

def load_concept_embedding(path: str, token_override: Optional[str] = None):
    """Return `(token_string, 768‑d torch.Tensor)` from various checkpoint layouts."""
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict) and "string_to_param" in ckpt:
        token, emb = next(iter(ckpt["string_to_param"].items()))

    elif isinstance(ckpt, dict) and len(ckpt) == 1 and isinstance(next(iter(ckpt.values())), torch.Tensor):
        token, emb = next(iter(ckpt.items()))

    elif isinstance(ckpt, torch.Tensor):
        if token_override is None:
            raise ValueError(
                "Checkpoint is a bare tensor. Pass the token via --token-name."
            )
        token, emb = token_override, ckpt

    else:
        raise ValueError("Unrecognised embedding checkpoint structure → " + path)

    # ensure shape (768,)
    emb = emb.squeeze(0) if emb.dim() == 2 else emb
    if emb.dim() != 1:
        raise ValueError(f"Embedding tensor must be 1‑D (got shape {list(emb.shape)})")

    return token, emb


def add_token_to_clip(processor, model, token: str, vector: torch.Tensor):
    """Insert `token` into tokenizer and grow CLIP's text-embedding layer if needed."""
    tokenizer = processor.tokenizer
    new_id = tokenizer.add_tokens([token])
    if isinstance(new_id, list):  # HF ≥ 4.39 returns a list
        new_id = new_id[0]

    old_emb = model.text_model.embeddings.token_embedding
    vocab_old, dim = old_emb.weight.shape
    vocab_new = len(tokenizer)

    device = old_emb.weight.device
    dtype = old_emb.weight.dtype

    if vocab_new > vocab_old:
        new_emb = torch.nn.Embedding(vocab_new, dim, device=device)
        with torch.no_grad():
            new_emb.weight[:vocab_old] = old_emb.weight
            torch.nn.init.normal_(new_emb.weight[vocab_old:], std=0.02)
        model.text_model.embeddings.token_embedding = new_emb

    with torch.no_grad():
        vec = vector.to(device=device, dtype=model.text_model.embeddings.token_embedding.weight.dtype)
        model.text_model.embeddings.token_embedding.weight[new_id] = vec


@torch.no_grad()
def clip_similarity(processor, model, image: Image.Image, text: str, device):
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    outs = model(**inputs)
    img = outs.image_embeds / outs.image_embeds.norm(dim=-1, keepdim=True)
    txt = outs.text_embeds / outs.text_embeds.norm(dim=-1, keepdim=True)
    return (img @ txt.T).item()


# ---------------------------------------------------------------------------
# 4.  Main CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token-emb", help="Path to token_emb.pt")
    p.add_argument("--experiment", required=True, help="Experiment root (contains 'outputs/')")
    p.add_argument("--token-name", help="Concept token string (needed if .pt lacks it)")
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    # 4.1 load
    if args.token_emb:
        tok, vec = load_concept_embedding(args.token_emb, args.token_name)
        add_token_to_clip(processor, model, tok, vec)
        print(f"[INFO] Added concept token {tok!r} to CLIP")
    else:
        print("[INFO] No token embedding supplied — using vanilla CLIP")

    # 4.2 iterate images
    out_dir = Path(args.experiment) / "outputs"
    images = sorted(out_dir.glob("*.png"))
    if not images:
        sys.exit(f"[FATAL] No .png files found in {out_dir}")

    rows: list[dict] = []
    for img_path in tqdm(images, desc="Scoring"):
        meta_candidates = list(out_dir.glob(f"{img_path.stem}*metadata*.json"))
        if not meta_candidates:
            print(f"[WARN] No metadata JSON for {img_path.name}, skipping.")
            continue
        with open(meta_candidates[0]) as f:
            meta = json.load(f)
        s = clip_similarity(processor, model, Image.open(img_path).convert("RGB"), meta.get("prompt", ""), device)
        meta.update({"image_path": str(img_path), "clip_score": s})
        rows.append(meta)

    if not rows:
        sys.exit("[FATAL] No images were scored. Check folder structure and metadata names.")

    # 4.3 save CSV into eval/clip_scores
    exp_path = Path(args.experiment)      
    root = exp_path.parent.parent
    csv_dir  = root / "eval" / "clip_scores"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{exp_path.name}.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {csv_path}")


if __name__ == "__main__":
    main()
