#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stand-alone script
MUST BE RUN BEFORE LOADING ANY MODEL

adapted from https://huggingface.co/ThakiCloud/Qwen3.8-27B-vi-cjk-suppressed/blob/main/apply_mask.py

only lm_head.weight changes, so rather than making a dozen-GB copy, it's better to download the original weight and apply the mask.
should take a few minutes on CPU, no GPU needed.
"""

import json
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from safetensors.torch import load_file, save_file

# default download dir: /home/marimo/.cache/huggingface/hub
MODEL_DIR = snapshot_download(repo_id="UnstableLlama/gemma-4-31B-it-uncensored-heretic-exl3-8.00bpw")
# MODEL_DIR = snapshot_download(repo_id="turboderp/Qwen3.8-Flash-Next-exl3", revision="6.05bpw_h6_ng6")

repl32    = json.loads((Path(__file__).parent / "repl32_gemma4.json").read_text(encoding="utf-8"))["repl32"]
TOKEN_IDs = json.loads((Path(__file__).parent /   "mask_gemma4.json").read_text(encoding="utf-8"))["target_token_ids"]

# find shard:
# - Gemma4: "model.language_model.embed_tokens.weight" (lm_head is tied to embed_tokens)
# - Qwen38: "lm_head.weight"
LAYER = "model.language_model.embed_tokens.weight"
SHARD = json.loads(Path(MODEL_DIR, "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"][LAYER]
WEIGHTS = load_file(Path(MODEL_DIR, SHARD))

WEIGHTS[LAYER][torch.tensor(TOKEN_IDs, dtype=torch.long)] = torch.tensor(repl32, dtype=WEIGHTS[LAYER].dtype).unsqueeze(dim=0).expand(len(TOKEN_IDs), -1)
save_file(WEIGHTS, Path(MODEL_DIR, SHARD))

# FAILED ATTEMPT: Gemma4 still returns output with language contamination
