#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stand-alone script
generate list of tokens to be masked to prevent language confusion/contamination in LLM outputting Vietnamese

adapted from https://huggingface.co/ThakiCloud/Qwen3.8-27B-vi-cjk-suppressed/blob/main/build_masks_multi.py

DO NOT MASK Chinese tokens because in Gemma 4 `lm_head` is tied to `embed_tokens` (`tie_word_embeddings=True`),
i.e. masking would prevent model from accepting Chinese input
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

class URange: # unicode range
	def __init__(self, lower: int, upper: int) -> None:
		assert lower <= upper
		self.lower = lower
		self.upper = upper
	def __contains__(self, text: str) -> bool:
		for char in text:
			tmp = ord(char)
			if self.lower <= tmp <= self.upper:
				return True
		return False
	def __lt__(self, other) -> bool: # to sort (see below)
		if self.upper < other.lower:
			return True
		elif self.lower > other.upper:
			return False
		else:
			raise ValueError("not disjoint")

EXCLUDED_RANGES = sorted([ # only highly impactful scripts, no need to list all unicode scripts
	# URange(0x4E00, 0x9FFF), URange(0x3400, 0x4DBF), URange(0xF900, 0xFAFF), # Script=Han
	URange(0x3040, 0x309F), URange(0x30A0, 0x30FF), URange(0x31F0, 0x31FF), # Script=Katakana
	URange(0xAC00, 0xD7A3), URange(0x1100, 0x11FF), URange(0x3130, 0x318F), # Script=Hangul
	URange(0x0600, 0x06FF), URange(0x0750, 0x077F), URange(0x08A0, 0x08FF), # Script=Arabic
	URange(0x0E00, 0x0E7F), # Script=Thai
	URange(0x0400, 0x04FF), # Script=Cyrillic
	URange(0x0900, 0x097F), # Script=Devanagari
	URange(0x0590, 0x05FF), # Script=Hebrew
	URange(0x0370, 0x03FF), # Script=Greek
])

TOKENIZER = AutoTokenizer.from_pretrained("UnstableLlama/gemma-4-31B-it-uncensored-heretic-exl3-8.00bpw")
# TOKENIZER = AutoTokenizer.from_pretrained("turboderp/Qwen3.8-Flash-Next-exl3", revision="6.05bpw_h6_ng6")

N = max(TOKENIZER.get_vocab().values()) + 1
target_token_ids = []
for tid in range(N):
	tmp = TOKENIZER.decode([tid], skip_special_tokens=False)
	for interval in EXCLUDED_RANGES:
		if tmp in interval:
			target_token_ids.append(tid)
			break # because list is sorted, no need to check for others

(Path(__file__).parent / "mask_gemma4.json").write_text(
	json.dumps({"target_token_ids": sorted(target_token_ids)}, ensure_ascii=False, indent="\t"),
	encoding="utf-8"
)
