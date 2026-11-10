# -*- coding: utf-8 -*-

from typing import Final
from huggingface_hub import snapshot_download
import exllamav3 as exl3

from .prompts import INSTRUCTIONS_DỊCH, format_prompt_gemma4 as format_prompt

# default download dir: /home/marimo/.cache/huggingface/hub
_MODEL_CONFIG: Final = exl3.Config.from_directory(snapshot_download(repo_id="UnstableLlama/gemma-4-31B-it-uncensored-heretic-exl3-8.00bpw"))
_MODEL: Final = exl3.Model.from_config(_MODEL_CONFIG, swa_full=True) # should enable SWA for gemma 4 but take much more VRAM
_CACHE: Final = exl3.Cache(_MODEL, max_batch_size=1, max_num_tokens=76800) # context must be multiple of 256
_MODEL.load(progressbar=True, device="cuda:0") # for multi gpu, replace `device="cuda"` with `tensor_p=True, tp_backend="nccl"`

_TOKENIZER: Final = exl3.Tokenizer.from_config(_MODEL_CONFIG)
_STOP_TOKENS: Final = ["<eos>", "<turn|>"]
_GENERATOR: Final = exl3.Generator(model=_MODEL, tokenizer=_TOKENIZER, cache=_CACHE)
_GEN_CONFIG: Final = exl3.ComboSampler(temperature=.2, top_k=20, top_p=.7, min_p=.1, rep_p=1.1)

def dịch(text: str) -> str:
	return _GENERATOR.generate(
		prompt=format_prompt(INSTRUCTIONS_DỊCH, text),
		stop_conditions=_STOP_TOKENS,
		sampler=_GEN_CONFIG,
		completion_only=True,
		encode_special_tokens=True,
	)

def dịch_streaming(text: str) -> None:
	_GENERATOR.enqueue(exl3.Job(
		input_ids=_TOKENIZER.encode(format_prompt(INSTRUCTIONS_DỊCH, text), encode_special_tokens=True),
		stop_conditions=_STOP_TOKENS,
		sampler=_GEN_CONFIG,
	))
	while _GENERATOR.num_remaining_jobs():
		for result in _GENERATOR.iterate():
			print(result.get("text", ""), end = "", flush = True)
