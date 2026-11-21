# -*- coding: utf-8 -*-

from typing import Final
from pathlib import Path
from huggingface_hub import snapshot_download
import exllamav3 as exl3

from .misc import read_json
from .prompts import INSTRUCTIONS_DỊCH, format_prompt_gemma4

_CACHE_SIZE: Final = 71680 # context must be multiple of 256
_DEVICE: Final = "cuda:0"
_BATCH_SIZE: Final = 1

_DRAFT_CONFIG: Final = exl3.Config.from_directory(snapshot_download(repo_id="turboderp/gemma4-31b-it-DFlash-exl3", revision="6.00bpw"))
_DRAFT_MODEL: Final = exl3.Model.from_config(_DRAFT_CONFIG, swa_full=True)
_DRAFT_CACHE: Final = exl3.Cache(_DRAFT_MODEL, max_batch_size=_BATCH_SIZE, max_num_tokens=_CACHE_SIZE)
_DRAFT_MODEL.load(progressbar=True, device=_DEVICE)

_MAIN_CONFIG: Final = exl3.Config.from_directory(snapshot_download(repo_id="UnstableLlama/gemma-4-31B-it-uncensored-heretic-exl3-8.00bpw"))
_MAIN_MODEL: Final = exl3.Model.from_config(_MAIN_CONFIG, swa_full=True) # should enable SWA for gemma 4 but take much more VRAM
_MAIN_CACHE: Final = exl3.Cache(_MAIN_MODEL, max_batch_size=_BATCH_SIZE, max_num_tokens=_CACHE_SIZE)
_MAIN_MODEL.load(progressbar=True, device=_DEVICE)

_TOKENIZER: Final = exl3.Tokenizer.from_config(_MAIN_CONFIG)
_STOP_TOKENS: Final = ["<eos>", "<turn|>"]
_BANNED_STUFF_gemma4: Final = read_json(Path(__file__).with_name("banned_gemma4.json"))

_GENERATOR: Final = exl3.Generator(
	model=_MAIN_MODEL,
	cache=_MAIN_CACHE,
	tokenizer=_TOKENIZER,
	draft_model=_DRAFT_MODEL,
	draft_cache=_DRAFT_CACHE
)
_GEN_CONFIG: Final = exl3.ComboSampler(
	temperature=.2,
	top_k=20,
	top_p=.7,
	min_p=.1,
	rep_p=1.1, # TODO: exl3 has loop detection which may be more useful
	logit_bias={i: "-inf" for i in _BANNED_STUFF_gemma4["banned_token_ids"]}
)

def dịch(text: str) -> str:
	return _GENERATOR.generate(
		prompt=format_prompt_gemma4(INSTRUCTIONS_DỊCH, text),
		# prompt=[format_prompt(…, i) for i in …], # batched but speed not better
		stop_conditions=_STOP_TOKENS,
		sampler=_GEN_CONFIG,
		completion_only=True,
		encode_special_tokens=True,
	)

def dịch_streaming(text: str) -> None:
	_GENERATOR.enqueue(exl3.Job(
		input_ids=_TOKENIZER.encode(format_prompt_gemma4(INSTRUCTIONS_DỊCH, text), encode_special_tokens=True),
		stop_conditions=_STOP_TOKENS,
		sampler=_GEN_CONFIG,
	))
	while _GENERATOR.num_remaining_jobs() > 0:
		for result in _GENERATOR.iterate():
			print(result.get("text", ""), end = "", flush = True)

def dịch_batched(texts: list[str], progress_bar: "tqdm" = None) -> list[str]:
	N = len(texts)
	res = [""] * N # placeholder
	for i, t in enumerate(texts):
		_GENERATOR.enqueue(exl3.Job(
			identifier=i, # to save order, in case order not conserved
			input_ids=_TOKENIZER.encode(format_prompt_gemma4(INSTRUCTIONS_DỊCH, t), encode_special_tokens=True),
			stop_conditions=_STOP_TOKENS,
			sampler=_GEN_CONFIG,
			banned_strings=_BANNED_STUFF_gemma4["banned_strings"],
		))
	while _GENERATOR.num_remaining_jobs() > 0:
		for result in _GENERATOR.iterate():
			# do not process progress_bar for each, too many connections lag marimo molab
			if result["eos"]:
				i = result["identifier"]
				res[i] = result["full_completion"]
				if progress_bar is not None:
					chr_count = len(texts[i].replace("\n", ""))
					tps = result["new_tokens"] / result["time_generate"]
					progress_bar.update(chr_count)
					progress_bar.set_postfix({"prev_chap": f"{i}/{N} ({chr_count}chr)", "prev_TPS": f"{tps:.1f}tok/s"})
	return res

# _DRAFT_MODEL.unload()
# _DRAFT_CACHE.detach_from_model(_DRAFT_MODEL)
# _MAIN_MODEL.unload()
# _MAIN_CACHE.detach_from_model(_MAIN_MODEL)
