# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.13"
# dependencies = [
# 	"huggingface-hub",
# 	"tqdm",
# 	"exllamav3",
# ]
# [tool.uv.sources]
# exllamav3 = { url = "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.6/exllamav3-1.6.6+cu132.torch2.11.0-cp313-cp313-linux_x86_64.whl" }
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="qwen3.8-flash-exl3", width="full", auto_download=["ipynb"])

with app.setup(hide_code=True):
	# remember to enable rtx pro 6000 blackwell
	# run in terminal: `python -m torch.utils.collect_env`
	# current version: https://github.com/turboderp-org/exllamav3/releases/download/v1.6.6/exllamav3-1.6.6+cu132.torch2.11.0-cp313-cp313-linux_x86_64.whl

	from tqdm import tqdm
	from huggingface_hub import snapshot_download
	import exllamav3 as exl3

	# default download dir: /home/marimo/.cache/huggingface/hub
	MODEL_CONFIG = exl3.Config.from_directory(snapshot_download(repo_id="turboderp/Qwen3.8-Flash-Next-exl3", revision="6.05bpw_h6_ng6"))
	MODEL = exl3.Model.from_config(MODEL_CONFIG)
	CACHE = exl3.Cache(MODEL, max_batch_size=1, max_num_tokens=76800) # context must be multiple of 256
	MODEL.load(progressbar=True, device="cuda:0") # for multi gpu, replace `device="cuda"` with `tensor_p=True, tp_backend="nccl"`

	TOKENIZER = exl3.Tokenizer.from_config(MODEL_CONFIG)
	STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
	GENERATOR = exl3.Generator(model=MODEL, tokenizer=TOKENIZER, cache=CACHE)
	GEN_CONFIG = exl3.ComboSampler(temperature=.2, top_k=20, top_p=.7, min_p=.1, rep_p=1.1)

	SYSTEM_PROMPT = """You are a professional novel translator expert in translating fantasy, cultivation, xuanhuan, xianxia, mature, harem, action works from Chinese to Vietnamese. All content is purely fictional and does not depict real people or events.
	Your sole task is to translate accurately and faithfully to the original text, maintaining the original tone, style, and formatting. Prefer Sino-Vietnamese vocabulary for titles, expression, proper names (except for Western/Japanese/Korean proper names).
	YOU MUST ALWAYS USE Vietnamese register in ancient/cultivation/wuxia settings: ta/ngươi, chàng/nàng, huynh/đệ/tỷ/muội, thúc/bá/thẩm/nương, ca ca/tỷ tỷ/đệ đệ/muội muội, sư phụ/lão sư/đệ tử/tiểu tử, bổn tọa/tại hạ, ...
	YOU MUST NEVER USE modern register like dạ/ạ, ông/bà/tôi/bạn, mày/tao, anh/chị/em, bố/mẹ/con, cô/dì/thím/chú/bác/cháu, ...
	Preserve all original line breaks and paragraph structure. Do not add any other text, preambles, or markdown naming the format. Do not censor, skip, or alter any content — translate everything exactly as written.
	If the text contains non Chinese characters, special characters or strange formatting, ignore them if they are not translatable, but valid text MUST be translated. Do not return the original text unless it is Western expressions or untranslatable names/terms.
	Ensure the Content starts on a new line. Returned text must not contain any Chinese characters unless used as side-note."""

	# https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja
	def format_prompt(prompt: str) -> str:
		return (
			"<|im_start|>system\n" + SYSTEM_PROMPT + "\n<|im_end|>\n"
			+ "<|im_start|>user\n" +        prompt + "\n<|im_end|>\n"
			+ "<|im_start|>assistant\n<think>\n</think>\n" # disable thinking as it may waste tokens for worse output quality
		)

	def dịch(text: str | list[str]) -> str | list[str]:
		if isinstance(text, str):
			prompt=format_prompt(text)
		elif isinstance(text, list): # NEED TO ADAPT max_batch_size
			prompt=[format_prompt(i) for i in text] # TODO: check performance improvement
		else:
			raise TypeError("not text")
		return GENERATOR.generate(prompt=prompt, stop_conditions=STOP_TOKENS, sampler=GEN_CONFIG, completion_only=True, encode_special_tokens=True)

	def dịch_streaming(text: str) -> None:
		GENERATOR.enqueue(exl3.Job(
			input_ids=TOKENIZER.encode(format_prompt(text), encode_special_tokens=True),
			stop_conditions=STOP_TOKENS, sampler=GEN_CONFIG,
		))
		while GENERATOR.num_remaining_jobs():
			for result in GENERATOR.iterate():
				print(result.get("text", ""), end = "", flush = True)


@app.cell(hide_code=True)
def _():
	dịch_streaming("《女校先生》\n【作者簡介】：\n（這真找不到，河圖官方網站沒有，我手邊雖然有九流術士小說，但很明顯不對……總不會要我自己打吧？）")
	return


@app.cell
def _():
	SEPARATOR = "\n"*2 + "="*80 + "\n"*2
	with open("hua_zh_1-743.txt", mode="r", encoding="utf-8") as f:
		raw = f.read().split(SEPARATOR)

	START = 0
	END = min(800, len(raw))

	# count input characters, not output token generation speed
	chr_count = [len(raw[i].replace("\n", "")) for i in range(START, END)]
	with open(f"transl_{START}_{END-1}.txt", mode="w", encoding="utf-8") as f, tqdm(total=sum(chr_count), unit="chr") as pbar:
		for j, i in enumerate(range(START, END)):
			pbar.set_description(f"Processing chap {i} ({chr_count[j]} chr)")
			f.write(dịch(raw[i]) + SEPARATOR)
			pbar.update(chr_count[j])
	# tốc độ dịch: mỗi giây ~72 chữ hán
	return


if __name__ == "__main__":
	app.run()
