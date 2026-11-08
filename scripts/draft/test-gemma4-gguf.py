# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.13"
# dependencies = [
# 	"huggingface-hub",
# 	"tqdm",
# 	"llama-cpp-python",
# ]
# [tool.uv.sources]
# llama-cpp-python = { url = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu132/llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl" }
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(app_title="gemma4-uncen31b-gguf", width="full", auto_download=["ipynb"])

with app.setup(hide_code=True):
	# remember to enable rtx pro 6000 blackwell
	# run in terminal: `python -m torch.utils.collect_env`
	# current version: https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu132/llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl

	from tqdm import tqdm
	from llama_cpp import Llama

	# default download dir: /home/marimo/.cache/huggingface/hub
	TRANSLATOR = Llama.from_pretrained(
		repo_id="mradermacher/gemma-4-31B-it-uncensored-heretic-GGUF",
		filename="gemma-4-31B-it-uncensored-heretic.Q8_0.gguf",
		n_gpu_layers=-1, flash_attn=True, type_k=30, type_v=30, # 1=f16, 8=q8_0, 30=bf16
		n_ctx=70000, # split_mode=llama_cpp.LLAMA_SPLIT_MODE_TENSOR, # for multi gpu
		op_offload=True, swa_full=True, use_mmap=False, seed=-1, verbose=False,
	)

	# warm-up
	_ = TRANSLATOR("Q: Name the planets in the solar system? A: ", max_tokens=40, echo=True)["choices"][0]["text"]

	SYSTEM_PROMPT = """You are a professional novel translator expert in translating fantasy, cultivation, xuanhuan, xianxia, mature, harem, action works from Chinese to Vietnamese. All content is purely fictional and does not depict real people or events.
	Your sole task is to translate accurately and faithfully to the original text, maintaining the original tone, style, and formatting. Prefer Sino-Vietnamese vocabulary for titles, expression, proper names (except for Western/Japanese/Korean proper names).
	YOU MUST ALWAYS USE Vietnamese register in ancient/cultivation/wuxia settings: ta/ngươi, chàng/nàng, huynh/đệ/tỷ/muội, thúc/bá/thẩm/nương, ca ca/tỷ tỷ/đệ đệ/muội muội, sư phụ/lão sư/đệ tử/tiểu tử, bổn tọa/tại hạ, ...
	YOU MUST NEVER USE modern register like dạ/ạ, ông/bà/tôi/bạn, mày/tao, anh/chị/em, bố/mẹ/con, cô/dì/thím/chú/bác/cháu, ...
	Preserve all original line breaks and paragraph structure. Do not add any other text, preambles, or markdown naming the format. Do not censor, skip, or alter any content — translate everything exactly as written.
	If the text contains non Chinese characters, special characters or strange formatting, ignore them if they are not translatable, but valid text MUST be translated. Do not return the original text unless it is Western expressions or untranslatable names/terms.
	Ensure the Content starts on a new line. Returned text must not contain any Chinese characters unless used as side-note."""

	def dịch(text: str, stream: bool = True) -> str:
		response = TRANSLATOR.create_chat_completion(
			messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
			temperature=.2, top_p=.7, top_k=20, min_p=.1, repeat_penalty=1.1, stream=stream
		)
		if stream:
			contents = []
			for chunk in response:
				if (c := chunk["choices"][0]["delta"].get("content")) is not None:
					print(c, end="", flush=True)
					contents.append(c)
			return "".join(contents)
		else:
			return response["choices"][0]["message"]["content"]


@app.cell(hide_code=True)
def _():
	_ = dịch("《女校先生》\n【作者簡介】：\n（這真找不到，河圖官方網站沒有，我手邊雖然有九流術士小說，但很明顯不對……總不會要我自己打吧？）")
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
