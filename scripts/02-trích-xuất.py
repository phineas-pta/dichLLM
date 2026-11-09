#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# rm -r /home/marimo/.cache/huggingface/hub

from tqdm import tqdm

from .utils.cste import CHAPTERS_LIST, METADATA_DIR
from .utils.model_qwen38 import trích, trích_streaming
from .utils.misc import dump_json

# Qwen3.8 xử lí nhanh hơn (mỗi giây ~120 chữ hán) và kết quả đầy đủ hơn
# Gemma4 chậm hơn (mỗi giây ~220 chữ hán) và trả về nửa chừng

# warm-up
trích_streaming(CHAPTERS_LIST[0].read_text(encoding="utf-8"))

chr_count = [
	len(j.read_text(encoding="utf-8").replace("\n", ""))
	for j in CHAPTERS_LIST
]
with tqdm(total=sum(chr_count), unit="chr") as pbar:
	for i, j in enumerate(CHAPTERS_LIST):
		pbar.set_description(f"Processing chap {i} ({chr_count[i]} chr)")
		json_file = (METADATA_DIR / j.name).with_suffix(".json")
		metadata = trích(j.read_text(encoding="utf-8"))
		dump_json(metadata, json_file)
		pbar.update(chr_count[i])

# zip -r ../metadata.zip metadata/
