# -*- coding: utf-8 -*-

from tqdm import tqdm

from .utils.cste import CHAPTERS_LIST, METADATA_DIR
from .utils.model import trích, trích_streaming

# warm-up
trích_streaming(CHAPTERS_LIST[0].read_text(encoding="utf-8"))

START = 0
END = min(100, len(CHAPTERS_LIST))

chr_count = [
	len(CHAPTERS_LIST[i].read_text(encoding="utf-8").replace("\n", ""))
	for i in range(START, END)
]

with tqdm(total=sum(chr_count), unit="chr") as pbar:
	for j, i in enumerate(range(START, END)):
		pbar.set_description(f"Processing chap {i} ({chr_count[j]} chr)")
		json_file = (METADATA_DIR / CHAPTERS_LIST[i].name).with_suffix(".json")
		metadata = trích(CHAPTERS_LIST[i].read_text(encoding="utf-8"))
		json_file.write_text(metadata, encoding="utf-8")
		pbar.update(chr_count[j])
# tốc độ xử lí: mỗi giây ~120 chữ hán

# zip -r metadata.zip data/metadata/
