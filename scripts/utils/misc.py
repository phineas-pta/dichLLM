# -*- coding: utf-8 -*-

from pathlib import Path
import json

def read_json(file: Path) -> dict:
	return json.loads(file.read_text(encoding="UTF-8"))

def dump_json(text: str, file: Path) -> None:
	file.write_text(
		json.dumps(json.loads(text), ensure_ascii=False, indent="\t"),
		encoding="UTF-8"
	)
