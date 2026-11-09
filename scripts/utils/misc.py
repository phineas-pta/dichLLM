# -*- coding: utf-8 -*-

from typing import Final
from pathlib import Path
import json

def read_json(file: Path) -> dict:
	return json.loads(file.read_text(encoding="UTF-8"))

def dump_json(text: str, file: Path) -> None:
	with file.open(mode="w", encoding="UTF-8") as f:
		json.dump(json.loads(text), f, ensure_ascii=False, indent="\t")
