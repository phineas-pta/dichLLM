# -*- coding: utf-8 -*-

from typing import Final
from pathlib import Path

_DATA_DIR: Final = Path(__file__).parents[2] / "data"

METADATA_DIR: Final = _DATA_DIR / "metadata"
TRANSL_DIR: Final = _DATA_DIR / "transl"
CHAPTERS_LIST: Final = sorted((_DATA_DIR / "raw").glob("*.txt"))
