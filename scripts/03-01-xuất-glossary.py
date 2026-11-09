#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from .utils.cste import METADATA_DIR
from .utils.misc import read_json

glossary_list = []
for i in METADATA_DIR.glob("*.json"):
	tmp = read_json(i)["glossary"]
	for j in tmp:
		j["chapter"] = i.stem
		glossary_list.append(j)

(pd.DataFrame
	.from_records(glossary_list)
	.groupby(["chinese", "vietnamese", "category"], as_index = False)
	.agg({"chapter": "-".join})
	.sort_values(by=["chinese"])
	.to_csv(METADATA_DIR / "out.csv", index=False)
)

# USER đọc lại glossary và sửa theo ý muốn
