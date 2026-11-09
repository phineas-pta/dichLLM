#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from .utils.cste import METADATA_DIR
from .utils.misc import read_json

DF = pd.read_csv(METADATA_DIR / "out.csv")
DF["chapter"] = DF["chapter"].str.split("-")
DF = DF.explode("chapter")
