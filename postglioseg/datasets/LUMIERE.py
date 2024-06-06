"""Suter, Yannick, et al. "The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation." Scientific data 9.1 (2022): 768."""
from typing import Sequence, Any
import os
import polars as pl

_RANO = "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)"
_RANO_RATIONALE = "Rating rationale (CRET: complete resection of the enhancing tumor, PRET: partial resection of the enhancing tumor, T2-Progr.: T2-Progression, L: Lesion)"

class LUMIERE:
    def __init__(self):
        self.df = pl.read_avro(r"F:\Stuff\Programming\AI\glio_diff\glio\datasets\LUMIERE.avro")

    def get_cols(self, cols:Sequence[str]) -> Sequence[dict[str, Any]]:
        return self.df.select(cols).drop_nulls().to_dicts()

    def col_names(self) -> Sequence[str]:
        return self.df.columns
