"""Siodj"""

from typing import Sequence, Any
import polars as pl
class BRaTS2023_GBM:
    def __init__(self):
        self.data = pl.read_csv(r"F:\Stuff\Programming\AI\glio_diff\glio\datasets\BRaTS2023-GLI.csv").with_columns(
                center=pl.col("center").map_elements(lambda x: [int(float(i)) for i in str(x).split(", ")]),
                src=pl.lit("center"),)

    def get_cols(self, cols) -> Sequence[dict[str, Any]] :
        return self.data.select(cols).drop_nulls().to_dicts()

    def col_names(self):
        return self.data.columns
