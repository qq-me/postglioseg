"""Cepeda S, García-García S, Arrese I, Herrero F, Escudero T, Zamora T, Sarabia R. The Río Hortega University Hospital Glioblastoma dataset: A comprehensive collection of preoperative, early postoperative and recurrence MRI scans (RHUH-GBM). Data Brief. 2023 Sep 23;50:109617. doi: 10.1016/j.dib.2023.109617. PMID: 37808543; PMCID: PMC10551826."""

from typing import Sequence, Any
import polars as pl
class RHUH_GBM:
    def __init__(self):
        self.clinical_data = pl.read_csv(r"F:\Stuff\Programming\AI\glio_diff\glio\datasets\RHUH-GBM.csv").with_columns(
                center=pl.col("center").map_elements(lambda x: [int(float(i)) for i in str(x).split(", ")]),
                src=pl.lit("center"),)
        self.path = r"E:\dataset\RHUH-GBM\RHUH-GBM_nii_v1"

    def get_cols(self, cols) -> Sequence[dict[str, Any]] :
        return self.clinical_data.select(cols).drop_nulls().to_dicts()

    def col_names(self):
        return self.clinical_data.columns
