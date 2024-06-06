"""Suter, Yannick, et al. "The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation." Scientific data 9.1 (2022): 768."""
from typing import Sequence, Any
import os
import polars as pl

_RANO = "Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)"
_RANO_RATIONALE = "Rating rationale (CRET: complete resection of the enhancing tumor, PRET: partial resection of the enhancing tumor, T2-Progr.: T2-Progression, L: Lesion)"

class LUMIERE:
    def __init__(self, path = r"E:\dataset\LUMIERE"):
        # 1. Чтение csv и формирования DataFrame
        demographics_pathology = pl.read_csv(f"{path}/LUMIERE-Demographics_Pathology.csv", infer_schema_length=10000)


        def rano_norm(x: str):
            if x == "Post-Op ": return "Post-Op"
            elif x == "Post-Op/PD": return "PD"
            else: return x

        expert_rating = (
            pl.read_csv(f"{path}/LUMIERE-ExpertRating-v202211.csv", infer_schema_length=10000)
            .with_columns(RANO=pl.col(_RANO).map_elements(rano_norm))
            .drop(_RANO, _RANO_RATIONALE))

        deepbratumia = (
            pl.read_csv(
                f"{path}/LUMIERE-pyradiomics-deepbratumia-features.csv",
                infer_schema_length=10000,)
            .select("Patient","Time point","Image","Mask","Label name","Sequence","diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_CenterOfMassIndex")
            .with_columns(
                bbox=pl.col("diagnostics_Mask-original_BoundingBox").map_elements(lambda x: [int(i) for i in x[1:-1].split(", ")]),
                src=pl.lit("Deepbratumia"),)
            .drop("diagnostics_Mask-original_BoundingBox")
            .with_columns(
                bbox_center=pl.col("diagnostics_Mask-original_CenterOfMassIndex").map_elements(lambda x: [int(float(i)) for i in x[1:-1].split(", ")]),
                src=pl.lit("Deepbratumia"),)
            .drop("diagnostics_Mask-original_CenterOfMassIndex")
            .rename({"Time point": "Date"}))

        hdglioauto = (
            pl.read_csv(
                f"{path}/LUMIERE-pyradiomics-hdglioauto-features.csv", infer_schema_length=10000)
            .select("Patient","Time point","Image","Mask","Label name","Sequence","diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_CenterOfMassIndex")
            .with_columns(
                bbox=pl.col("diagnostics_Mask-original_BoundingBox").map_elements(lambda x: [int(i) for i in x[1:-1].split(", ")]),
                src=pl.lit("HD-GLIO-AUTO"),)
            .drop("diagnostics_Mask-original_BoundingBox")
            .with_columns(
                bbox_center=pl.col("diagnostics_Mask-original_CenterOfMassIndex").map_elements(lambda x: [int(float(i)) for i in x[1:-1].split(", ")]),
                src=pl.lit("HD-GLIO-AUTO"),)
            .drop("diagnostics_Mask-original_CenterOfMassIndex")
            .rename({"Time point": "Date"}))

        combined = (
            demographics_pathology.join(expert_rating, on="Patient", how="outer_coalesce")
            .join(deepbratumia, on=["Patient", "Date"], how="outer_coalesce")
            .join(hdglioauto, on=["Patient", "Date"], how="outer_coalesce"))

        # 2. Аггрегация колонок
        d = {}
        for ob in combined.rows(named=True):
            name = f'{ob["Patient"]}/{ob["Date"]}'
            if name not in d: d[name] = ob.copy()
            if "Sequence" in d[name]: del d[name]["Sequence"]
            if "Label name" in d[name]: del d[name]["Label name"]
            if "Image" in d[name]: del d[name]["Image"]
            if "Mask" in d[name]: del d[name]["Mask"]
            if "src" in d[name]: del d[name]["src"]

            if "Image_right" in d[name]: del d[name]["Image_right"]
            if "Mask_right" in d[name]: del d[name]["Mask_right"]
            if "Label name_right" in d[name]: del d[name]["Label name_right"]
            if "Sequence_right" in d[name]: del d[name]["Sequence_right"]
            if "bbox_center_right" in d[name]: del d[name]["bbox_center_right"]
            if "bbox_right" in d[name]: del d[name]["bbox_right"]
            if "src_right" in d[name]: del d[name]["src_right"]
            seq = ob["Sequence"]
            label = ob["Label name"].lower() if ob["Label name"] is not None else None
            d[name][f"{seq} image"] = f'{path}/Imaging/{ob["Image"]}'.replace("LUMIERE/Patient", "LUMIERE/Imaging/Patient")
            if not os.path.isfile(d[name][f"{seq} image"]): d[name][f"{seq} image"] = None
            d[name][f"{seq} mask"] = f'{path}/{ob["Mask"]}'.replace("LUMIERE/Patient", "LUMIERE/Imaging/Patient")
            if not os.path.isfile(d[name][f"{seq} mask"]): d[name][f"{seq} mask"] = None
            d[name][f"{seq} {label} bbox"] = ob["bbox"]
            d[name][f"{seq} {label} bbox center"] = ob["bbox_center"]

            seq = ob["Sequence_right"]
            label = ob["Label name_right"].lower() if ob["Label name_right"] is not None else None
            d[name][f"{seq} hga-image"] = f'{path}/{ob["Image"]}'.replace("LUMIERE/Patient", "LUMIERE/Imaging/Patient")
            if not os.path.isfile(d[name][f"{seq} hga-image"]): d[name][f"{seq} hga-image"] = None
            d[name][f"{seq} hga-mask"] = f'{path}/{ob["Mask"]}'.replace("LUMIERE/Patient", "LUMIERE/Imaging/Patient")
            if not os.path.isfile(d[name][f"{seq} hga-mask"]): d[name][f"{seq} hga-mask"] = None
            d[name][f"{seq} hga-{label} bbox"] = ob["bbox"]
            d[name][f"{seq} hga-{label} bbox center"] = ob["bbox_center"]

        self.df = pl.DataFrame(list(d.values()))

    def get_cols(self, cols:Sequence[str]) -> Sequence[dict[str, Any]]:
        return self.df.select(cols).drop_nulls().to_dicts()

    def col_names(self) -> Sequence[str]:
        return self.df.columns
