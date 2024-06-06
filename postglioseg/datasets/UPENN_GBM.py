from torchvision.transforms import v2
import torch
import polars as pl
from ..data import DS
from ..loaders import nifti

_MEAN, _STD = torch.tensor([0.0002149488]), torch.tensor([0.0005426282])
_normalize = v2.Normalize(_MEAN, _STD) # type:ignore

def _loader(study_row:str):
    study_id = study_row[0]
    path = rf"E:\dataset\PKG - UPENN-GBM-NIfTI\UPENN-GBM\NIfTI-files\images_DTI\images_DTI\{study_id}\{study_id}_DTI_AD.nii.gz"
    return _normalize(nifti.niireadtensor(path)[44:191, 36:222, 2:141].unsqueeze(0))


def get_dataset(path = r"E:\dataset\PKG - UPENN-GBM-NIfTI\UPENN-GBM") -> DS.DSRegression:
    df = pl.read_csv(f"{path}/UPENN-GBM_clinical_info_v2.1.csv", null_values = ["Not Available", "NA"]).drop_nulls("PsP_TP_score")
    ds = DS.DSRegression()
    ds.add_samples(df.rows(), target=lambda x: x[-1], loader=_loader)
    return ds
