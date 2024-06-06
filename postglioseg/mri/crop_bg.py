"""PostGlioSeg, автор - Никишев Иван Олегович"""
from collections.abc import Sequence
import SimpleITK as sitk
def crop_bg(input:str | sitk.Image) -> sitk.Image:
    if isinstance(input, str): input = sitk.ReadImage(input)
    tissue_filter = sitk.LabelShapeStatisticsImageFilter()
    tissue_filter.Execute(sitk.OtsuThreshold(input, 0, 255))
    tissue = tissue_filter.GetBoundingBox(255)
    return sitk.RegionOfInterest( input, tissue[int(len(tissue) / 2) :],  tissue[0 : int(len(tissue) / 2)],)

def crop_bg_imgs(inputs:str | sitk.Image | Sequence[str | sitk.Image]) -> list[sitk.Image]:
    if not isinstance(inputs, Sequence): inputs = [inputs]
    inputs = [sitk.ReadImage(i) if isinstance(i, str) else i for i in inputs]
    tissue_filter = sitk.LabelShapeStatisticsImageFilter()
    tissue_filter.Execute(sitk.OtsuThreshold(inputs[0], 0, 255))
    tissue = tissue_filter.GetBoundingBox(255)
    return [sitk.RegionOfInterest(i, tissue[int(len(tissue) / 2) :],  tissue[0 : int(len(tissue) / 2)],) for i in inputs]