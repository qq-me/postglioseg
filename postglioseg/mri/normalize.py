"""PostGlioSeg, автор - Никишев Иван Олегович"""
from collections.abc import Sequence
import SimpleITK as sitk
def znormalize(input: str | sitk.Image) -> sitk.Image:
    if isinstance(input, str): input = sitk.ReadImage(input)
    return sitk.Normalize(input)

def znormalize_imgs(inputs: str | sitk.Image | Sequence[str | sitk.Image]) -> list[sitk.Image]:
    return [znormalize(i) for i in inputs]