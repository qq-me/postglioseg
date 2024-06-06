"""PostGlioSeg, автор - Никишев Иван Олегович"""
from collections.abc import Sequence
from typing import Any
import numpy as np
import SimpleITK as sitk


from ..data import MNI152_NII_PATH as MNI152
from ..data import SRI24_NII_PATH as SRI24

def resample_to(input:str | sitk.Image, reference: str | sitk.Image, interpolation=sitk.sitkNearestNeighbor) -> sitk.Image:
    """Resample `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded.

    Resampling uses spatial information embedded in nifti file / sitk.Image - size, origin, spacing and direction.

    `input` is transformed in such a way that those attributes will match `reference`.
    That doesn't guarantee perfect allginment.
    """
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    return sitk.Resample(
            input,
            reference,
            sitk.Transform(),
            interpolation
        )

def default_pmap():
    """Default parameter maps for registration"""
    pmap = sitk.VectorOfParameterMap()
    pmap.append(sitk.GetDefaultParameterMap("translation"))
    pmap.append(sitk.GetDefaultParameterMap("rigid"))
    pmap.append(sitk.GetDefaultParameterMap("affine"))
    return pmap

class ProgressDummy:
    def __call__(self, *args, **kwargs): pass
    
def register_to(input:str | sitk.Image, reference: str | sitk.Image, pmap = default_pmap()) -> sitk.Image:
    """Register `input` to `reference`, both can be either a `sitk.Image` or a path to a nifti file that will be loaded. Returns `input` registered to `reference`.

    Registering means input image is transformed using affine transforms to match the reference,
    where affine matrix is found using adaptive gradient descent (by elastix default) 
    with a loss function that somehow measures how well `input` matches reference.
    it will have the same size, orientation, etc, and the should be perfectly alligned."""
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    elastix.LogToConsoleOff()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(input)

    # set it to elastix filter and execute
    if pmap is not None: elastix.SetParameterMap(pmap)
    elastix.Execute()
    return elastix.GetResultImage()


def register_with(input:str | sitk.Image, other: str | sitk.Image | Any, reference: str | sitk.Image, pmap = default_pmap(), label=True) -> tuple[sitk.Image,sitk.Image]:
    """Register `input` to reference, then use that transformation to also register `other`, which is usually segmentation."""
    # load inputs
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(reference, str): reference = sitk.ReadImage(reference)

    if isinstance(other, str): other = sitk.ReadImage(other)
    # copy spatial info from input to the segmentation if it is just an array.
    elif not isinstance(other, sitk.Image):
        if isinstance(other, np.ndarray): other = sitk.GetImageFromArray(other)
        else: other = sitk.GetImageFromArray(other.numpy()) # torch tensor
        other.CopyInformation(input)

    # create elastix filter
    elastix = sitk.ElastixImageFilter()
    elastix.LogToConsoleOff()
    elastix.SetFixedImage(reference)
    elastix.SetMovingImage(input)

    # set it to elastix filter and execute
    if pmap is not None: elastix.SetParameterMap(pmap)
    input_reg = elastix.Execute()

    # create filter that will apply the trained elastix parameters to the segmentation
    transform = sitk.TransformixImageFilter()
    tmap = elastix.GetTransformParameterMap()

    # set nearest neighbour interpolation when registering segmentation
    if label:
        tmap[0]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        tmap[1]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        tmap[2]["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

    transform.SetTransformParameterMap(tmap)
    transform.SetMovingImage(other)
    transform.LogToConsoleOff()

    return input_reg, transform.Execute()



def register_t1_to_MNI152(input:str|sitk.Image, reference=MNI152) -> sitk.Image:
    """Register `input` to MNI. `input` must be path/sitk.Image of a T1 scan. Returns `input` registered to MNI."""
    return register_to(input, reference)

def register_t1_to_SRI24(input:str|sitk.Image, reference=SRI24) -> sitk.Image:
    """Register `input` to SRI24. `input` must be path/sitk.Image of a T1 scan. Returns `input` registered to SRI24."""
    return register_to(input, reference)

def register_imgs_to(
    template_input: str | sitk.Image,
    other_inputs: str | sitk.Image | Sequence[str | sitk.Image],
    reference: str | sitk.Image,
    _progress:Any=ProgressDummy(),
) -> list[sitk.Image]:
    """Register `template_input` to `reference`, then register `other_inputs` to registered `template_input`.

    Returns registered `[template_input, *other_inputs]`."""
    if not isinstance(other_inputs, Sequence): other_inputs = [other_inputs]

    _progress(0.1, "Совмещение T1 с атласом SRI24...")
    # Register `template_input` to `reference`
    registered_template_input = register_to(template_input, reference)

    _progress(0.2, "Совмещение T1CE, T2 и FLAIR с T1...")
    # register `other_inputs` to registered `template_input`.
    registered_other_inputs = [register_to(i, registered_template_input) for i in other_inputs]

    return [registered_template_input, *registered_other_inputs]

def register_imgs_to_MNI152(t1:str|sitk.Image, other: str|sitk.Image | Sequence[str|sitk.Image], reference=MNI152) -> list[sitk.Image]:
    return register_imgs_to(t1, other, reference)

def register_imgs_to_SRI24(t1:str|sitk.Image, other: str|sitk.Image | Sequence[str|sitk.Image], reference=SRI24, _progress:Any=ProgressDummy()) -> list[sitk.Image]:
    return register_imgs_to(t1, other, reference, _progress=_progress)