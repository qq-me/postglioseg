"""PostGlioSeg, автор - Никишев Иван Олегович"""
from typing import Any
from collections.abc import Sequence
import os, subprocess, shutil
import tempfile
import SimpleITK as sitk

def get_brain_mask(input:str | sitk.Image, mode='accurate', do_tta=True) -> sitk.Image:
    """Runs skullstripping using HD BET (https://github.com/MIC-DKFZ/HD-BET). Requires it to be installed.

    Returns `outpath` which is path to brain mask for convenience.

    Args:
        inputs (str | sitk.Image): Path to a nifti file or a sitk.Image of the image to generate brain mask from, all inputs must be in MNI152 space.
        outpaths (str): Path to output file, must include `.nii.gz`.
        mkdirs (bool, optional): Whether to create `outfolder` if it doesn't exist, otherwise throws an error. Defaults to True.
    """
    from HD_BET.run import run_hd_bet

    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as temp2:

        if isinstance(input, sitk.Image): sitk.WriteImage(input, os.path.join(temp, 't1.nii.gz'))
        else: shutil.copyfile(input, os.path.join(temp, 't1.nii.gz'))

        # run skullstripping
        run_hd_bet(os.path.join(temp, 't1.nii.gz'), os.path.join(temp2, 't1.nii.gz'), mode=mode, do_tta=do_tta)

        # return sitk image
        #print(os.listdir(temp2))
        return sitk.ReadImage(os.path.join(temp2, 't1_mask.nii.gz'))

def apply_brain_mask(input:str | sitk.Image, mask:str | sitk.Image) -> sitk.Image:
    """Applies brain mask to input image.

    Args:
        input (str | sitk.Image): Path to a nifti file or a sitk.Image of the image to generate brain mask from, all inputs must be in MNI152 space.
        mask (str | sitk.Image): Path to a nifti file or a sitk.Image of the brain mask to apply to the input image.
    """
    if isinstance(input, str): input = sitk.ReadImage(input)
    if isinstance(mask, str): mask = sitk.ReadImage(mask)
    mask = sitk.Cast(mask, sitk.sitkFloat32)
    return sitk.Multiply(input, mask)


def skullstrip(input:str | sitk.Image) -> sitk.Image:
    mask = get_brain_mask(input)
    return apply_brain_mask(input, mask)

class ProgressDummy:
    def __call__(self, *args, **kwargs): pass
    
def skullstrip_imgs(template:str | sitk.Image, other: str | sitk.Image | Sequence[str | sitk.Image], erode=0, _progress:Any=ProgressDummy()) -> Sequence[sitk.Image]:
    if not isinstance(other, Sequence): other = [other]
    _progress(0.4, "Удаление черепа...")
    mask = get_brain_mask(template)
    if erode > 0: mask = sitk.BinaryErode(mask, [erode, erode, erode])
    _progress(0.48, "Применение маски черепа...")
    return [apply_brain_mask(i, mask) for i in [template, *other]]