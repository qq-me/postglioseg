"""PostGlioSeg, автор - Никишев Иван Олегович"""
from typing import Any, Optional
import logging, os
import torch
import numpy as np
import SimpleITK as sitk
from .dicom_to_nifti import dicom2sitk
from .registration import register_imgs_to_SRI24, register_with, resample_to
from .skullstrip import skullstrip_imgs
from .normalize import znormalize_imgs
from .crop_bg import crop_bg_imgs

class ProgressDummy:
    def __call__(self, *args, **kwargs): pass

def _toImage(x) -> sitk.Image:
    if isinstance(x, sitk.Image): return x
    elif os.path.isfile(x): return sitk.ReadImage(x)
    elif os.path.isdir(x): return dicom2sitk(x)
    else: raise ValueError("Неверный тип данных")

def pipeline(t1:str|sitk.Image, t1ce:str|sitk.Image, flair:str|sitk.Image, t2w:str|sitk.Image,
             register=True, skullstrip=True, erode=1, cropbg=True, _progress = ProgressDummy()) -> tuple[list[sitk.Image],list[sitk.Image]]:
    _progress(0.02, "Загрузка Т1")
    t1_orig = _toImage(t1)
    _progress(0.04, "Загрузка Т1CE")
    t1ce_orig = _toImage(t1ce)
    _progress(0.06, "Загрузка FLAIR")
    flair_orig = _toImage(flair)
    _progress(0.08, "Загрузка T2")
    t2w_orig = _toImage(t2w)

    logging.info("Регистрация модальностей в SRI24")
    if register: t1_sri, t1ce_sri, flair_sri, t2w_sri = register_imgs_to_SRI24(t1_orig, (t1ce_orig, flair_orig, t2w_orig), _progress=_progress)
    else: t1_sri, t1ce_sri, flair_sri, t2w_sri = t1_orig, t1ce_orig, flair_orig, t2w_orig

    logging.info("Удаление черепа")
    if skullstrip:
        t1ce_skullstrip, t1_skullstrip, flair_skullstrip, t2w_skullstrip = (
            skullstrip_imgs(t1ce_sri, (t1_sri, flair_sri, t2w_sri), erode=erode, _progress=_progress)
        )
    else: t1ce_skullstrip, t1_skullstrip, flair_skullstrip, t2w_skullstrip = t1ce_sri, t1_sri, flair_sri, t2w_sri

    logging.info("Нормализация")


    #if cropbg: return [t1_sri, t1ce_sri, flair_sri, t2w_sri], crop_bg_imgs(norm)
    #else: return [t1_sri, t1ce_sri, flair_sri, t2w_sri, t2w_skullstrip], norm

    if cropbg:
        t1ce_crop, t1_crop, flair_crop, t2w_crop = crop_bg_imgs((t1ce_skullstrip, t1_skullstrip,flair_skullstrip, t2w_skullstrip))
        return [t1_sri, t1ce_sri, flair_sri, t2w_sri], znormalize_imgs((t1_crop, t1ce_crop, flair_crop, t2w_crop))
    else: return [t1_sri, t1ce_sri, flair_sri, t2w_sri, t2w_skullstrip], znormalize_imgs((t1_skullstrip,t1ce_skullstrip,flair_skullstrip, t2w_skullstrip))


class Pipeline:
    def __init__(self, t1:str|sitk.Image, t1ce:str|sitk.Image, flair:str|sitk.Image, t2w:str|sitk.Image):
        """Stores native images and images before skullstripping."""
        self.t1_native, self.t1ce_native, self.flair_native, self.t2w_native = [_toImage(i) for i in (t1,t1ce,flair,t2w)]

    def preprocess(self, register=True, skullstrip=True, erode=1, cropbg=True, _progress:Any=ProgressDummy()) -> torch.Tensor:
        """Returns 4 channel torch tensor with (T1, T1CE, FLAIR, T2w) channels, registered to SRI24, skullstripped, and with cropped background."""
        (
            (self.t1_sri, self.t1ce_sri, self.flair_sri, self.t2w_sri),
            (self.t1_final, self.t1ce_final, self.flair_final, self.t2w_final),
        ) = pipeline(
            t1=self.t1_native,
            t1ce=self.t1ce_native,
            flair=self.flair_native,
            t2w=self.t2w_native,
            register=register,
            skullstrip=skullstrip,
            erode=erode,
            cropbg=cropbg,
            _progress=_progress,
        )

        return torch.from_numpy(np.stack(
            [sitk.GetArrayFromImage(i) for i in (self.t1_final, self.t1ce_final, self.flair_final, self.t2w_final)]
            ).astype(np.float32))

    def save(self, path, mkdirs=True):
        if mkdirs: os.makedirs(path, exist_ok=True)

        sitk.WriteImage(self.t1_native, os.path.join(path, 't1_native.nii.gz'))
        sitk.WriteImage(self.t1ce_native, os.path.join(path, 't1ce_native.nii.gz'))
        sitk.WriteImage(self.flair_native, os.path.join(path, 'flair_native.nii.gz'))
        sitk.WriteImage(self.t2w_native, os.path.join(path, 't2w_native.nii.gz'))

        sitk.WriteImage(self.t1_sri, os.path.join(path, 't1_sri.nii.gz'))
        sitk.WriteImage(self.t1ce_sri, os.path.join(path, 't1ce_sri.nii.gz'))
        sitk.WriteImage(self.flair_sri, os.path.join(path, 'flair_sri.nii.gz'))
        sitk.WriteImage(self.t2w_sri, os.path.join(path, 't2w_sri.nii.gz'))

        sitk.WriteImage(self.t1_final, os.path.join(path, 't1_final.nii.gz'))
        sitk.WriteImage(self.t1ce_final, os.path.join(path, 't1ce_final.nii.gz'))
        sitk.WriteImage(self.flair_final, os.path.join(path, 'flair_final.nii.gz'))
        sitk.WriteImage(self.t2w_final, os.path.join(path, 't2w_final.nii.gz'))

        if hasattr(self, 'seg'):
            sitk.WriteImage(self.seg, os.path.join(path, 'seg.nii.gz'))
            sitk.WriteImage(self.seg_native, os.path.join(path, 'seg_native.nii.gz'))

    def _load_seg(self, seg:str|sitk.Image|np.ndarray|torch.Tensor):
        if isinstance(seg, torch.Tensor): seg = seg.detach().cpu().numpy()
        if isinstance(seg, np.ndarray): seg = sitk.GetImageFromArray(seg)
        else: seg = _toImage(seg)

        self.seg = seg

    def postprocess(self, seg:str|sitk.Image|np.ndarray|torch.Tensor, to = 't1') -> sitk.Image:
        """Returns segmentation registered to native input."""
        self._load_seg(seg)
        self.seg.CopyInformation(self.t1ce_final)

        to = to.lower()
        self.seg = resample_to(self.seg, self.t1ce_sri)
        self.seg.CopyInformation(self.t1ce_sri)

        if to == 't1': unreg, self.seg_native = register_with(self.t1_sri, self.seg, self.t1_native)
        elif to == 't1ce': unreg, self.seg_native = register_with(self.t1ce_sri, self.seg, self.t1ce_native)
        elif to == 'flair': unreg, self.seg_native = register_with(self.flair_sri, self.seg, self.flair_native)
        elif to == 't2w' or to == 't2': unreg, self.seg_native = register_with(self.t2w_sri, self.seg, self.t2w_native)
        else: raise ValueError(f"Unknown target {to}")

        # if not os.path.exists(r'F:\Stuff\Programming\AI\vkr\testx'): os.mkdir(r'F:\Stuff\Programming\AI\vkr\testx')
        # import random
        # sitk.WriteImage(unreg, os.path.join(r'F:\Stuff\Programming\AI\vkr\testx', f'unreg{random.random()}.nii.gz'))
        return self.seg_native

    def save_segmentation_jpegs(self, outdir, seg:Optional[str|sitk.Image|np.ndarray|torch.Tensor] =None):
        """Save slice segmentation visualizations as JPEGs."""
        if seg is not None: self._load_seg(seg)

        from ..inference import save_slices
        save_slices((self.t1_sri, self.t1ce_sri, self.flair_sri, self.t2w_sri), self.seg, outdir=outdir)

