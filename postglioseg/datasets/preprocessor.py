"""Siodj"""
import math
from typing import Sequence, Any
import polars as pl
import torch, numpy as np
from monai import transforms as mtf
import SimpleITK as sitk
from glio.loaders import niireadtensor
from glio.transforms import norm_to01, z_normalize_channels
from torchvision.transforms import v2


class _SitkT:
    def __init__(self, tfm):
        self.tfm = tfm
    def __call__(self, x):
        print('toarr...')
        if isinstance(x, torch.Tensor): x = np.asarray(x)
        print('toimgs...')
        imgs = [sitk.GetImageFromArray(i, isVector=False) for i in x]
        print('tfming...')
        tfmed = [self.tfm(i) for i in imgs]
        print('fromarr...')
        arrays = [sitk.GetArrayFromImage(i) for i in tfmed]
        print('totensor...')
        res = torch.from_numpy(np.asarray(arrays))
        print("done")
        return res

def _measure_noise(image):
    """Measures std of the front top left 5% of the image"""
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D, but it is {image.ndim}D")
    image = norm_to01(image.to(torch.float32))
    shape = image.shape
    corner = image[:int(shape[0]/20), :int(shape[1]/20), :int(shape[2]/20)]
    return norm_to01(corner).std()

def _is_noisy(image):
    """Measures std of the front top left 5% of the image, returns True if above 0.05"""
    return _measure_noise(image) > 0.05


def _load_if_needed(path_or_arr: torch.Tensor | np.ndarray | str, dtype=torch.float32):
    if isinstance(path_or_arr, str):
        return niireadtensor(path_or_arr)
    elif isinstance(path_or_arr, np.ndarray):
        return torch.as_tensor(path_or_arr, dtype)
    elif path_or_arr.dtype != dtype:
        return path_or_arr.to(dtype)
    elif not isinstance(path_or_arr, torch.Tensor):
        raise TypeError(f"Unknown type {type(path_or_arr)}")
    return path_or_arr

def _hist_correction(img: torch.Tensor):
    corrector = mtf.HistogramNormalize()# type:ignore
    img = torch.cat([corrector(i.unsqueeze(0)) for i in img], 0) # type:ignore
    return img

class Preprocessor:
    def __init__(self,
                crop = mtf.CropForeground(select_fn=lambda x: x>0.5, allow_smaller=False), # type:ignore
                rotate = None,
                size = (140,180,140),
                zoom_mode = "trilinear",
                zoom_mode_seg = "nearest-exact",
                hist_correction = _hist_correction, # type:ignore
                z_norm = z_normalize_channels,
                post_img_tfms = None,
                post_seg_tfms = None,
                 ):
        self.crop = crop
        self.rotate = rotate
        self.size = size
        self.ratios = [int(i / math.gcd(*size)) for i in size]
        #print(self.ratios)
        self.zoom_mode = zoom_mode
        self.zoom_mode_seg = zoom_mode_seg
        self.hist_correction = hist_correction
        self.z_norm = z_norm
        self.post_img_tfms = post_img_tfms
        self.post_seg_tfms = post_seg_tfms


    def __call__(self, *imgs, seg = None, return_nohist=False) ->tuple | Any:
        """With `return_nohist = True`, returns `imgs, imgs_hist, seg`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        if self.hist_correction is None: return_nohist = False

        # load all images
        imgs = [norm_to01(_load_if_needed(m)) for m in imgs]
        if seg is not None: seg = _load_if_needed(seg)

        # create a stack of modalities + seg for cropping
        if seg is not None: stacked = torch.stack(imgs + [seg], dim=0)  # 5, 155, 240, 240 #type:ignore
        else: stacked = torch.stack(imgs, dim=0)  # 4, 155, 240, 240

        # crop foreground by selecting values above 0.5, as all images are normalized to 0-1 range
        if self.crop is not None:
            stacked = self.crop(stacked)

        # rotate if needed
        if self.rotate is not None:
            stacked = self.rotate(stacked)

        # pad to some common ratio (7,9,7)
        # so all sizes need to be divisible by 7,9,7 to pad evenly
        #stacked = stacked[:, stacked.shape[1] - stacked.shape[1] % self.size[0]]
        pad = mtf.DivisiblePad(self.ratios, mode = "constant", constant_values=0)# type:ignore
        stacked = pad(stacked)

        # unstack images and seg
        if seg is not None:
            imgs, seg = stacked[:-1], stacked[-1].unsqueeze(0)
        else: imgs = stacked

        # zoom to common size
        # calculate the zoom factor using first size
        zoom_factor = self.size[0] / imgs.shape[1]
        # construct the zoom
        zoom = mtf.Zoom(zoom_factor, mode=self.zoom_mode) # type:ignore
        zoom_seg = mtf.Zoom(zoom_factor, mode=self.zoom_mode_seg) # type:ignore
        # apply the zoom
        imgs = zoom(imgs)
        if seg is not None: seg = zoom_seg(seg)


        # make sure the shape is correct
        pad_crop_resizer = mtf.ResizeWithPadOrCrop(self.size, mode = "constant", constant_values=0) # type:ignore
        imgs = pad_crop_resizer(imgs)[:,:self.size[0], :self.size[1], :self.size[2]]
        if seg is not None: seg = pad_crop_resizer(seg)[:, :self.size[0], :self.size[1], :self.size[2]]


        # histogram correction
        if self.hist_correction is not None:
            cor_imgs = self.hist_correction(imgs)
        else: cor_imgs = imgs

        # z-normalize
        if self.z_norm is not None:
            cor_imgs = self.z_norm(cor_imgs)
            if return_nohist: imgs = self.z_norm(imgs)

        # custom tfms
        if self.post_img_tfms is not None:
            cor_imgs = self.post_img_tfms(cor_imgs)
            if return_nohist: imgs = self.post_img_tfms(imgs)
        if self.post_seg_tfms is not None and seg is not None:
            seg = self.post_seg_tfms(seg)

        if seg is not None:
            if return_nohist: return imgs, cor_imgs, seg[0]
            else: return imgs, seg[0]
        else:
            if return_nohist: return imgs, cor_imgs
            else: return imgs