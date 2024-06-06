"""PostGlioSeg, автор - Никишев Иван Олегович"""
from typing import Any
from PIL import Image
import SimpleITK as sitk
import torch
import numpy as np

def sliding_inference_neighbouring(inputs:torch.Tensor, inferer, size, step, around, nlabels, expand = None):
    """Input must be a 4D C* or 5D BC* tensor. Unbatched sliding inference, VERY SLOW!"""
    if inputs.ndim == 4: inputs = inputs.unsqueeze(0)
    results = torch.zeros((inputs.shape[0], nlabels, *inputs.shape[2:]), device=inputs.device,)
    counts = torch.zeros_like(results)
    for x in range(around, inputs.shape[2]-around, 1):
        for y in range(0, inputs.shape[3]-size[0], step):
            for z in range(0, inputs.shape[4]-size[1], step):
                #print(x, y, z, end='    \r')
                # get current slice and neighboring slices
                inputs_slice = inputs[:, :, x-1:x+around+1, y:y+size[0], z:z+size[1]].flatten(1,2)
                if expand: inputs_slice = torch.cat((inputs_slice, torch.zeros((1, expand - inputs_slice.shape[1], *inputs_slice.shape[2:]))), dim=1)
                preds = inferer(inputs_slice)
                results[:, :, x, y:y+size[0], z:z+size[1]] += preds
                counts[:, :, x, y:y+size[0], z:z+size[1]] += 1

    results /= counts
    return results.nan_to_num(0)



def sliding_inference_neighbouring_batched(inputs:torch.Tensor, inferer, size, step, nlabels, expand = None):
    """Input must be a 4D C* tensor, around is 1. Batched sliding inference."""
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension

    # since input is 3 neighbouring slices, this creates a new dimension and flattens it so that each slice contains 12 channels,
    # each channel has 4 modalities and 3 neighbouring slices per modality.
    inputs = torch.stack(( inputs[:-2], inputs[1:-1],inputs[2:]), 2).flatten(1,2)

    # we store results and counts of how many predictions were made in each pixel
    results = torch.zeros((inputs.shape[0], nlabels, *inputs.shape[2:]), device=inputs.device,)
    counts = torch.zeros_like(results)

    for x in range(0, inputs.shape[2]-size[0], step):
        for y in range(0, inputs.shape[3]-size[1], step):
            #print(f'{x}/{inputs.shape[2]-size[0]}, {y}/{inputs.shape[3]-size[1]}', end='    \r')
            inputs_slice = inputs[:, :, x:x+size[0], y:y+size[1]]
            if expand: inputs_slice = torch.cat((inputs_slice, torch.zeros((inputs_slice.shape[0], expand - inputs_slice.shape[1], *inputs_slice.shape[2:]))), dim=1)
            preds = inferer(inputs_slice)
            results[:, :, x:x+size[0], y:y+size[1]] += preds
            counts[:, :, x:x+size[0], y:y+size[1]] += 1

    results /= counts

    # add 1 pixel padding to each side of the first spatial dimension restore the original shape
    padding = torch.zeros((1, *results.shape[1:],))
    results = torch.cat((padding, results, padding))

    # return C* tensor
    return results.nan_to_num(0).swapaxes(0,1)

def sliding_inference_neighbouring_batched_gaussian(inputs:torch.Tensor, inferer, size, overlap=0.75, expand = None, progress=False):
    """Input must be a 4D C* tensor, around is 1. Sliding inference using gaussian overlapping."""
    from monai.inferers import SlidingWindowInferer # type:ignore
    inputs = inputs.swapaxes(0,1) # First spatial dimension becomes batch dimension

    # input is 3 neighbouring slices, this creates a new dimension and flattens it so that each slice contains 12 channels,
    # each channel has 4 modalities and 3 neighbouring slices per modality.
    # this also makes the input smaller by 1 pixel on each side of the first spatial dimension
    inputs = torch.stack(( inputs[:-2], inputs[1:-1],inputs[2:]), 2).flatten(1,2)
    if expand: inputs = torch.cat((inputs, torch.zeros((inputs.shape[0], expand-inputs.shape[1], *inputs.shape[2:]))), 1)
    sliding = SlidingWindowInferer(size, 32, overlap, mode='gaussian', progress=progress)

    results = sliding(inputs, inferer).cpu() # type:ignore

    # add 1 pixel padding to each side of the first spatial dimension restore the original shape
    padding = torch.zeros((1, *results.shape[1:],)) # type:ignore
    results = torch.cat((padding, results, padding)) # type:ignore

    # return C* tensor
    return results.swapaxes(0,1) # type:ignore

class ProgressDummy:
    def __call__(self, *args, **kwargs): pass

def _tta_separate(
    inputs: torch.Tensor,
    inferer,
    size,
    expand=None,
    ttaid=(0, 1, 2, 3, 4, 5, 6, 7),
    inferer_fn=sliding_inference_neighbouring_batched_gaussian,
    _progress:Any = ProgressDummy(),
    **inferer_kwargs,
):
    """Input must be a 4D C* tensor. Does test time augmentation by rotating the inputs, returns all results as a tuple."""
    results = []
    _progress(0.5, "Сегментация...")
    if 0 in ttaid: results.append(inferer_fn(inputs, inferer, size=size,  expand=expand, **inferer_kwargs))
    _progress(0.53, "Сегментация...")
    if 1 in ttaid: results.append(inferer_fn(inputs.flip(-1), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-1))
    _progress(0.56, "Сегментация...")
    if 2 in ttaid: results.append(inferer_fn(inputs.flip(-2), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-2))
    _progress(0.59, "Сегментация...")
    if 3 in ttaid: results.append(inferer_fn(inputs.flip((-1, -2)), inferer, size=size,  expand=expand, **inferer_kwargs).flip((-1, -2)))

    inputs2 = inputs.swapaxes(-1, -2)
    _progress(0.62, "Сегментация...")
    if 4 in ttaid: results.append(inferer_fn(inputs2, inferer, size=size, expand=expand, **inferer_kwargs).swapaxes(-1, -2))
    _progress(0.65, "Сегментация...")
    if 5 in ttaid: results.append(inferer_fn(inputs2.flip(-1), inferer, size=size, expand=expand, **inferer_kwargs).flip(-1).swapaxes(-1, -2))
    _progress(0.68, "Сегментация...")
    if 6 in ttaid: results.append(inferer_fn(inputs2.flip(-2), inferer, size=size,  expand=expand, **inferer_kwargs).flip(-2).swapaxes(-1, -2))
    _progress(0.71, "Сегментация...")
    if 7 in ttaid: results.append(inferer_fn(inputs2.flip((-1, -2)), inferer, size=size,  expand=expand, **inferer_kwargs).flip((-1, -2)).swapaxes(-1, -2))

    return results


def tta(
    inputs: torch.Tensor,
    inferer,
    size,
    expand=None,
    ttaid=(0, 1, 2, 3, 4, 5, 6, 7),
    inferer_fn=sliding_inference_neighbouring_batched_gaussian,
    _progress:Any = ProgressDummy(),
    **inferer_kwargs,
):
    """Input must be a 4D C* tensor. Does test time augmentation by rotating the inputs."""
    results = _tta_separate(inputs=inputs, inferer=inferer, size=size, expand=expand, ttaid=ttaid, inferer_fn=inferer_fn, _progress=_progress, **inferer_kwargs)
    return torch.stack((results)).sum(0)

def niiread(path:str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def niireadtensor(path:str):
    arr = niiread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int32)
    return torch.from_numpy(arr)

def _load_if_needed(path_or_arr: torch.Tensor | np.ndarray | sitk.Image | str, dtype=torch.float32):
    if isinstance(path_or_arr, str): return niireadtensor(path_or_arr)
    elif isinstance(path_or_arr, np.ndarray): return torch.as_tensor(path_or_arr, dtype=dtype)
    elif isinstance(path_or_arr, sitk.Image): return torch.as_tensor(sitk.GetArrayFromImage(path_or_arr), dtype=dtype)
    elif path_or_arr.dtype != dtype: return path_or_arr.to(dtype)
    elif not isinstance(path_or_arr, torch.Tensor): raise TypeError(f"Unknown type {type(path_or_arr)}")
    return path_or_arr

def norm(x:torch.Tensor | np.ndarray, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]`"""
    x -= x.min()
    if x.max() != 0: x /= x.max()
    else: return x
    return x * (max - min) + min

def save_slices(imgs, seg, outdir, labels = ("T1", "T1CE", "T2-FLAIR", "T2w"), real_seg=None, mul=2., permute=None, flip=None):
    """Save slice segmentation visualizations as JPEGs."""
    imgs = [_load_if_needed(i) for i in imgs]
    seg = _load_if_needed(seg)
    # argmax seg if in one hot
    if seg.ndim == 4: seg = seg.argmax(0)
    if real_seg is not None:
        real_seg = _load_if_needed(real_seg)
        if real_seg.ndim == 4: real_seg = real_seg.argmax(0)

    # input tensor
    inputs = norm(torch.stack(imgs))
    # preview tensor
    preview_seg = norm(torch.stack((inputs,inputs,inputs))) # 34XYZ / CMXYZ # type:ignore

    if permute is not None:
        preview_seg = preview_seg.permute(0, 1, *[i+2 for i in permute]) # type:ignore
        seg = seg.permute(permute)
        if real_seg is not None:  real_seg = real_seg.permute(permute)

    if flip is not None:
        preview_seg = preview_seg.flip(0, 1, *[i+2 for i in flip]) # type:ignore
        seg = seg.flip(flip)
        if real_seg is not None:  real_seg = real_seg.flip(flip)

    preview_no_seg = preview_seg.clone() # type:ignore
    if real_seg is not None: preview_real_seg = preview_seg.clone() # type:ignore

    # overlay segmentation on each channel of the input
    preview_seg[0] = torch.where(seg == 3, preview_seg[0]*mul, preview_seg[0]).clip(0,1) # tumor - red
    preview_seg[1] = torch.where(seg == 2, preview_seg[1]*mul, preview_seg[2]).clip(0,1) # edema - green
    preview_seg[2] = torch.where(seg == 1, preview_seg[2]*mul, preview_seg[2]).clip(0,1) # necrosis - blue

    if real_seg is not None:
        preview_real_seg[0] = torch.where(real_seg == 3, preview_real_seg[0]*mul, preview_real_seg[0]).clip(0,1)# type:ignore
        preview_real_seg[1] = torch.where(real_seg == 2, preview_real_seg[1]*mul, preview_real_seg[2]).clip(0,1) # type:ignore
        preview_real_seg[2] = torch.where(real_seg == 1, preview_real_seg[2]*mul, preview_real_seg[2]).clip(0,1)# type:ignore

    if real_seg is None: preview_cat = torch.cat((preview_seg, preview_no_seg), 4).permute(1,2,3,4,0).numpy() # 4XYZC / MXYZC # type:ignore
    else: preview_cat = torch.cat((preview_seg, preview_real_seg, preview_no_seg), 4).permute(1,2,3,4,0).numpy() # 4XYZC / MXYZC # type:ignore

    # save slice visualizeations
    for mod, modname in zip(preview_cat, labels):
        for i, sl in enumerate(mod):
            im = Image.fromarray(norm(sl, 0, 255).astype(np.uint8)) # type:ignore
            im.save(f'{outdir}/{i}_{modname}.png')
