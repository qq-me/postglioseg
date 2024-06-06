import numpy as np
import torch
from torchvision.transforms import v2

from glio.data import DSToTarget, DSBase

MEAN_STD = [[10.0142], [45.6992]]
norm = v2.Normalize(*MEAN_STD) # type:ignore

def loader(x):
    return norm(torch.from_numpy(x[0].astype(np.float32)).unsqueeze(0)), torch.from_numpy(x[1].astype(np.float32)).permute(2,0,1)

def tfminput(x): return x[0]

def tfmtarget(x): return x[1]

def get_m2nist(path="D:/datasets/M2NIST", start=None, stop=None, loader_tfms = None):
    """Input of size (`1,  64, 84)`;

    target of size `(11, 64, 84)`, where 0-10 are digits, and 11 is background."""
    images = np.load(f"{path}/images.npy")
    masks = np.load(f"{path}/masks.npy")

    if isinstance(start, float): start = int(len(images)*start)
    if isinstance(stop, float): stop = int(len(images)*stop)

    if (start is not None) and (stop is not None):
        sl = slice(start, stop)
        images = images[sl].copy()
        masks = masks[sl].copy()

    if loader_tfms is not None: loader_fn = DSBase.smart_compose(loader, loader_tfms)
    else: loader_fn = loader

    ds = DSToTarget(1)
    ds.add_samples(list(zip(images,masks)), loader=loader_fn, transform_init=None, transform_sample=tfminput, transform_target=tfmtarget)
    ds.preload()
    return ds

def get_m2nist_traintest(path="D:/datasets/M2NIST", split=0.8, nsamples=None, loader_tfms = None):
    """Input of size (`1,  64, 84)`;

    target of size `(11, 64, 84)`, where 0-10 are digits, and 11 is background."""
    images = np.load(f"{path}/images.npy")
    masks = np.load(f"{path}/masks.npy")

    if isinstance(nsamples, float): nsamples = int(len(images)*nsamples)
    if nsamples is not None:
        images = images[:nsamples].copy()
        masks = masks[:nsamples].copy()

    if isinstance(split, float): split = int(len(images)*split)

    if loader_tfms is not None: loader_fn = DSBase.smart_compose(loader, loader_tfms)
    else: loader_fn = loader
    
    dstrain = DSToTarget(1)
    dstrain.add_samples(list(zip(images,masks))[:split], loader=loader_fn, transform_init=None, transform_sample=tfminput, transform_target=tfmtarget)
    dstrain.preload()

    dstest = DSToTarget(1)
    dstest.add_samples(list(zip(images,masks))[split:], loader=loader_fn, transform_init=None, transform_sample=tfminput, transform_target=tfmtarget)
    dstest.preload()

    return dstrain, dstest