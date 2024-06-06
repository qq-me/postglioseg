from typing import Any
import random
import torch
import joblib
from glio.torch_tools import one_hot_mask
from glio.python_tools import SliceContainer, reduce_dim

PATH = r"E:\dataset\BRaTS2024-GoAT"
BRATS2024_HIST96_TRAIN = rf"{PATH}/brats2024 hist96 train.joblib"
BRATS2024_HIST96_TEST = rf"{PATH}/brats2024 hist96 test.joblib"

def get_ds_2d(path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ds:list[list[tuple[SliceContainer,SliceContainer]]] = joblib.load(path)
    return [(i[0](), i[1]()) for j in ds for i in j]

def loader_2d(sample:tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return sample[0].to(torch.float32), one_hot_mask(sample[1], 4)

def get_ds_around(path, around=1) -> list[tuple[list[torch.Tensor],torch.Tensor]]:
    ds:list[list[tuple[SliceContainer,SliceContainer]]] = joblib.load(path)
    res = []
    for slices in ds:
        for i in range(around, len(slices) - around):
            stack = slices[i - around : i + around + 1]
            images = [s[0]() for s in stack]
            seg = slices[i][1]()
            res.append((images, seg))
    return res

def loader_around(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(sample[0], 0).to(torch.float32), one_hot_mask(sample[1], 4)

def loader_around_fix(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(sample[0], 0).to(torch.float32)[:,:96,:96], one_hot_mask(sample[1], 4)[:,:96,:96]

def loader_around_seq(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack(reduce_dim(list(zip(*sample[0]))), 0).to(torch.float32), one_hot_mask(sample[1], 4)

def loader_around_seq_fix(sample:tuple[list[torch.Tensor],torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack(reduce_dim(list(zip(*sample[0]))), 0).to(torch.float32)[:,:96,:96], one_hot_mask(sample[1], 4)[:,:96,:96]

def randcrop(x: tuple[torch.Tensor, torch.Tensor], size = (96,96)):
    if x[0].shape[1] == size[0] and x[0].shape[2] == size[1]: return x
    startx = random.randint(0, (x[0].shape[1] - size[0]) - 1)
    starty = random.randint(0, (x[0].shape[2] - size[1]) - 1)
    return x[0][:, startx:startx+size[0], starty:starty+size[1]], x[1][:, startx:startx+size[0], starty:starty+size[1]]

def shuffle_channels(x:torch.Tensor):
    return x[torch.randperm(x.shape[0])]

def shuffle_channels_around(x:torch.Tensor, channels_per = 3):
    num_groups = int(x.shape[0] / channels_per)
    perm = torch.randperm(num_groups, dtype=torch.int32)
    img= x.reshape(num_groups, channels_per, *x.shape[1:])[perm].flatten(0, 1)
    return img