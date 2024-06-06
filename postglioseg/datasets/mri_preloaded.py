from typing import Any
import random
import torch
import joblib
from glio.torch_tools import one_hot_mask
from glio.python_tools import SliceContainer, reduce_dim, Compose

# RHUH_PATH = r"E:\dataset\RHUH-GBM"
# RHUH_HIST140_TRAIN = rf"{RHUH_PATH}/rhuh hist140 train.joblib"
# RHUH_HIST140_TEST = rf"{RHUH_PATH}/rhuh hist140 test.joblib"
# RHUH_NOHIST140_TRAIN = rf"{RHUH_PATH}/rhuh nohist140 train.joblib"
# RHUH_NOHIST140_TEST = rf"{RHUH_PATH}/rhuh nohist140 test.joblib"

# RHUH_HIST140_NOADC_TRAIN = rf"{RHUH_PATH}/rhuh hist140 noadc train.joblib"
# RHUH_HIST140_NOADC_TEST = rf"{RHUH_PATH}/rhuh hist140 noadc test.joblib"
# RHUH_NOHIST140_NOADC_TRAIN = rf"{RHUH_PATH}/rhuh nohist140 noadc train.joblib"
# RHUH_NOHIST140_NOADC_TEST = rf"{RHUH_PATH}/rhuh nohist140 noadc test.joblib"

# BRATS_PATH = r"E:\dataset\BRaTS2024-GoAT"
# BRATS2024_NOHIST96_TRAIN = rf"{BRATS_PATH}/brats2024 nohist96 train.joblib"
# BRATS2024_NOHIST96_TEST = rf"{BRATS_PATH}/brats2024 nohist96 test.joblib"
# BRATS2024SMALL_HIST96_TRAIN = rf"{BRATS_PATH}/brats2024-small hist train.joblib"
# BRATS2024SMALL_HIST96_TEST = rf"{BRATS_PATH}/brats2024-small hist test.joblib"
# BRATS2024SMALL_NOHIST96_TRAIN = rf"{BRATS_PATH}/brats2024-small nohist train.joblib"
# BRATS2024SMALL_NOHIST96_TEST = rf"{BRATS_PATH}/brats2024-small nohist test.joblib"

RHUH_TRAIN = r"E:\dataset\RHUH-GBM\rhuh full v2 train.joblib"
RHUH_TEST = r"E:\dataset\RHUH-GBM\rhuh full v2 test.joblib"

BRATS_TRAIN = r"E:\dataset\BRaTS2024-GoAT\brats2024-96 v2 train.joblib"
BRATS_TEST = r"E:\dataset\BRaTS2024-GoAT\brats2024-96 v2 test.joblib"

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
    #print(x[0].shape)
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

def groupwise_tfms(x:torch.Tensor, tfms, channels_per = 3):
    num_groups = int(x.shape[0] / channels_per)
    groups = x.reshape(num_groups, channels_per, *x.shape[1:]).unbind(0)
    groups = [Compose(tfms)(i) for i in groups]
    return torch.cat(groups, 0)


class GroupwiseTfms:
    def __init__(self, tfms, channels_per = 3):
        self.tfms = tfms
        self.channels_per = channels_per
    def __call__(self, x):
        return groupwise_tfms(x, self.tfms, self.channels_per)