"""PostGlioSeg, автор - Никишев Иван Олегович"""
import os
import random
import torch, numpy as np
from torch import nn
from monai.networks.nets import HighResNet, SegResNetDS # type:ignore
from monai import transforms as mtf

from ..python_tools import Compose

def _prob(p:float): return random.random() < p

def rand_shuffle_channels_around(x:torch.Tensor, channels_per = 3):
    """50% to shuffle groups of `channels_per` channels."""
    if random.random() < 0.5:
        num_groups = int(x.shape[0] / channels_per)
        perm = torch.randperm(num_groups, dtype=torch.int32)
        img= x.reshape(num_groups, channels_per, *x.shape[1:])[perm].flatten(0, 1)
        return img
    return x

def rand_dropout(x):
    """Should be applied groupwise, for each group, 20% to drop and 1% to replace with gaussian noise."""
    if random.random() < 0.2: return torch.zeros_like(x)
    if random.random() < 0.01: return torch.randn_like(x)
    return x

def randnoise(x):
    """Should be applied groupwise, 10% to add gaussian noise."""
    if random.random() < 0.1: return x + torch.randn_like(x) * random.triangular(0, 0.5, 0)
    return x


def groupwise_tfms(x:torch.Tensor, tfms, channels_per = 3):
    """Apply `tfms` to groups of `channels_per` channels."""
    num_groups = int(x.shape[0] / channels_per)
    groups = x.reshape(num_groups, channels_per, *x.shape[1:]).unbind(0)
    groups = [Compose(tfms)(i) for i in groups]
    return torch.cat(groups, 0)

class GroupwiseTfms:
    def __init__(self, tfms, channels_per = 3):
        """Apply `tfms` to groups of `channels_per` channels."""
        self.tfms = tfms
        self.channels_per = channels_per
    def __call__(self, x):
        return groupwise_tfms(x, self.tfms, self.channels_per)

def rand_shift(x:torch.Tensor | np.ndarray, val = (-1., 1.)):
    """Shift the input by a random amount. """
    return x + random.uniform(*val)

class RandShift:
    def __init__(self, val = (-1., 1.), p=0.1):
        """Shift the input by a random amount. """
        self.val = val
        self.p = p
    def __call__(self, x):
        if _prob(self.p): return rand_shift(x, self.val)
        return x

def rand_scale(x:torch.Tensor | np.ndarray, val = (0.5, 2)):
    """Scale the input by a random amount."""
    return x * random.uniform(*val)

class RandScale:
    def __init__(self, val = (0.5, 2), p=0.1):
        """Scale the input by a random amount. """
        self.val = val
        self.p = p
    def __call__(self, x):
        if _prob(self.p): return rand_scale(x, self.val)
        return x

single_tfms = Compose(
    rand_shuffle_channels_around, # 50% to shuffle modality groups
    GroupwiseTfms(
        (
            randnoise, # 10% per group to add gaussian noise
            RandScale(p=0.3, val=(0.75, 1.5)), # 30% per group to scale
            RandShift(p=0.3, val=(-0.3, 0.3)), # 30% per group to shift
            rand_dropout, # 20% per group to drop it (replace with 0s)
        )
    ),
)

batch_tfms = torch.vmap(single_tfms, randomness="different") # batched transforms

mse = torch.nn.MSELoss() # loss for pretraining contextnet

def get_contextnet(in_channels=12, out_channels=20):
    """A sequence of residual blocks that never changes the size of the input, only channel num."""
    model = HighResNet(2, in_channels, out_channels)
    model.blocks[-1].adn = nn.Identity()
    return model

class ContextNet(nn.Module):
    def __init__(self, pretraining = False):
        """In `pretraining`, this is trained to reconstruct the input from the input as first 12 channels, 
        the segmentation as 12-16, and last 4 channels are left untrained so that it can use them later."""
        super().__init__()
        self.pretraining = pretraining
        self.batch_tfms = batch_tfms
        self.hrn = get_contextnet(12, 20)

    def forward(self, x:torch.Tensor):
        # in pre-training mode, corrupt the input with `batch_tfms` and it has to recreate original input,
        # so that it is more robust to new data acquired with different parameters
        # as well as return 12-16 channels as segmentation.
        if self.training and self.pretraining:
            # apply the random transforms e.g. noise, shuffle, dropout
            with torch.no_grad(): self.tfmed = self.batch_tfms(x)
            # pass the transformed input to the contextnet
            self.processed = self.hrn(self.tfmed)
            # calculate the reconstruction MSE loss from first 12 channels (this must be added to final loss)
            loss = mse(self.processed[:, :12], x)
            # return the 12-16 channels to be compared with `y` using dice focal loss.
            return self.processed[:,12:16], loss

        # in eval mode pass original `x` and return segmentation part
        elif self.pretraining:
            self.processed = self.hrn(x)
            return self.processed[:,12:16]

        # not pretraining, only pass original input and return full output for concatenation
        else:
            self.processed = self.hrn(x)
            return self.processed


class ContextResUNet(nn.Module):
    """Context-enchanced U-Net with residual blocks and deep supervision."""
    def __init__(self, load_pretrained_context=True):
        super().__init__()
        # construct contextnet in context mode
        self.context_block = ContextNet(False)
        # load weights pretrained on BraTS-2024
        if load_pretrained_context: self.context_block.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'ContextNet.pt')))
        # Residual U-Net with deep supervision (reconstruction loss term on each U-Net level)
        self.net = SegResNetDS(2, 32, 32, 4)
    def forward(self, x:torch.Tensor):
        # extract 20x96x96 context
        self.context = self.context_block(x)
        # pass the context and the input to the U-Net
        return self.net(torch.cat((x, self.context), 1))



def get_pretrained_model():
    model = ContextResUNet(False)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'ContextResUNet.pt')))
    return model