import torch
from monai.inferers import SlidingWindowInferer # type:ignore
from ..python_tools import find_file_containing
from glio.torch_tools import sliding_inference_around_3d
from glio.jupyter_tools import show_slices_arr
from glio.datasets.preprocessor import Preprocessor
from glio.train2 import Learner, CBCond
from glio.plot import *

def rotate(x:torch.Tensor):
    return x.flip(2)

#t1c, t1n, t2f, t2w, (adc)
def preprocess_inputs_from_folder(path, mods = ('T1c.', 'T1.', 'FLAIR.', 'T2.'), hist=True) -> torch.Tensor:
    """Loads all nii gz images from path, returns 3D tensor of size (`mods`, 140, 180, 140)"""
    files = [find_file_containing(path, i) for i in mods]
    preprocessor = Preprocessor(rotate=rotate)
    images, images_nohist = preprocessor(*files, return_nohist=True)

    if hist: return images
    else: return images_nohist

def preprocess_inputs_targets_from_folder(path, mods = ('T1c.', 'T1.', 'FLAIR.', 'T2.'), hist=True) -> tuple[torch.Tensor,torch.Tensor]:
    """Loads all nii gz images from path, returns 3D tensor of size (`mods`, 140, 180, 140)"""
    files = [find_file_containing(path, i) for i in mods]
    seg = find_file_containing(path, 'seg')
    preprocessor = Preprocessor(rotate=rotate)
    images, images_nohist, segmentation = preprocessor(*files, seg=seg, return_nohist=True)

    if hist: return images, segmentation
    else: return images_nohist, segmentation
    
def preprocess_from_our_folder(path, hist=True):
    return preprocess_inputs_from_folder(path, mods = ('T1c.', 'T1.', 'FLAIR.', 'T2.'), hist=hist)
def preprocess_from_brats_folder(path, hist=True):
    return preprocess_inputs_targets_from_folder(path, mods = ('t1c.', 't1n.', 't2f.', 't2w.'), hist=hist)
def preprocess_from_rhuh_folder(path, hist=True):
    return preprocess_inputs_targets_from_folder(path, mods = ('t1ce.', 't1.', 'flair.', 't2.', 'adc.'), hist=hist)


class PredictFrom(CBCond):
    def __init__(self, input, name = 'last preds'):
        self.name = name
        self.input = input
        super().__init__()

    def __call__(self, learner:Learner):
        learner.logger.set(self.name, learner.inference(self.input))

def _inferer(input, inferer):
    return inferer(input)

class PredictFromOurImages(CBCond):
    def __init__(self, input, channel = None, name = 'last preds our', sliding=True):
        self.name = name
        self.input = input
        if channel is None: self.channel = self.input[0, int(self.input.size(1)//2)]
        else: self.channel = channel
        if sliding: self.inferer = SlidingWindowInferer(roi_size=(96,96), sw_batch_size=16, overlap=0.5, mode='gaussian')
        else: self.inferer = _inferer
        super().__init__()

    def __call__(self, learner:Learner):
        preds = self.inferer(self.input, learner.inference)[0].argmax(0)
        preview = torch.stack([self.input[0,self.channel],self.input[0,self.channel],self.input[0,self.channel]], dim=0)
        preview[0] = torch.where(preds == 1, preview[0]*2, preview[0]).clip(0,1)
        preview[1] = torch.where(preds == 2, preview[1]*2, preview[1]).clip(0,1)
        preview[2] = torch.where(preds == 3, preview[2]*2, preview[2]).clip(0,1)
        learner.logger.set(self.name, preview, learner.total_batch)


def get_checkpoint_preds_on_our(cpath, model, imgpath, around = 1, hist=True):
    """Tests a checkpoint on our images"""
    input = preprocess_inputs_from_folder(imgpath, hist=hist)

    learner = Learner.from_checkpoint(cpath, model=model, cbs=())
    preds = sliding_inference_around_3d(input.unsqueeze(0), learner.inference, (96,96), 16, around, 4)

    return preds

def get_checkpoint_preds_on_our_color(cpath, model, imgpath, around = 1, hist=True, ch=0):
    """Tests a checkpoint on our images"""
    input = preprocess_inputs_from_folder(imgpath, hist=hist)

    learner = Learner.from_checkpoint(cpath, model=model, cbs=())
    preds = sliding_inference_around_3d(input.unsqueeze(0), learner.inference, (96,96), 16, around, 4)

    preview = torch.stack([input[ch],input[ch],input[ch]], dim=0)
    preview[0] = torch.where(preds == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(preds == 2, preview[1]*2, preview[1]).clip(0,1)
    preview[2] = torch.where(preds == 3, preview[2]*2, preview[2]).clip(0,1)
    return preview

def get_checkpoint_preds_on_our_color_showlices(cpath, model, imgpath, around = 1, hist=True, ch=0):
    """Tests a checkpoint on our images"""
    return show_slices_arr(get_checkpoint_preds_on_our_color(cpath, model, imgpath, around, hist, ch))