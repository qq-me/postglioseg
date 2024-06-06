from collections.abc import Sequence
import torch
from glio.train2 import *
from glio.plot import Figure
from glio.torch_tools import one_hot_mask

TITLE = "M2NIST"

_CBSTUFF = lambda:  [
    Save_Last('M2NIST checkpoints'), 
    PerformanceTweaks(True), 
    Accelerate("no"),
    SimpleProgressBar(step=1),
    LivePlot(128, plot_keys = ("4plotsplot01","10metrics01"),path_keys=("4plotspath250",)),
    Summary(),
    PlotSummary(),
    ]

_CBMETRICS = lambda: [
    Metric_Loss(),
    Metric_Accuracy(True, True, False, step=8),
    MONAI_IoU(11, True, True, step=8),
    Torcheval_Precision(11, True, True, step=8),
    Torcheval_Recall(11, True, True, step=8),
    Torcheval_Dice(11, True, True, step=8),
    Torcheval_AURPC(11, True, step=8),
    Torcheval_AUROC(11, True, step=8),
    Log_Time(),
    #Log_LR(),
    ]

_CBUPDATEMETRICS = lambda: [
    # Log_GradDist(16),
    # Log_GradUpdateAngle(16),
    # Log_LastGradsAngle(16),
    # Log_GradPath(1),
    Log_UpdateDist(16),
    Log_LastUpdatesAngle(16),
    Log_ParamDist(16),
    Log_ParamPath(1),
    Log_UpdatePath(1),
    ]

def CALLBACKS(extra = ()):
    if not isinstance(extra, (Sequence)): extra = [extra]
    return _CBSTUFF() + _CBMETRICS() + _CBUPDATEMETRICS() + list(extra)

def plot_preds(learner:Learner, sample:tuple[torch.Tensor,torch.Tensor]):
    input, target = sample
    preds = learner.inference(input.unsqueeze(0))
    fig = Figure()
    fig.add().imshow(input).style_img("input")
    fig.add().imshow_batch(target).style_img("target")
    fig.add().imshow_batch(preds[0]).style_img("preds")
    fig.add().imshow_batch(one_hot_mask(preds[0].argmax(0), 11)).style_img("preds binary")
    fig.show(1, figsize=(14,14))