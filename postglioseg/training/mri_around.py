import torch
from matplotlib.colors import ListedColormap
from glio.plot import *
from glio.train2 import *
from glio.transforms import norm_to01

def get_cbs(title, cbs = ()):
    CALLBACKS =  (Metric_Loss(), # Log_GradHistorgram(16), Log_SignalHistorgram(16), Log_LastGradsAngle(128), Log_GradPath(1)
                #Log_UpdateDist(128), Log_GradDist(128), Log_GradUpdateAngle(128), Log_ParamDist(128),
                #Log_LastUpdatesAngle(128),
                #Log_ParamPath(32), Log_UpdatePath(32),
                Log_Time(), Save_Best(title), Save_Last(title), Log_LR(), PerformanceTweaks(True), Accelerate("no"),
                Metric_Accuracy(True, True, False, name = 'accuracy', step=4),
                MONAI_IoU(4, True, True, step=32, name='iou'),
                Torcheval_Precision(4, True, True, step=16),
                Torcheval_Recall(4, True, True, step=16),
                Torcheval_Dice(4, True, True, step=8, name='f1'),
                Torcheval_AURPC(4, True, step=32),
                Torcheval_AUROC(4, True, step=32),
                FastProgressBar(step_batch=128, plot=True),
                Summary(),
                PlotSummary(path='summaries'),
                # CallTrainAndEvalOnOptimizer(),
                ) + cbs

    return CALLBACKS


brgb = ListedColormap(['black','red', 'green', 'blue'])
brgb_legend = "\nчерный:нет;\nсиний:отёк;\nзелёный:некротическое ядро,\nкрасный:усиливающая опухоль"


def visualize_3_segm_classes(inputs:torch.Tensor, segm:torch.Tensor):
    """
    inputs: HW
    segm: 4HW or HW, where 0th class is background"""
    preview = torch.stack([inputs,inputs,inputs], dim=0)
    if segm.ndim == 3: segm = segm.argmax(0)
    preview[0] = torch.where(segm == 1, preview[0]*2, preview[0]).clip(0,1)
    preview[1] = torch.where(segm == 2, preview[1]*2, preview[2]).clip(0,1)
    preview[2] = torch.where(segm == 3, preview[2]*2, preview[2]).clip(0,1)
    return preview

def visualize_predictions(inferer, sample:tuple[torch.Tensor, torch.Tensor], around=3, expand_channels = None):
    fig = Figure()
    inputs:torch.Tensor = sample[0].unsqueeze(0)
    targets_raw:torch.Tensor = sample[1]
    targets:torch.Tensor = targets_raw.argmax(0)
    if expand_channels is None: preds_raw:torch.Tensor = inferer(inputs)[0]
    else:
        expanded = torch.cat((inputs, torch.zeros((1, expand_channels - inputs.shape[1], *inputs.shape[2:]))), dim=1)
        preds_raw:torch.Tensor = inferer(expanded)[0]
    preds:torch.Tensor = preds_raw.argmax(0)

    inputs[0] = torch.stack([norm_to01(i) for i in inputs[0]])
    fig.add().imshow_batch(inputs[0, 1::around], scale_each=True).style_img('вход:\nT1c, T1, FLAIR, T2')
    fig.add().imshow_batch(preds_raw, scale_each=True).style_img(f'сырой выход\n{brgb_legend}')
    fig.add().imshow(targets, cmap=brgb).style_img(f'реальная карта\n{brgb_legend}')
    fig.add().imshow(preds, cmap=brgb).style_img(f'предсказанная карта\n{brgb_legend}')
    fig.add().imshow_batch((preds_raw - targets_raw).abs(), scale_each=True).style_img('сырая ошибка')
    fig.add().imshow(preds != targets,  cmap='gray').style_img('ошибка')

    fig.add().imshow(visualize_3_segm_classes(inputs[0][0], targets)).style_img(f'реальная карта\n{brgb_legend}')
    fig.add().imshow(visualize_3_segm_classes(inputs[0][0], preds)).style_img(f'предсказанная карта\n{brgb_legend}')

    fig.create(2, figsize=(16,16))
