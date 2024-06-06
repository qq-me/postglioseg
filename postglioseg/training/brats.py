import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from glio.visualize import Visualizer
from glio.transforms import norm_to01
from glio.train2 import *

color_legend = "\nчерный:нет;\nсиний:отёк;\nзелёный:некротическое ядро,\nкрасный:усиливающая опухоль"
cmp = ListedColormap(['black','red', 'green', 'blue'])

def plot_preds(learner:Learner, batch, softmax = True, unsqueeze = True, expand_channels = None):
    batch = list(batch)
    if unsqueeze:
        batch[0] = batch[0].unsqueeze(0)
        batch[1] = batch[1].unsqueeze(0)
    if expand_channels is not None:
        shape = list(batch[0].shape)
        shape[1] = expand_channels - shape[1]
        batch[0] = torch.cat((batch[0], torch.zeros(shape)), dim=1)
    preds = learner.inference(batch[0].to(learner.device))
    batch[0][0] = torch.stack([norm_to01(i) for i in batch[0][0]])
    batch[0][:,3,0,0] = 1
    preds[0][3,0,0] = 1
    v = Visualizer()
    v.imshow_grid(batch[0][0][1::3], mode="bhw", label="вход:\nT1n, T1c, FLAIR, T2W")
    v.imshow(batch[1][0].argmax(0), mode="bhw", label = f"реальная карта:{color_legend}", cmap=cmp)
    if softmax:
        output = torch.stack([preds[0],preds[0],preds[0]], dim=1)
        output[:,0] *=  F.softmax(preds[0],0)
        v.imshow_grid(output, mode="bchw", label="сырой выход:\nнет, отёк,\nнекротическое ядро,\nусиливающая опухоль")
    else:
        v.imshow_grid(preds[0], mode="bchw", label="сырой выход:\nнет, отёк,\nнекротическое ядро,\nусиливающая опухоль")
    v.imshow(preds[0].argmax(0), mode="bhw", label=f"предсказанная карта:{color_legend}", cmap=cmp)
    v.imshow(preds[0].argmax(0) != batch[1][0].argmax(0), mode="bhw", label="ошибка", cmap='gray')
    imgs = torch.stack([batch[0][0],batch[0][0],batch[0][0]], dim=0)
    imgsreal = torch.stack([batch[0][0],batch[0][0],batch[0][0]], dim=0)
    apreds = preds[0].argmax(0)
    aptargets = batch[1][0].argmax(0)
    imgs[0] = torch.where(apreds == 1, imgs[0]*2, imgs[0]).clip(0,1)
    imgs[1] = torch.where(apreds == 2, imgs[1]*2, imgs[1]).clip(0,1)
    imgs[2] = torch.where(apreds == 3, imgs[2]*2, imgs[2]).clip(0,1)
    imgsreal[0] = torch.where(aptargets == 1, imgsreal[0]*2, imgsreal[0]).clip(0,1)
    imgsreal[1] = torch.where(aptargets == 2, imgsreal[1]*2, imgsreal[1]).clip(0,1)
    imgsreal[2] = torch.where(aptargets == 3, imgsreal[2]*2, imgsreal[2]).clip(0,1)
    v.imshow_grid(imgs.swapaxes(0,1), mode="bchw", label=f"предсказанная карта:{color_legend}")
    v.imshow(imgs[:,0], label=f"предсказанная карта:{color_legend}")
    v.imshow(imgsreal[:,0], label=f"реальная карта:{color_legend}")
    v.show(figsize=(24, 24), nrows=1, fontsize=12)