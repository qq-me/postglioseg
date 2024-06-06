import numpy as np
import torch
import joblib
from stuff.found.mnist1d.data import get_dataset, get_dataset_args
from ..data import DSClassification, DSToTarget
from ..torch_tools import CUDA_IF_AVAILABLE

__all__ = ["get_mnist1d_classification", "get_mnist1d_autoenc"]
def _loader(x, device=CUDA_IF_AVAILABLE):
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

def get_mnist1d_classification(path='D:/datasets/mnist1d_data.pkl', download=False, device=CUDA_IF_AVAILABLE):
    args = get_dataset_args()
    data = get_dataset(args, path, download=download)
    dstrain = DSClassification()
    dstest = DSClassification()
    x = data['x']
    y = data['y']
    xtest = data['x_test']
    ytest = data['y_test']
    for sx, sy in zip(x, y):
        dstrain.add_sample(data = sx, target = torch.nn.functional.one_hot(torch.from_numpy(np.array([sy])), 10)[0].to(device, torch.float32), loader = lambda x: _loader(x, device=device), target_encoder=None) # pylint:disable=E1102

    for sx, sy in zip(xtest, ytest):
        dstest.add_sample(data = sx, target = torch.nn.functional.one_hot(torch.from_numpy(np.array([sy])), 10)[0].to(device, torch.float32), loader = lambda x: _loader(x, device=device), target_encoder=None)# pylint:disable=E1102

    dstrain.preload(); dstest.preload()
    return dstrain, dstest

def get_mnist1d_autoenc(path='D:/datasets/mnist1d_data.pkl', download=False, device=CUDA_IF_AVAILABLE):
    args = get_dataset_args()
    data = get_dataset(args, path, download=download)
    dstrain = DSToTarget()
    dstest = DSToTarget()
    dstrain.add_samples(data = data["x"], loader = lambda x: _loader(x, device=device))

    dstest.add_samples(data = data['x_test'], loader = lambda x: _loader(x, device=device))

    dstrain.preload(); dstest.preload()
    return dstrain, dstest
