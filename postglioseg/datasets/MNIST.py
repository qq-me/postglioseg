import torch
from torchvision.transforms import v2
import joblib
from torchvision.datasets import MNIST
from ..data import DSClassification

MEAN_STD = ((0.1307,), (0.3081,))
norm = v2.Normalize(*MEAN_STD)
loader = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), norm])

def create_mnist(root='D:/datasets'):
    mnist_train = MNIST(root=root, train=True)
    mnist_test = MNIST(root=root, train=False)

    dstrain = DSClassification().from_external_dataset(mnist_train, loader=loader, n_threads=0)
    dstest = DSClassification().from_external_dataset(mnist_test, loader=loader, n_threads=0)
    dstrain.preload(nthreads=16, log=True)
    dstest.preload(nthreads=16, log=True)

    #return dstrain, dstest
    joblib.dump(dstrain, 'D:/datasets/MNIST_classification_train.joblib', compress=3)
    joblib.dump(dstest, 'D:/datasets/MNIST_classification_test.joblib', compress=3)

def get_mnist_classification(path = 'D:/datasets'):
    dstrain:DSClassification = joblib.load(path + '/MNIST_classification_train.joblib')
    dstest:DSClassification = joblib.load(path + '/MNIST_classification_test.joblib')
    return dstrain, dstest