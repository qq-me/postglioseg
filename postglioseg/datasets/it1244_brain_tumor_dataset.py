from torchvision.transforms import v2
from ..data import DSClassification
from ..loaders.image import read
from ..transforms import ToChannels
MEAN = (0.2466959655, 0.2469369471, 0.2472953051)
STD = (0.2389927208, 0.2390660942, 0.2392502129)
def get_dataset(path = "D:/datasets/it1244-brain-tumor-dataset/data", loader = v2.Compose([read, ToChannels(3), v2.Normalize(MEAN, STD)])):
    ds = DSClassification()
    with open(f"{path}/train/data.csv", 'r', encoding='utf8') as f:
        for line in f:
            num, label = line.strip().replace("\ufeff", '').split(',')
            ds.add_sample(f"{path}/train/{num}.jpg", target = label, loader=loader)
    return ds