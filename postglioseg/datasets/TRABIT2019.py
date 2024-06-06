# Автор Никишев И.О.
# TODO Загрузчик

import torch
from torchvision.transforms import v2
from ..data.old import Dataset_Label
from ..loaders import nifti
import csv, functools

# Предрассчитанные значения для нормализации
mean, std = ((234.26439405168807,), (431.60501534598217,))
age_mean, age_std = 38.9000, 30.0628
age_min, age_max = 18.0, 96.0

def age_norm_z(x):
    return (x - age_mean) / age_std

def age_norm_01(x):
    return (x - age_min) / (age_max - age_min)

def load_train(path = r'D:\datasets\trabit2019-imaging-biomarkers',
         loader = [nifti.niiread, functools.partial(torch.squeeze, dim = 3), functools.partial(torch.unsqueeze, dim = 0), v2.Normalize(mean, std)],
         transforms = None,
         label_fn = age_norm_z
         ):
    ds = Dataset_Label(loader = loader, transform=transforms, label_encoding='float')
    with open(f'{path}/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'scan_id': continue
            ds.add_file(f'{path}/train/{row[2]}', label_fn = label_fn(float(row[1])))
    return ds

def load_test(path = r'D:\datasets\trabit2019-imaging-biomarkers',
         loader = [nifti.niiread, functools.partial(torch.squeeze, dim = 3), functools.partial(torch.unsqueeze, dim = 0), v2.Normalize(mean, std)],
         transforms = None,
         ):
    ds = Dataset_Label(loader = loader, transform=transforms, label_encoding='float')

    with open(f'{path}/test_sample_submission.csv', 'r') as f:
        reader = list(csv.reader(f))
        for i, row in enumerate(reader):
            if row[0] == 'scan_id': continue
            filepath = f'{path}/test/mri_{row[0].rjust(8, "0")}.nii'
            ds.add_file(filepath)
    return ds

def age_unnorm_z(x):
    inverse_mean = -age_mean/age_std
    inverse_std = 1/age_std
    return (x - inverse_mean) / inverse_std

def age_unnorm_01(x):
    return x * (age_max-age_min) + age_min


def test_inference(l, path = r'D:\datasets\trabit2019-imaging-biomarkers',
         loader = v2.Compose([nifti.niiread, functools.partial(torch.squeeze, dim = 3), functools.partial(torch.unsqueeze, dim = 0), v2.Normalize(mean, std)]),
         transforms = None,
         label_fn = age_unnorm_z,
         test_time_aug_n = 1
         ):

    with open(f'{path}/test_sample_submission.csv', 'r') as f:
        kagle_reader = list(csv.reader(f))
        kaggle_submission = kagle_reader.copy()
        for i, row in enumerate(kagle_reader):
            if row[0] == 'scan_id': continue
            filepath = f'{path}/test/mri_{row[0].rjust(8, "0")}.nii'
            file: list[torch.Tensor] = [None for _ in range(test_time_aug_n)] # pyright: ignore
            for j in range(len(file)):
                file[j] = transforms(loader(filepath)) if transforms is not None else loader(filepath)
            batch = torch.stack(file, 0)
            #print(file.shape)
            if l.in_size is None: l.in_size = (1, 1, 176, 208, 176)
            l.inference(batch)
            prediction = l.preds.mean()
            kaggle_submission[i][1] = (label_fn(float(prediction.cpu().detach())))

    with open(f'{path}/test_sample_submission {l.name}.csv', 'w', newline='') as f:
        writer = csv.writer(f,delimiter=",")
        writer.writerows(kaggle_submission)
        print('Done')