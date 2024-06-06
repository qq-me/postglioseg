import atexit
from typing import Optional, Any
from collections.abc import Sequence
import  os, shutil, random
import torch
import zarr
import zarr.storage
import numpy as np
from glio.torch_tools import area_around
from glio.python_tools import get_all_files
from glio.jupyter_tools import show_slices, show_slices_arr

# exit handle
print("Loading BRaTS2024 ZipStores... ", end = '')
brats_stores = [zarr.ZipStore(i, mode='r') for i in get_all_files(r'E:\dataset\BRaTS2024 2D full norm\dataset', False, 'zip')]
print("Done.")

def close_stores():
    for store in brats_stores:
        store.close()

def exit_handler():
    close_stores()
    print("Closed stores.")

atexit.register(exit_handler)


def get_dataset(stores:zarr.storage.StoreLike | Sequence[zarr.storage.StoreLike] | int | Sequence[int] | slice = tuple(brats_stores),
                preps:str|Sequence[str] = ('nohist', 'hist'),
                dims:str|Sequence[str] = ('side', 'front', 'top'),
                mods:Sequence[str] | tuple[int,int] = (0,4),
                slices_around = 1,
                ):
        # create dataset
        dataset:list[dict] = []

        # preprocess args
        if isinstance(stores, int): stores = (stores,)
        if isinstance(stores, slice): stores = tuple(brats_stores[stores])
        elif isinstance(stores[0], int): stores = tuple([brats_stores[i] for i in stores]) # type:ignore
        elif isinstance(stores, zarr.storage.StoreLike): stores = (stores,)
        
        if isinstance(preps, str): preps = (preps,)
        if isinstance(dims, str): dims = (dims,)
        if isinstance(mods, str): mods = (mods,)

        preps_i = [('nohist', 'hist').index(p.lower()) for p in preps]
        if isinstance(mods[0], str):
            mods_i = [('t1c','t1n','t2f','t2w').index(m.lower()) for m in mods] # type:ignore

        for store in stores:
            root = zarr.group(store)
            for study_key, study in root.items():
                for dim in dims:
                    centers = study[f"{dim}_centers"] # type:ignore
                    for prep in preps_i:
                        if isinstance(mods[0], int):
                            if slices_around == 0:
                                for sl, center in enumerate(centers):
                                    dataset.append(dict(root=root, study=study_key, dim=dim, slice=(prep, slice(*mods), sl), center=center))
                            else:
                                for sl, center in enumerate(centers[slices_around:-slices_around]):
                                    sl+=slices_around
                                    dataset.append(dict(root=root, study=study_key, dim=dim, slice=(prep, slice(*mods), slice(sl-slices_around, sl+slices_around+1)), center=center))
                        else:
                            for mod in mods_i: # type:ignore
                                if slices_around == 0:
                                    for sl, center in enumerate(centers):
                                        dataset.append(dict(root=root, study=study_key, dim=dim, slice=(prep, mod, sl), center=center))
                                else:
                                    for sl, center in enumerate(centers[slices_around:-slices_around]):
                                        sl+=slices_around
                                        dataset.append(dict(root=root, study=study_key, dim=dim, slice=(prep, mod, slice(sl-slices_around, sl+slices_around+1)), center=center))

        return dataset


def loader(study:dict, area=(96,96), flatten_slices=True):
    # load attrubutes
    root = study["root"]
    study_id = study["study"]
    dim = study["dim"]
    center = study["center"]
    slice = study["slice"] #pylint:disable=W0622
    # load data
    study_data = root[study_id]
    image = torch.from_numpy(area_around(study_data[dim][slice], center, area))
    segm = torch.from_numpy(area_around(study_data[f"{dim}_seg"][slice[2]], center, area))
    if flatten_slices: image = image.flatten(0, 1)
    return image, segm