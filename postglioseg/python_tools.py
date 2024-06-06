"""PostGlioSeg, автор - Никишев Иван Олегович"""
from collections.abc import Sequence, Callable, Iterable
from typing import Optional, Any
import os
def get_all_files(
    path: str,
    recursive: bool = True,
    extensions: Optional[str | Sequence[str]] = None,
    path_filter: Optional[Callable] = None,
) -> list[str]:
    """Returns a list of full paths to all folders and files within the given path."""
    all_files = []
    if isinstance(extensions, str): extensions = [extensions]
    if extensions is not None: extensions = tuple(extensions)
    if recursive:
        for root, _, files in (os.walk(path)):
            for file in files:
                file_path = os.path.join(root, file)
                if path_filter is not None and not path_filter(file_path): continue
                if extensions is not None and not file.lower().endswith(extensions): continue
                if os.path.isfile(file_path): all_files.append(file_path)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if path_filter is not None and not path_filter(file_path): continue
            if extensions is not None and not file.lower().endswith(extensions): continue
            if os.path.isfile(file_path): all_files.append(file_path)

    return all_files

def find_file_containing(folder, contains:str, recursive = True, error = True, lower=False) -> str:
    """Returns full path to the first folder or file that contains `contains` in its name."""
    for f in get_all_files(folder, recursive=recursive):
        if lower and contains.lower() in f.lower(): return f
        if (not lower) and contains in f: return f
    if error: raise FileNotFoundError(f"File containing {contains} not found in {folder}")
    return None # type:ignore

def flatten(iterable:Iterable) -> list[Any]:
    """Flatten an iterable"""
    if isinstance(iterable, Iterable): return [a for i in iterable for a in flatten(i)]
    else: return [iterable]


class Compose:
    """Compose functions"""
    def __init__(self, *transforms):
        self.transforms = flatten(transforms)

    def __call__(self, x):
        for t in self.transforms: x = t(x)
        return x

    def __add__(self, other: "Compose | Callable | Iterable"):
        return Compose(*self.transforms, other)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.transforms)})"

    def __iter__(self): return iter(self.transforms)
    def __getitem__(self, i): return self.transforms[i]
    def __setitem__(self, i, v): self.transforms[i] = v
    def __delitem__(self, i): del self.transforms[i]