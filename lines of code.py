
from glob import glob
from json import load

def loc(nb):
    with open(nb, encoding='utf8') as f:
        cells = load(f)["cells"]
        return sum(len(c["source"]) for c in cells)

def num_python_lines(file):
    with open(file, encoding='utf8') as f:
        py = f.read()
        count = 0
        for line in py.split("\n"):
            if line.startswith("#"):
                continue
            if line.strip():
                count += 1
    return count

root_folder = "."
summation = 0
sum_py = 0
for File in glob(root_folder+"/**/*.ipynb", recursive=True):
    summation += loc(File)
for File in glob(root_folder+"/**/*.py", recursive=True):
    sum_py += num_python_lines(File)
print(summation, sum_py)