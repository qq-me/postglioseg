import matplotlib.pyplot as plt
from glio.jupyter_tools import show_slices
def imslices(files): show_slices([plt.imread(i) for i in files])
