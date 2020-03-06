import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

class Plotter:

    def __init__(self, data_handler, file_manager):
        self.dh = data_handler
        self.fm = file_manager

    def new_fig(self, nrows=1, ncols=1, cell_size=1):
        fig = plt.figure(figsize=(cell_size * ncols, cell_size * nrows), constrained_layout=True)

    def kde_plot(self, ax: ):
