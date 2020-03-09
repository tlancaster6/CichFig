import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
import os

class Plotter:

    def __init__(self, data_object, file_manager):
        self.do = data_object
        self.fm = file_manager
        self.pid = self.do.pid
        self.bids = ['c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x']
        self.n_days = len(self.do.cluster_data.daily.splits)

    def plot_all(self):
        print('plotting')
        fig = plt.figure(FigureClass=CustomFigure, nrows=11, ncols=self.n_days + 1)
        self.plot_daily_kdes(row=0, fig=fig)
        self.plot_daily_depths(row=10, fig=fig)
        # self.save_fig(fig, os.path.join(self.pid, 'Figures/all_figures.pdf'))
        # plt.close(fig)
        fig.show()

    def save_fig(self, fig, relative_path):
        print('    saving')
        dest = self.fm.get_local_path(relative_path)
        fig.savefig(dest)

    def plot_daily_kdes(self, row=0, fig=None):
        print('    kde images')
        kdes = self.do.cluster_data.daily.kdes.__dict__
        v = max([np.nanmax(np.abs(kde)) for kde in kdes.values()])
        if fig is None:
            fig = CustomFigure(nrows=10, ncols=self.n_days + 1, cell_size=1)
            row = 0
        for day in range(self.n_days):
            counter = row
            for bid in self.bids:
                ax = fig.new_subplot(row=row, col=day)
                fig.array_plot(kdes[bid][day], ax=ax, v=v, title='day {}'.format(day))
                counter += 1

    def plot_daily_depths(self, row=0, fig=None):
        print('    depth change images')
        if fig is None:
            fig = CustomFigure(nrows=1, ncols=self.n_days + 1, cell_size=1)
            row = 0
        for day in range(self.n_days):
            ax = fig.new_subplot(row=row, col=day)
            fig.array_plot(self.do.depth_data.daily.height_change[day], ax=ax)


class CustomFigure(Figure):

    def __init__(self, nrows=1, ncols=1, cell_size=1):
        super().__init__(figsize=(cell_size * ncols, cell_size * nrows), constrained_layout=True)
        self.spec = self.add_gridspec(nrows, ncols)

    def new_subplot(self, row, col, row_span=1, col_span=1):
        return self.add_subplot(self.spec[row: row + row_span, col: col + col_span])

    def array_plot(self, array, ax=None, v=None, **kwargs):
        # uses matplotlib.pyplot.imshow() to create a colorized image of array. Used to plot depth change and kde arrays
        if ax is None:
            ax = self.gca()
        ax.set(**kwargs)
        if v is None:
            v = np.nanmax(np.abs(array))
        plot = ax.imshow(array, cmap='viridis', aspect='equal', vmin=-v, vmax=v)
        ax.tick_params(colors=[0, 0, 0, 0])
        return plot

    def index_plot(self, t_data, y_data, ax=None, **kwargs):
        # creates a line plot of index data over time. used to plot the bower-index and scoop-spread index
        if ax is None:
            ax = self.gca()
        ax.set(**kwargs)
        ax.axhline(linewidth=1, alpha=0.5, y=0)
        plot = ax.scatter(t_data, y_data)
        return plot

