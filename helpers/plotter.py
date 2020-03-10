import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

import numpy as np
import os

bluegreen = (0.00392156862745098, 0.47843137254901963, 0.4745098039215686)
amber = (0.996078431372549, 0.7019607843137254, 0.03137254901960784)
purple = (0.49411764705882355, 0.11764705882352941, 0.611764705882353)

class Plotter:

    def __init__(self, data_object, file_manager):
        self.do = data_object
        self.fm = file_manager
        self.pid = self.do.pid
        self.bids = ['c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x']
        self.n_days = len(self.do.cluster_data.daily.splits)

    def plot_all(self):
        fig = plt.figure(FigureClass=CustomFigure, nrows=18, ncols=self.n_days + 1)
        self.plot_daily_kdes(start_row=0, fig=fig)
        self.plot_daily_spit_scoop_difference(start_row=11, fig=fig)
        self.plot_daily_heights(start_row=13, fig=fig)
        self.plot_daily_indices(start_row=15, fig=fig)
        self.save_fig(fig, os.path.join(self.pid, 'Figures/all_figures.pdf'))
        plt.close('all')

    def save_fig(self, fig, relative_path):
        dest = self.fm.get_local_path(relative_path)
        fig.savefig(dest)

    def plot_daily_kdes(self, start_row=0, fig=None):
        # requires 11 rows
        kdes = self.do.cluster_data.daily.kdes.__dict__
        save_flag = False
        if fig is None:
            fig = CustomFigure(nrows=11, ncols=self.n_days + 1, cell_size=1)
            start_row = 0
            save_flag = True
        title_ax = fig.new_subplot(start_row, 0, col_span=self.n_days, despine=True)
        title_ax.text(0.5, 0.5, r'Daily Behavioral Densities ($events/cm^3$)', ha='center', va='center',
                      size=fig.font_size['L'])
        for row, bid in enumerate(self.bids):
            vmax = np.nanmax(np.abs(kdes[bid]))
            for day in range(self.n_days):
                ax = fig.new_subplot(row=row + start_row + 1, col=day)
                fig.array_plot(kdes[bid][day], ax=ax, vmin=0, vmax=vmax)
                if day == 0:
                    ax.set_ylabel(bid)
                if row == start_row + 10:
                    ax.set_xlabel('day {}'.format(day))
            fig.cbar(vmin=0, vmax=vmax, cax=fig.new_subplot(start_row + row + 1, self.n_days))
        if save_flag:
            self.save_fig(fig, os.path.join(self.pid, 'Figures/daily_kdes.pdf'))
            plt.close('all')

    def plot_daily_spit_scoop_difference(self, start_row=0, fig=None):
        # requires 2 rows
        diff = self.do.cluster_data.daily.kdes.p - self.do.cluster_data.daily.kdes.c
        save_flag = False
        if fig is None:
            fig = CustomFigure(nrows=2, ncols=self.n_days + 1, cell_size=1)
            start_row = 0
            save_flag = True
        title_ax = fig.new_subplot(start_row, 0, col_span=self.n_days, despine=True)
        title_ax.text(0.5, 0.5, r'Daily Spit Density Minus Scoop Density ($\Delta events/cm^3$)', ha='center', va='center',
                      size=fig.font_size['L'])
        vmax = np.nanmax(np.abs(diff))
        for day in range(self.n_days):
            ax = fig.new_subplot(row=start_row + 1, col=day)
            fig.array_plot(diff[day], ax=ax, vmax=vmax)
            ax.set_xlabel('day {}'.format(day))
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(start_row + 1, self.n_days))
        if save_flag:
            self.save_fig(fig, os.path.join(self.pid, 'Figures/daily_spit_scoop_diff.pdf'))
            plt.close('all')

    def plot_daily_heights(self, start_row=0, fig=None):
        # requires 2 rows
        save_flag = False
        vmax = np.nanquantile(np.abs(self.do.depth_data.daily.height_change), q=0.99)
        if fig is None:
            fig = CustomFigure(nrows=2, ncols=self.n_days + 1, cell_size=1)
            start_row = 0
            save_flag = True
        title_ax = fig.new_subplot(start_row, 0, col_span=self.n_days, despine=True)
        title_ax.text(0.5, 0.5, r'Daily Height Changes ($cm$)', ha='center', va='center', size=fig.font_size['L'])
        for day in range(self.n_days):
            ax = fig.new_subplot(row=start_row + 1, col=day)
            ax.set_xlabel('day {}'.format(day))
            fig.array_plot(self.do.depth_data.daily.height_change[day], ax=ax, vmax=vmax)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(start_row + 1, self.n_days))
        if save_flag:
            self.save_fig(fig, os.path.join(self.pid, 'Figures/daily_height_change.pdf'))
            plt.close('all')

    def plot_daily_bowers(self, start_row=0, dtype='depth', fig=None):
        pass

    def plot_daily_bower_overlap(self):

    def plot_daily_indices(self, start_row=0, fig=None):
        #requires 3 rows
        save_flag = False
        if fig is None:
            fig = CustomFigure(nrows=2, ncols=self.n_days + 1, cell_size=1)
            start_row = 0
            save_flag = True
        title_ax = fig.new_subplot(start_row, 0, col_span=self.n_days, despine=True)
        title_ax.text(0.5, 0.5, r'Daily Bower Index', ha='center', va='center', size=fig.font_size['L'])
        days = list(range(self.n_days))
        plotting_ax = fig.new_subplot(start_row + 1, 0, 2, self.n_days)
        fig.index_plot(t_data=days, y_data=self.do.cluster_data.daily.bower_index, ax=plotting_ax,
                       label='from cnn results', c=amber)
        fig.index_plot(t_data=days, y_data=self.do.depth_data.daily.bower_index, ax=plotting_ax,
                       label='from depth data', c=purple)
        plotting_ax.legend(ncol=2, fontsize=fig.font_size['S'])
        if save_flag:
            self.save_fig(fig, os.path.join(self.pid, 'Figures/daily_bower_indices.pdf'))
            plt.close('all')

class CustomFigure(Figure):

    def __init__(self, nrows=1, ncols=1, cell_size=1, **kwargs):
        super().__init__(figsize=(cell_size * ncols, cell_size * nrows))
        self.spec = self.add_gridspec(nrows, ncols)
        self._set_figure_params()

    def new_subplot(self, row, col, row_span=1, col_span=1, despine=False):
        ax = self.add_subplot(self.spec[row: row + row_span, col: col + col_span])
        if despine:
            sns.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(colors=[0, 0, 0, 0])
        return ax

    def array_plot(self, array, ax=None, vmin=None, vmax=None, **kwargs):
        # uses matplotlib.pyplot.imshow() to create a colorized image of array. Used to plot depth change and kde arrays
        if ax is None:
            ax = self.gca()
        if vmin is None and vmax is None:
            vmax = np.nanmax(np.abs(array))
            vmin = -vmax
        elif vmin is None or vmax is None:
            vmin = vmin if vmax is None else -vmax
            vmax = vmax if vmin is None else -vmin
        plot = ax.imshow(array, cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax, **kwargs)
        ax.tick_params(colors=[0, 0, 0, 0])
        return plot

    def cbar(self, vmin, vmax, cax):
        cax.set(aspect=10, anchor='W')
        mappable = ScalarMappable(Normalize(vmin, vmax), 'viridis')
        cbar = plt.colorbar(mappable, cax)
        return cbar

    def index_plot(self, t_data, y_data, ax=None, **kwargs):
        # creates a line plot of index data over time. used to plot the bower-index and scoop-spread index
        if ax is None:
            ax = self.gca()
        ax.axhline(linewidth=0.5, alpha=0.5, y=0, c='k')
        line, = ax.plot(t_data, y_data, **kwargs)
        ax.xaxis.set_major_locator(MultipleLocator(base=1.0))
        return line

    def _set_figure_params(self):
        self.font_size = {'S': 8, 'M': 12, 'L': 16}
