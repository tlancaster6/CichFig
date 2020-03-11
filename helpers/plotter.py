import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from functools import wraps

import numpy as np
import pandas as pd
import os

bluegreen = (0.00392156862745098, 0.47843137254901963, 0.4745098039215686)
amber = (0.996078431372549, 0.7019607843137254, 0.03137254901960784)
purple = (0.49411764705882355, 0.11764705882352941, 0.611764705882353)


def plotter_wrapper(plotter_method):
    def wrapper(plotter, fig):
        method_name = plotter_method.__name__
        fig = plotter.generate_fig(method_name) if fig is None else fig
        plotter.title(fig, method_name)
        plotter_method(plotter, fig)
        plotter.save_fig(fig, method_name)
    return wrapper


class Plotter:

    def __init__(self, data_object, file_manager):
        self.do = data_object
        self.fm = file_manager
        self.pid = self.do.pid
        self.bids = ['c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x']
        self.n_days = len(self.do.cluster_data.daily.splits)
        self.plot_params = self.load_params()

    def plot_all(self):
        method_name = 'plot_all'
        fig = self.generate_fig(method_name)
        fig.save_flag = False
        plotting_order = [
            self.total_kdes,
            self.total_correlations,
            self.total_spit_scoop_difference,
            self.total_cluster_bowers,
            self.total_heights,
            self.total_height_bowers,
            self.total_bower_overlap,
            self.daily_kdes,
            self.daily_spit_scoop_difference,
            self.daily_cluster_bowers,
            self.daily_heights,
            self.daily_height_bowers,
            self.daily_bower_overlap,
            self.daily_indices,
            self.hourly_spit_scoop_difference,
            self.hourly_heights
        ]
        for method in plotting_order:
            method(fig=fig)
        fig.save_flag = True
        self.save_fig(fig, method_name)

    @plotter_wrapper
    def daily_kdes(self, fig=None):
        kdes = self.do.cluster_data.daily.kdes.__dict__
        start_row = fig.current_row
        for bid in self.bids:
            vmax = np.nanmax(np.abs(kdes[bid]))
            for day in range(self.n_days):
                ax = fig.new_subplot(row=fig.current_row, col=day)
                fig.array_plot(kdes[bid][day], ax=ax, vmin=0, vmax=vmax)
                if day == 0:
                    ax.set_ylabel(bid)
                if fig.current_row == start_row + 10:
                    ax.set_xlabel('day {}'.format(day))
            fig.cbar(vmin=0, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days))
            fig.current_row += 1

    @plotter_wrapper
    def daily_spit_scoop_difference(self, fig=None):
        diff = self.do.cluster_data.daily.kdes.p - self.do.cluster_data.daily.kdes.c
        vmax = np.nanmax(np.abs(diff))
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            fig.array_plot(diff[day], ax=ax, vmax=vmax)
            ax.set_xlabel('day {}'.format(day))
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days))
        fig.current_row += 1

    @plotter_wrapper
    def hourly_spit_scoop_difference(self, fig=None):
        diff = self.do.cluster_data.hourly.kdes.p - self.do.cluster_data.hourly.kdes.c
        vmax = np.nanmax(np.abs(diff))
        start_row = fig.current_row
        for hour in range(10):
            for day in range(self.n_days):
                ax = fig.new_subplot(row=fig.current_row, col=day)
                fig.array_plot(diff[10 * day + hour], ax=ax, vmax=vmax)
                if hour == 9:
                    ax.set_xlabel('day {}'.format(day))
                if day == 0:
                    ax.set_ylabel('{}:00 - {}:00'.format(hour + 8, hour + 9), rotation=0, size=fig.font_size['S'])
            fig.current_row += 1
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(row=start_row, col=self.n_days, row_span=10), aspect=20)


    @plotter_wrapper
    def daily_heights(self, fig=None):
        vmax = np.nanquantile(np.abs(self.do.depth_data.daily.height_change), q=0.999)
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            ax.set_xlabel('day {}'.format(day))
            fig.array_plot(self.do.depth_data.daily.height_change[day], ax=ax, vmax=vmax)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days))
        fig.current_row += 1

    @plotter_wrapper
    def hourly_heights(self, fig=None):
        height = self.do.depth_data.hourly.height_change
        vmax = np.nanquantile(np.abs(height), q=0.999)
        start_row = fig.current_row
        for hour in range(10):
            for day in range(self.n_days):
                ax = fig.new_subplot(row=fig.current_row, col=day)
                fig.array_plot(height[10 * day + hour], ax=ax, vmax=vmax)
                if hour == 9:
                    ax.set_xlabel('day {}'.format(day))
                if day == 0:
                    ax.set_ylabel('{}:00 - {}:00'.format(hour + 8, hour + 9), rotation=0, size=fig.font_size['S'])
            fig.current_row += 1
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(row=start_row, col=self.n_days, row_span=10), aspect=20)


    @plotter_wrapper
    def daily_height_bowers(self, fig=None):
        pass

    @plotter_wrapper
    def daily_cluster_bowers(self, fig=None):
        pass

    @plotter_wrapper
    def daily_bower_overlap(self, fig=None):
        pass

    @plotter_wrapper
    def daily_indices(self, fig=None):
        days = list(range(self.n_days))
        plotting_ax = fig.new_subplot(row=fig.current_row, col=0, row_span=2, col_span=self.n_days)
        fig.index_plot(t_data=days, y_data=self.do.cluster_data.daily.bower_index, ax=plotting_ax,
                       label='from cnn results', c=amber)
        fig.index_plot(t_data=days, y_data=self.do.depth_data.daily.bower_index, ax=plotting_ax,
                       label='from depth data', c=purple)
        plotting_ax.legend(ncol=2, fontsize=fig.font_size['S'])
        fig.current_row += 2

    @plotter_wrapper
    def total_spit_scoop_difference(self, fig=None):
        pass

    @plotter_wrapper
    def total_correlations(self, fig=None):
        pass

    @plotter_wrapper
    def total_kdes(self, fig=None):
        pass

    @plotter_wrapper
    def total_heights(self, fig=None):
        pass

    @plotter_wrapper
    def total_height_bowers(self, fig=None):
        pass

    @plotter_wrapper
    def total_cluster_bowers(self, fig=None):
        pass

    @plotter_wrapper
    def total_bower_overlap(self, fig=None):
        pass

    def save_fig(self, fig, method_name):
        if fig.save_flag:
            path = self.fm.get_local_path(os.path.join(self.pid, self.plot_params.loc[method_name, 'fname']))
            fig.savefig(path)
            plt.close('all')

    def load_params(self):
        params = pd.read_csv('helpers/plot_params.csv', index_col=0)
        params['w'] = self.n_days + 1
        params.loc['plot_all', 'h'] = params['h'].sum()
        return params

    def generate_fig(self, method_name):
        params = self.plot_params.loc[method_name]
        rows, cols, cell = params['h'], params['w'], params['c']
        return CustomFigure(nrows=rows, ncols=cols, cell_size=cell)

    def title(self, fig, method_name):
        title = self.plot_params.loc[method_name, 'title']
        title_ax = fig.new_subplot(row=fig.current_row, col=0, col_span=fig.w - 1, despine=True)
        title_ax.text(0.5, 0.5, r'{}'.format(title), ha='center', va='center', size=fig.font_size['L'])
        fig.current_row += 1


class CustomFigure(Figure):

    def __init__(self, nrows=1, ncols=1, cell_size=1, **kwargs):
        super().__init__(figsize=(cell_size * ncols, cell_size * nrows))
        self.spec = self.add_gridspec(nrows, ncols, hspace=0.4)
        self._set_figure_params()
        self.current_row = 0
        self.save_flag = True
        self.h = nrows
        self.w = ncols
        self.c = cell_size

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

    def cbar(self, vmin, vmax, cax, aspect=10):
        cax.set(aspect=aspect, anchor='W')
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
