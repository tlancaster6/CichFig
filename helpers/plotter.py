import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import MultipleLocator
from scipy.cluster.hierarchy import linkage, dendrogram
from functools import wraps
import numpy as np
import pandas as pd
import os

bluegreen = (0.00392156862745098, 0.47843137254901963, 0.4745098039215686)
amber = (0.996078431372549, 0.7019607843137254, 0.03137254901960784)
purple = (0.49411764705882355, 0.11764705882352941, 0.611764705882353)

viridis_gapped = get_cmap('viridis', 512)
viridis_gapped = np.concatenate((viridis_gapped(np.linspace(0, 0.4, 10)), np.zeros((1, 4)),
                                 viridis_gapped(np.linspace(0.6, 1, 10))))
viridis_gapped = ListedColormap(viridis_gapped)


def plotter_wrapper(plotter_method):
    @wraps(plotter_method)
    def wrapper(plotter, fig=None):
        method_name = plotter_method.__name__
        fig = plotter.generate_fig(method_name) if fig is None else fig
        plotter.title(fig, method_name)
        plotter_method(plotter, fig)
        plotter.save_fig(fig, method_name)
    return wrapper


class Plotter:

    def __init__(self, data_object):
        self.do = data_object
        self.pid = self.do.pid
        self.bids = ['c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x']
        self.n_days = len(self.do.cluster_data.daily.splits)
        self.plot_params = self.load_params()

    def plot_multiple(self, plotting_order, f_name):
        fig = self.generate_fig(*[m.__name__ for m in plotting_order])
        fig.save_flag = False
        for method in plotting_order:
            method(fig=fig)
        fig.save_flag = True
        self.save_fig(fig, f_name)

    def plot_all_depth_and_kde(self):
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
        self.plot_multiple(plotting_order, 'plot_all_depth_and_kde')

    def plot_all_hmm(self):
        plotting_order = [
            self.hmm_background,
            self.three_fish_hmm_background,
            self.hmm_progressions
        ]
        self.plot_multiple(plotting_order, 'plot_all_hmm')


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
                    ax.set_ylabel(bid, labelpad=-10)
                if fig.current_row == start_row + 9:
                    ax.set_xlabel('day {}'.format(day), labelpad=-10)
            fig.cbar(vmin=0, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days), label='$events/cm^3$')
            fig.current_row += 1

    @plotter_wrapper
    def daily_spit_scoop_difference(self, fig=None):
        diff = self.do.cluster_data.daily.kdes.p - self.do.cluster_data.daily.kdes.c
        vmax = np.nanmax(np.abs(diff))
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            fig.array_plot(diff[day], ax=ax, vmax=vmax)
            ax.set_xlabel('day {}'.format(day), labelpad=-10)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days), label='$events/cm^3$')
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
                    ax.set_xlabel('day {}'.format(day), labelpad=-10)
                if day == 0:
                    ax.set_ylabel('{}:00 - {}:00'.format(hour + 8, hour + 9), rotation=0, size=fig.font_size['S'])
            fig.current_row += 1
        cax = fig.new_subplot(row=start_row, col=self.n_days, row_span=10)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=cax, aspect=100, label='$events/cm^3$')


    @plotter_wrapper
    def daily_heights(self, fig=None):
        vmax = np.nanquantile(np.abs(self.do.depth_data.daily.height_change), q=0.999)
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            ax.set_xlabel('day {}'.format(day), labelpad=-10)
            fig.array_plot(self.do.depth_data.daily.height_change[day], ax=ax, vmax=vmax)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=fig.new_subplot(fig.current_row, self.n_days), label='cm')
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
                    ax.set_xlabel('day {}'.format(day), labelpad=-10)
                if day == 0:
                    ax.set_ylabel('{}:00 - {}:00'.format(hour + 8, hour + 9), rotation=0, size=fig.font_size['S'])
            fig.current_row += 1
        cax = fig.new_subplot(row=start_row, col=self.n_days, row_span=10)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=cax, aspect=100, label='cm')


    @plotter_wrapper
    def daily_height_bowers(self, fig=None):
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            ax.set_xlabel('day {}'.format(day), labelpad=-10)
            fig.array_plot(self.do.depth_data.daily.bower_locations[day], ax=ax, vmax=1)
        fig.current_row += 1

    @plotter_wrapper
    def daily_cluster_bowers(self, fig=None):
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            ax.set_xlabel('day {}'.format(day), labelpad=-10)
            fig.array_plot(self.do.cluster_data.daily.bower_locations[day], ax=ax, vmax=1)
        fig.current_row += 1
        pass

    @plotter_wrapper
    def daily_bower_overlap(self, fig=None):
        for day in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=day)
            ax.set_xlabel('day {}'.format(day), labelpad=-10)
            fig.array_plot(self.do.cluster_data.daily.bower_locations[day], ax=ax, vmax=1, alpha=0.75)
            fig.array_plot(self.do.depth_data.daily.bower_locations[day], ax=ax, vmax=1, alpha=0.75)
        fig.current_row += 1

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
        diff = self.do.cluster_data.total.kdes.p - self.do.cluster_data.total.kdes.c
        vmax = np.nanmax(np.abs(diff))
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        fig.array_plot(diff, ax=ax, vmax=vmax)
        cax = fig.new_subplot(row=fig.current_row, col=self.n_days // 2 + 2, row_span=4)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=cax, label='$events/cm^3$', aspect=40)
        fig.current_row += 4

    @plotter_wrapper
    def total_correlations(self, fig=None):
        data = {'height': np.ravel(self.do.depth_data.total.height_change)}
        for bid in self.bids:
            data.update({bid: np.abs(np.ravel(self.do.cluster_data.total.kdes.__dict__[bid]))})
        data = pd.DataFrame(data).dropna(axis=0, how='any').corr().fillna(0)
        order = [data.columns.to_list()[i] for i in dendrogram(linkage(data.values), no_plot=True)['leaves']]
        data = data[order].reindex(order)
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        ax.imshow(data.values, vmin=-1, vmax=1)
        labels = data.columns.to_list()
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(data.columns.to_list(), rotation=90, size=fig.font_size['S'])
        ax.set_yticklabels(data.columns.to_list(), size=fig.font_size['S'])
        for row in range(11):
            for col in range(11):
                ax.text(row, col, '{:.2f}'.format(data.values[row, col]), ha='center', va='center',
                        size=fig.font_size['XS'])
        fig.current_row += 4

    @plotter_wrapper
    def total_kdes(self, fig=None):
        kdes = self.do.cluster_data.total.kdes.__dict__
        for bid in self.bids:
            vmax = np.nanmax(np.abs(kdes[bid]))
            ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
            fig.array_plot(kdes[bid], ax=ax, vmin=0, vmax=vmax)
            ax.set_ylabel(bid, size=fig.font_size['L'], rotation=0)
            cax = fig.new_subplot(row=fig.current_row, col=self.n_days // 2 + 2, row_span=4)
            fig.cbar(vmin=0, vmax=vmax, cax=cax, label='$events/cm^3$', aspect=40)
            fig.current_row += 4

    @plotter_wrapper
    def total_heights(self, fig=None):
        height = self.do.depth_data.total.height_change
        vmax = np.nanmax(np.abs(height))
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        fig.array_plot(height, ax=ax, vmax=vmax)
        cax = fig.new_subplot(row=fig.current_row, col=self.n_days // 2 + 2, row_span=4)
        fig.cbar(vmin=-vmax, vmax=vmax, cax=cax, label='$cm$', aspect=40)
        fig.current_row += 4

    @plotter_wrapper
    def total_height_bowers(self, fig=None):
        bowers = self.do.depth_data.total.bower_locations
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        fig.array_plot(bowers, ax=ax, vmax=1)
        fig.current_row += 4

    @plotter_wrapper
    def total_cluster_bowers(self, fig=None):
        bowers = self.do.cluster_data.total.bower_locations
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        fig.array_plot(bowers, ax=ax, vmax=1)
        fig.current_row += 4

    @plotter_wrapper
    def total_bower_overlap(self, fig=None):
        ax = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days)
        fig.array_plot(self.do.cluster_data.total.bower_locations, ax=ax, vmax=1, alpha=0.75)
        fig.array_plot(self.do.depth_data.total.bower_locations, ax=ax, vmax=1, alpha=0.75)
        fig.current_row += 4

    @plotter_wrapper
    def hmm_background(self, fig=None):
        # show how the hmm is capable of isolating the background, ignoring transient objects
        if len(self.do.hmm_data.originals) > self.n_days:
            originals = self.do.hmm_data.originals[:self.n_days]
            backgrounds = self.do.hmm_data.backgrounds[:self.n_days]
        else:
            originals = self.do.hmm_data.originals
            backgrounds = self.do.hmm_data.backgrounds
        for col in range(self.n_days):
            ax = fig.new_subplot(row=fig.current_row, col=col, despine=True)
            ax.imshow(originals[col], cmap='gray')
            ax = fig.new_subplot(row=fig.current_row + 1, col=col, despine=True)
            ax.imshow(backgrounds[col], cmap='gray')
        fig.current_row += 2

    @plotter_wrapper
    def three_fish_hmm_background(self, fig=None):
        # same as hmm_background, but instead of multiple single fish examples, shows a single 4-fish example
        original = self.do.hmm_data.three_fish_originals[0]
        background = self.do.hmm_data.three_fish_backgrounds[0]
        axes = fig.new_subplot(row=fig.current_row, col=0, row_span=4, col_span=self.n_days, despine=True, subspec=(1, 2))
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 1].imshow(background, cmap='gray')
        fig.current_row += 4

    @plotter_wrapper
    def hmm_progressions(self, fig=None):
        # show how the hmm changes before, during, and after a cluster event
        axes = fig.new_subplot(row=fig.current_row, col=0, row_span=10, col_span=self.n_days, despine=True, subspec=(10, 5))
        for row in range(20):
            for col in range(5):
                axes[row, col].imshow(self.do.hmm_data.hmm_progressions[row][col], cmap='gray')
        for col in range(5):
            axes[len(self.bids) - 1, col].set_xlabel('{} seconds'.format((col - 2) * 10))
        fig.current_row += 20

    def save_fig(self, fig, method_name):
        if fig.save_flag:
            path = self.do.fm.localFiguresDir + method_name + '.pdf'
            fig.savefig(path)
            plt.close('all')

    def load_params(self):
        params = pd.read_csv('helpers/plot_params.csv', index_col=0)
        params['w'] = self.n_days + 1
        params['c'] = 1
        return params

    def generate_fig(self, *method_names):
        params = self.plot_params.loc[list(method_names)]
        rows, cols, cell = params['h'].sum(), params['w'].max(), params['c'].max()
        return CustomFigure(nrows=rows, ncols=cols, cell_size=cell)

    def title(self, fig, method_name):
        title = self.plot_params.loc[method_name, 'title']
        title_ax = fig.new_subplot(row=fig.current_row, col=0, col_span=self.n_days, despine=True)
        title_ax.text(0.5, 0.25, r'{}'.format(title), ha='center', va='center', size=fig.font_size['L'])
        fig.current_row += 1


class CustomFigure(Figure):

    def __init__(self, nrows=1, ncols=1, cell_size=1, **kwargs):
        super().__init__(figsize=(cell_size * ncols, cell_size * nrows))
        self.spec = self.add_gridspec(nrows, ncols, hspace=0.5)
        self._set_figure_params()
        self.current_row = 0
        self.save_flag = True
        self.h = nrows
        self.w = ncols

    def new_subplot(self, row, col, row_span=1, col_span=1, despine=False, aspect='equal', subspec=None):
        # takes a row number, column number, row span, and column span, and returns a new subplot instance ax.
        # if 'despine' is set to True, the subplot axes and ticks are made invisible.
        # the parameter 'aspect' can be used to control the subplot aspect ratio (see matplotlib's add_subplot function)
        # passing a tuple (m, n) to 'subspec' will instead return an (m x n) array of equally sized subplot objects
        # occupying the space that would otherwise be allocated to the single subplot object returned when subspec is
        # None
        if subspec is None:
            ax = self.add_subplot(self.spec[row: row + row_span, col: col + col_span], aspect=aspect)
            if despine:
                sns.despine(ax=ax, left=True, bottom=True)
                ax.tick_params(colors=[0, 0, 0, 0])
            return ax
        else:
            inner_spec = self.spec[row: row + row_span, col: col + col_span].subgridspec(subspec[0], subspec[1])
            ax = []
            for r in range(subspec[0]):
                ax.append([])
                for c in range(subspec[1]):
                    ax[r].append(self.add_subplot(inner_spec[r, c], aspect='equal'))
                    if despine:
                        sns.despine(ax=ax[r][c], left=True, bottom=True)
                        ax[r][c].tick_params(colors=[0, 0, 0, 0])
            return np.array(ax)

    def array_plot(self, array, ax=None, vmin=None, vmax=None, cmap='viridis', **kwargs):
        # uses matplotlib.pyplot.imshow() to create a colorized image of array. Used to plot depth change and kde arrays
        if ax is None:
            ax = self.gca()
        if vmin is None and vmax is None:
            vmax = np.nanmax(np.abs(array))
            vmin = -vmax
        elif vmin is None or vmax is None:
            vmin = vmin if vmax is None else -vmax
            vmax = vmax if vmin is None else -vmin
        plot = ax.imshow(array, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax, **kwargs)
        ax.tick_params(colors=[0, 0, 0, 0])
        return plot

    def cbar(self, vmin, vmax, cax, aspect=10, label=None, label_size='S'):
        cax.set(aspect=aspect, anchor='W')
        mappable = ScalarMappable(Normalize(vmin, vmax), 'viridis')
        cbar = plt.colorbar(mappable, cax, format='% 4.1f')
        cax.tick_params(labelsize=self.font_size['S'])

        if label is not None:
            cax.text(10, 0.5, r'{}'.format(label), ha='center', va='center', size=self.font_size[label_size],
                     rotation=270, transform=cax.transAxes)
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
        self.font_size = {'XS': 5, 'S': 8, 'M': 12, 'L': 16}
