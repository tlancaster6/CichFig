from helpers.data_handler import DataHandler
from helpers.plotter import Plotter
from time import time


class CichFig:

    def __init__(self):
        self.pids = []
        self.dh = DataHandler()

    def add_projects(self, *pids):
        for pid in pids:
            self.pids.append(pid)
            self.dh.add_projects(pid)

    def plot(self, *plotting_order, name=None):
        # if a single plotter method is passed, 'name' is set automatically to that method's __name__. If multiple
        # plotter methods are passed, and name is not specified, the name defaults to 'plot_multiple_*', with *
        # replaced by the unix timestamp. Each value passed to plotting order should be a string corresponding to a
        # method of the plotter object.
        n_methods = len(locals()) - 2
        for pid in self.pids:
            self.dh.load_projects(pid)
            plotter = Plotter(self.dh.data_objects[pid])
            plotting_order = [getattr(plotter, plotter_method) for plotter_method in plotting_order]
            if n_methods > 1:
                name = 'plot_multiple_' + str(int(time())) if name is None else name
                getattr(plotter, 'plot_multiple')(plotting_order, name)
            else:
                plotting_order[0]()
            self.dh.unload_projects(pid)


