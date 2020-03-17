from helpers.data_handler import DataHandler
from helpers.plotter import Plotter


class CichFig:

    def __init__(self):
        self.pids = []
        self.dh = DataHandler()

    def add_projects(self, *pids):
        for pid in pids:
            self.pids.append(pid)
            self.dh.add_projects(pid)

    def plot_all(self):
        for pid in self.pids:
            self.dh.load_projects(pid)
            plotter = Plotter(self.dh.data_objects[pid])
            plotter.plot_all()
            self.dh.unload_projects(pid)

    def temp_plot(self):
        for pid in self.pids:
            self.dh.load_projects(pid)
            plotter = Plotter(self.dh.data_objects[pid])
            plotter.hmm_background()
            self.dh.unload_projects(pid)




