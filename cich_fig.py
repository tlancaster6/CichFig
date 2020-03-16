from helpers.file_manager import FileManager
from helpers.data_handler import DataHandler
from helpers.plotter import Plotter


class CichFig:

    def __init__(self):
        self.pids = []
        self.fm = FileManager()
        self.dh = DataHandler(self.fm)

    def identify_projects(self):
        print('identifying projects ready for figure creation. This might take a while')
        projects = self.fm.identify_projects()
        print('projects with the files required for figure creation:')
        print('\n'.join(projects))
        return projects

    def add_projects(self, *pids):
        for pid in pids:
            self.pids.append(pid)
            self.fm.add_projects(pid)
            self.dh.add_projects(pid)

    def plot_all(self):
        for pid in self.pids:
            self.dh.load_projects(pid)
            plotter = Plotter(self.dh.data_objects[pid], self.fm)
            plotter.plot_all()
            self.dh.unload_projects(pid)





