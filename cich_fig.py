from helpers.file_manager import FileManager
from helpers.data_handler import DataHandler
import helpers.plotting_utilities as plute


class CichFig:

    def __init__(self):
        self.fm = FileManager()
        self.dh = DataHandler()

    def full_auto(self):
        for pid in self.fm.identify_projects():
            self.add_project(pid)

    def add_project(self, pid):
        self.fm.add_project(pid)
        self.dh.add_project(pid)




