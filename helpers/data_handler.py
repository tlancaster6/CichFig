from helpers.data_object import DataObject as DO
from helpers.legacy.FileManager import FileManager as FM
import pandas as pd


class DataHandler:

    def __init__(self):
        self.data_objects = {}
        self.multiproject_data = self.load_multiproject_data()

    def add_projects(self, *pids):
        self.data_objects.update({pid: DO(pid) for pid in pids})

    def load_projects(self, *pids):
        [self.data_objects[pid].load_data() for pid in pids]

    def unload_projects(self, *pids):
        self.data_objects.update({pid: DO(pid) for pid in pids})

    def prep_projects(self, *pids, unload=True):
        for pid in pids:
            self.data_objects[pid].prep_data()
            if unload:
                self.unload_projects(pid)

    def load_multiproject_data(self):
        return pd.read_csv(FM().localMultiProjectData, index_col='project')




