from helpers.data_object import DataObject


class DataHandler:

    def __init__(self):
        self.data_objects = {}

    def add_projects(self, *pids):
        self.data_objects.update({pid: DataObject(pid) for pid in pids})

    def load_projects(self, *pids):
        [self.data_objects[pid].load_data() for pid in pids]

    def unload_projects(self, *pids):
        self.data_objects.update({pid: DataObject(pid) for pid in pids})

    def prep_projects(self, *pids, unload=True):
        for pid in pids:
            self.data_objects[pid].prep_data()
            if unload:
                self.unload_projects(pid)


