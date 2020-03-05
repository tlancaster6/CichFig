from helpers.data_object import DataObject

class DataHandler:

    def __init__(self):
        self.data_objects = {}

    def add_project(self, pid):
        self.data_objects.update({pid, DataObject(pid)})
        pass
