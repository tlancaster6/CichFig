from helpers.file_manager import FileManager
from helpers.legacy.DepthAnalyzer import DepthAnalyzer as DA
from helpers.legacy.ClusterAnalyzer import ClusterAnalyzer as CA
from helpers.legacy.LogParser import LogParser as LP
from types import SimpleNamespace
import pandas as pd
import numpy as np
import os


class DataObject:

    def __init__(self, pid):
        self.pid = pid
        self.fm = FileManager()
        self.depth_data = SimpleNamespace()
        self.cluster_data = SimpleNamespace()
        pass

    def full_auto(self):
        self.validate_data()
        self.prep_cluster_data()
        self.prep_depth_data()

    def validate_data(self):
        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache')):
            return 0
        else:
            data_dir = self.fm.get_local_path(self.pid)
            required = [os.path.join(data_dir, req) for req in self.fm.required_files()]
            valid = all(os.path.exists(path) for path in required)
            return int(not valid)

    def determine_splits(self):
        lp = LP(self.generate_legacy_filemanager().localLogFile)

        total_start = pd.Timestamp(max(lp.movies[0].startTime, lp.frames[0].time)).ceil('H')
        total_stop = pd.Timestamp(min(lp.movies[-1].endTime, lp.frames[-1].time)).floor('H')
        total_splits = [total_start.to_pydatetime(), total_stop.to_pydatetime()]

        hourly_splits = pd.date_range(total_start, total_stop, freq='H')
        hourly_splits = hourly_splits[(hourly_splits.hour >= 8) & (hourly_splits.hour <= 18)]

        daily_splits = hourly_splits[(hourly_splits.hour == 8) | (hourly_splits.hour == 18)]
        if daily_splits[0].hour == 18:
            daily_splits = daily_splits[1:]
        if daily_splits[-1].hour == 8:
            daily_splits = daily_splits[:-1]

        hourly_splits = hourly_splits.to_pydatetime().tolist()
        hourly_splits = list(zip(hourly_splits[:-1], hourly_splits[1:]))
        daily_splits = daily_splits.to_pydatetime().tolist()
        daily_splits = list(zip(daily_splits[:-1], daily_splits[1:]))

        return hourly_splits, daily_splits, total_splits

    def prep_depth_data(self):

        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache')):
            pass
        else:
            self.depth_data.hourly = SimpleNamespace()
            self.depth_data.daily = SimpleNamespace()
            self.depth_data.total = SimpleNamespace()

            splits = self.determine_splits()
            self.depth_data.hourly.splits = splits[0]
            self.depth_data.daily.splits = splits[1]
            self.depth_data.total.splits = splits[2]

            da = DA(self.generate_legacy_filemanager())

            self.depth_data.hourly.height_change = np.stack(
                [da.returnHeightChange(t0, t1, cropped=True) for t0, t1 in splits[0]])
            self.depth_data.daily.height_change = np.stack(
                [da.returnHeightChange(t0, t1, cropped=True) for t0, t1 in splits[1]])
            self.depth_data.total.height_change = da.returnHeightChange(splits[2][0], splits[2][1], cropped=True)

            self.depth_data.hourly.bower_locations = np.stack(
                [da.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[0]])
            self.depth_data.daily.bower_locations = np.stack(
                [da.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[1]])
            self.depth_data.total.bower_locations = da.returnBowerLocations(splits[2][0], splits[2][1], cropped=True)

            self.depth_data.hourly.bower_index = [da.returnVolumeSummary(self.depth_data.hourly.bower_locations[i],
                                                  self.depth_data.hourly.height_change[i]).BowerIndex
                                                  for i in self.depth_data.hourly.bower_locations.shape[0]]
            self.depth_data.daily.bower_index = [da.returnVolumeSummary(self.depth_data.daily.bower_locations[i],
                                                 self.depth_data.daily.height_change[i]).BowerIndex
                                                 for i in self.depth_data.daily.bower_locations.shape[0]]
            self.depth_data.total.bower_index = da.returnVolumeSummary(self.depth_data.total.bower_locations,
                                                self.depth_data.total.height_change).BowerIndex


    def prep_cluster_data(self):
        self.cluster_data.hourly = SimpleNamespace()
        self.cluster_data.daily = SimpleNamespace()
        self.cluster_data.total = SimpleNamespace()

        splits = self.determine_splits()
        self.cluster_data.hourly.splits = splits[0]
        self.cluster_data.daily.splits = splits[1]
        self.cluster_data.total.splits = splits[2]

        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache')):
            pass
        else:
            self.cluster_data.hourly = SimpleNamespace()
            self.cluster_data.daily = SimpleNamespace()
            self.cluster_data.total = SimpleNamespace()

            splits = self.determine_splits()
            self.cluster_data.hourly.splits = splits[0]
            self.cluster_data.daily.splits = splits[1]
            self.cluster_data.total.splits = splits[2]

            ca = CA(self.generate_legacy_filemanager())

            self.cluster_data.hourly.kdes = SimpleNamespace()
            self.cluster_data.daily.kdes = SimpleNamespace()
            self.cluster_data.total.kdes = SimpleNamespace()

            self.cluster_data.hourly.height_change = np.stack(
                [ca.returnClusterKDE(t0, t1, cropped=True) for t0, t1 in splits[0]])
            self.cluster_data.daily.height_change = np.stack(
                [ca.returnClusterKDE(t0, t1, cropped=True) for t0, t1 in splits[1]])
            self.cluster_data.total.height_change = ca.returnClusterKDE(splits[2][0], splits[2][1], cropped=True)

            self.cluster_data.hourly.bower_locations = np.stack(
                [ca.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[0]])
            self.cluster_data.daily.bower_locations = np.stack(
                [ca.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[1]])
            self.cluster_data.total.bower_locations = ca.returnBowerLocations(splits[2][0], splits[2][1], cropped=True)

            self.cluster_data.hourly.bower_index = [ca.returnClusterSummary(self.cluster_data.hourly.bower_locations[i],
                                                  self.cluster_data.hourly.height_change[i]).BowerIndex
                                                  for i in self.cluster_data.hourly.bower_locations.shape[0]]
            self.cluster_data.daily.bower_index = [ca.returnClusterSummary(self.cluster_data.daily.bower_locations[i],
                                                 self.cluster_data.daily.height_change[i]).BowerIndex
                                                 for i in self.cluster_data.daily.bower_locations.shape[0]]
            self.cluster_data.total.bower_index = ca.returnClusterSummary(self.cluster_data.total.bower_locations,
                                                self.cluster_data.total.height_change).BowerIndex

    def generate_legacy_filemanager(self):
        pfm = SimpleNamespace()

        pfm.localLogFile = self.fm.get_local_path(self.pid, 'Logfile.txt')
        pfm.localTransMFile = self.fm.get_local_path(self.pid, 'MasterAnalysisFiles/TransMFile.npy')
        pfm.localAllLabeledClustersFile = self.fm.get_local_path(self.pid, 'MasterAnalysisFiles/AllLabeledClusters.csv')
        pfm.localTrayFile = self.fm.get_local_path(self.pid, 'MasterAnalysisFiles/DepthCrop.txt')
        pfm.localSmoothDepthFile = self.fm.get_local_path(self.pid, 'MasterAnalysisFiles/smoothedDepthData.npy')

        pfm.hourlyDepthThreshold = 0.2  # cm
        pfm.dailyDepthThreshold = 0.4  # cm
        pfm.totalDepthThreshold = 1.0  # cm
        pfm.hourlyClusterThreshold = 1.0  # events/cm^2
        pfm.dailyClusterThreshold = 2.0  # events/cm^2
        pfm.totalClusterThreshold = 5.0  # events/cm^2
        pfm.pixelLength = 0.1030168618  # cm / pixel
        pfm.hourlyMinPixels = 1000
        pfm.dailyMinPixels = 1000
        pfm.totalMinPixels = 1000
        pfm.bowerIndexFraction = 0.1
        pfm.lightsOnTime = 8
        pfm.lightsOffTime = 18

        return pfm

