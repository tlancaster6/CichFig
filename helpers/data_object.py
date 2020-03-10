from helpers.file_manager import FileManager
from helpers.legacy.DepthAnalyzer import DepthAnalyzer as DA
from helpers.legacy.ClusterAnalyzer import ClusterAnalyzer as CA
from helpers.legacy.LogParser import LogParser as LP
from types import SimpleNamespace
import pandas as pd
import numpy as np
import os, pickle


class DataObject:

    def __init__(self, pid, file_manager):
        self.pid = pid
        self.fm = file_manager
        self.depth_data = SimpleNamespace()
        self.cluster_data = SimpleNamespace()

    def load_data(self):
        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache', 'depth_data.pkl')):
            self.unpickle_data('depth')
        else:
            print('no pkl file found. running depth data prep')
            self.prep_depth_data()

        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache', 'cluster_data.pkl')):
            self.unpickle_data('cluster')
        else:
            print('no pkl file found. running cluster data prep')
            self.prep_cluster_data()

    def prep_data(self):
        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache', 'cluster_data.pkl')):
            print('cluster data already prepped. delete cluster_data.pkl to forcibly re-prep')
        else:
            print('running cluster data prep')
            self.prep_cluster_data()

        if os.path.exists(self.fm.get_local_path(self.pid, 'data_cache', 'depth_data.pkl')):
            print('depth data already prepped. delete depth_data.pkl to forcibly re-prep')
        else:
            print('running depth data prep')
            self.prep_depth_data()

    def validate_data(self):
        # asserts that all files required for data prep are present
        data_dir = self.fm.get_local_path(self.pid)
        required = [os.path.join(data_dir, req) for req in self.fm.required_files()]
        valid = all(os.path.exists(path) for path in required)
        assert valid is True

    def prep_depth_data(self):

        self.validate_data()

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
                                              self.depth_data.hourly.height_change[i]).bowerIndex
                                              for i in range(self.depth_data.hourly.bower_locations.shape[0])]
        self.depth_data.daily.bower_index = [da.returnVolumeSummary(self.depth_data.daily.bower_locations[i],
                                             self.depth_data.daily.height_change[i]).bowerIndex
                                             for i in range(self.depth_data.daily.bower_locations.shape[0])]
        self.depth_data.total.bower_index = da.returnVolumeSummary(self.depth_data.total.bower_locations,
                                            self.depth_data.total.height_change).bowerIndex

        self.pickle_data(dtype='depth')

    def prep_cluster_data(self):

        self.validate_data()

        self.cluster_data.hourly = SimpleNamespace()
        self.cluster_data.daily = SimpleNamespace()
        self.cluster_data.total = SimpleNamespace()

        splits = self.determine_splits()
        self.cluster_data.hourly.splits = splits[0]
        self.cluster_data.daily.splits = splits[1]
        self.cluster_data.total.splits = splits[2]

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

        for bid in ca.bids:
            self.cluster_data.hourly.kdes.__dict__.update(
                {bid: np.stack([ca.returnClusterKDE(t0, t1, bid, cropped=True) for t0, t1 in splits[0]])})
            self.cluster_data.daily.kdes.__dict__.update(
                {bid: np.stack([ca.returnClusterKDE(t0, t1, bid, cropped=True) for t0, t1 in splits[1]])})
            self.cluster_data.total.kdes.__dict__.update(
                {bid: ca.returnClusterKDE(splits[2][0], splits[2][1], bid, cropped=True)})

        self.cluster_data.hourly.bower_locations = np.stack(
            [ca.returnBowerLocations(t0, t1, cropped=True, scoopKde=self.cluster_data.hourly.kdes.c[i],
             spitKde=self.cluster_data.hourly.kdes.p[i]) for i, (t0, t1) in enumerate(splits[0])])
        self.cluster_data.daily.bower_locations = np.stack(
            [ca.returnBowerLocations(t0, t1, cropped=True, scoopKde=self.cluster_data.daily.kdes.c[i],
             spitKde=self.cluster_data.daily.kdes.p[i]) for i, (t0, t1) in enumerate(splits[1])])
        self.cluster_data.total.bower_locations = ca.returnBowerLocations(
            splits[2][0], splits[2][1], cropped=True, scoopKde=self.cluster_data.total.kdes.c,
            spitKde=self.cluster_data.total.kdes.p)

        self.cluster_data.hourly.bower_index = [ca.returnClusterSummary(self.cluster_data.hourly.bower_locations[i],
                                                self.cluster_data.hourly.kdes.p[i],
                                                self.cluster_data.hourly.kdes.c[i]).bowerIndex for i in
                                                range(self.cluster_data.hourly.bower_locations.shape[0])]
        self.cluster_data.daily.bower_index = [ca.returnClusterSummary(self.cluster_data.daily.bower_locations[i],
                                               self.cluster_data.daily.kdes.p[i],
                                               self.cluster_data.daily.kdes.c[i]).bowerIndex for i in
                                               range(self.cluster_data.daily.bower_locations.shape[0])]
        self.cluster_data.total.bower_index = ca.returnClusterSummary(self.cluster_data.total.bower_locations,
                                              self.cluster_data.total.kdes.p,
                                              self.cluster_data.total.kdes.c).bowerIndex

        self.pickle_data(dtype='cluster')

    def determine_splits(self):
        lp = LP(self.generate_legacy_filemanager().localLogfile)

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
        hourly_splits = [split for split in hourly_splits if split[0].hour != 18]

        daily_splits = daily_splits.to_pydatetime().tolist()
        daily_splits = list(zip(daily_splits[:-1:2], daily_splits[1::2]))

        return hourly_splits, daily_splits, total_splits

    def pickle_data(self, dtype):
        print('pickling {} data'.format(dtype))
        if dtype == 'cluster':
            cluster_pickle = self.fm.get_local_path(self.pid, 'data_cache', 'cluster_data.pkl')
            self.fm.add_to_uploads(cluster_pickle)
            with open(cluster_pickle, 'wb') as f:
                pickle.dump(self.cluster_data, f)
            return 0 if os.path.exists(cluster_pickle) else 1
        elif dtype == 'depth':
            depth_pickle = self.fm.get_local_path(self.pid, 'data_cache', 'depth_data.pkl')
            self.fm.add_to_uploads(depth_pickle)
            with open(depth_pickle, 'wb') as f:
                pickle.dump(self.depth_data, f)
            return 0 if os.path.exists(depth_pickle) else 1

    def unpickle_data(self, dtype):
        print('unpickling {} data'.format(dtype))
        if dtype == 'cluster':
            cluster_pickle = self.fm.get_local_path(self.pid, 'data_cache', 'cluster_data.pkl')
            with open(cluster_pickle, 'rb') as f:
                self.cluster_data = pickle.load(f)
        elif dtype == 'depth':
            depth_pickle = self.fm.get_local_path(self.pid, 'data_cache', 'depth_data.pkl')
            with open(depth_pickle, 'rb') as f:
                self.depth_data = pickle.load(f)

    def generate_legacy_filemanager(self):
        pfm = SimpleNamespace()

        pfm.localLogfile = self.fm.get_local_path(self.pid, 'Logfile.txt')
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

