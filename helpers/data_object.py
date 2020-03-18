from helpers.legacy.FileManager import FileManager
from helpers.legacy.DepthAnalyzer import DepthAnalyzer as DA
from helpers.legacy.ClusterAnalyzer import ClusterAnalyzer as CA
from helpers.legacy.LogParser import LogParser as LP
from helpers.legacy.HMMAnalyzer import HMMAnalyzer as HA
from types import SimpleNamespace
import pandas as pd
import numpy as np
import os, pickle


class DataObject:

    def __init__(self, pid):
        self.pid = pid
        self.ca = None
        self.da = None
        self.lp = None
        self.fm = FileManager(self.pid)
        self.depth_data = SimpleNamespace()
        self.cluster_data = SimpleNamespace()

    def load_data(self):
        if not os.path.exists(self.fm.localProjectDir):
            self.fm.downloadProjectData(dtype='Figures')
        if os.path.exists(self.fm.localDepthPickle):
            self.unpickle_data('depth')
        else:
            print('no pkl file found. running depth data prep')
            self.prep_depth_data()

        if os.path.exists(self.fm.localClusterPickle):
            self.unpickle_data('cluster')
        else:
            print('no pkl file found. running cluster data prep')
            self.prep_cluster_data()
        self.lp = LP(self.fm.localLogfile)

    def prep_data(self):
        if not os.path.exists(self.fm.localProjectDir):
            self.fm.downloadProjectData(dtype='Figures')
        print('running cluster data prep')
        self.prep_cluster_data()
        print('running depth data prep')
        self.prep_depth_data()
        print('running hmm data prep')
        self.prep_hmm_data()

    def prep_hmm_data(self):
        pass

    def prep_depth_data(self):

        self.depth_data.hourly = SimpleNamespace()
        self.depth_data.daily = SimpleNamespace()
        self.depth_data.total = SimpleNamespace()

        splits = self.determine_splits()
        self.depth_data.hourly.splits = splits[0]
        self.depth_data.daily.splits = splits[1]
        self.depth_data.total.splits = splits[2]

        self.da = DA(self.fm)

        self.depth_data.hourly.height_change = np.stack(
            [self.da.returnHeightChange(t0, t1, cropped=True) for t0, t1 in splits[0]])
        self.depth_data.daily.height_change = np.stack(
            [self.da.returnHeightChange(t0, t1, cropped=True) for t0, t1 in splits[1]])
        self.depth_data.total.height_change = self.da.returnHeightChange(splits[2][0], splits[2][1], cropped=True)

        self.depth_data.hourly.bower_locations = np.stack(
            [self.da.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[0]])
        self.depth_data.daily.bower_locations = np.stack(
            [self.da.returnBowerLocations(t0, t1, cropped=True) for t0, t1 in splits[1]])
        self.depth_data.total.bower_locations = self.da.returnBowerLocations(splits[2][0], splits[2][1], cropped=True)

        self.depth_data.hourly.bower_index = [self.da.returnVolumeSummary(self.depth_data.hourly.bower_locations[i],
                                              self.depth_data.hourly.height_change[i]).bowerIndex
                                              for i in range(self.depth_data.hourly.bower_locations.shape[0])]
        self.depth_data.daily.bower_index = [self.da.returnVolumeSummary(self.depth_data.daily.bower_locations[i],
                                             self.depth_data.daily.height_change[i]).bowerIndex
                                             for i in range(self.depth_data.daily.bower_locations.shape[0])]
        self.depth_data.total.bower_index = self.da.returnVolumeSummary(self.depth_data.total.bower_locations,
                                            self.depth_data.total.height_change).bowerIndex

        self.pickle_data(dtype='depth')

    def prep_cluster_data(self):

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

        self.ca = CA(self.fm)

        self.cluster_data.hourly.kdes = SimpleNamespace()
        self.cluster_data.daily.kdes = SimpleNamespace()
        self.cluster_data.total.kdes = SimpleNamespace()

        for bid in self.ca.bids:
            self.cluster_data.hourly.kdes.__dict__.update(
                {bid: np.stack([self.ca.returnClusterKDE(t0, t1, bid, cropped=True) for t0, t1 in splits[0]])})
            self.cluster_data.daily.kdes.__dict__.update(
                {bid: np.stack([self.ca.returnClusterKDE(t0, t1, bid, cropped=True) for t0, t1 in splits[1]])})
            self.cluster_data.total.kdes.__dict__.update(
                {bid: self.ca.returnClusterKDE(splits[2][0], splits[2][1], bid, cropped=True)})

        self.cluster_data.hourly.bower_locations = np.stack(
            [self.ca.returnBowerLocations(t0, t1, cropped=True, scoopKde=self.cluster_data.hourly.kdes.c[i],
             spitKde=self.cluster_data.hourly.kdes.p[i]) for i, (t0, t1) in enumerate(splits[0])])
        self.cluster_data.daily.bower_locations = np.stack(
            [self.ca.returnBowerLocations(t0, t1, cropped=True, scoopKde=self.cluster_data.daily.kdes.c[i],
             spitKde=self.cluster_data.daily.kdes.p[i]) for i, (t0, t1) in enumerate(splits[1])])
        self.cluster_data.total.bower_locations = self.ca.returnBowerLocations(
            splits[2][0], splits[2][1], cropped=True, scoopKde=self.cluster_data.total.kdes.c,
            spitKde=self.cluster_data.total.kdes.p)

        self.cluster_data.hourly.bower_index = [self.ca.returnClusterSummary(self.cluster_data.hourly.bower_locations[i],
                                                self.cluster_data.hourly.kdes.p[i],
                                                self.cluster_data.hourly.kdes.c[i]).bowerIndex for i in
                                                range(self.cluster_data.hourly.bower_locations.shape[0])]
        self.cluster_data.daily.bower_index = [self.ca.returnClusterSummary(self.cluster_data.daily.bower_locations[i],
                                               self.cluster_data.daily.kdes.p[i],
                                               self.cluster_data.daily.kdes.c[i]).bowerIndex for i in
                                               range(self.cluster_data.daily.bower_locations.shape[0])]
        self.cluster_data.total.bower_index = self.ca.returnClusterSummary(self.cluster_data.total.bower_locations,
                                              self.cluster_data.total.kdes.p,
                                              self.cluster_data.total.kdes.c).bowerIndex

        self.pickle_data(dtype='cluster')

    def determine_splits(self):
        lp = LP(self.fm.localLogfile)

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
            with open(self.fm.localClusterPickle, 'wb') as f:
                pickle.dump(self.cluster_data, f)
        elif dtype == 'depth':
            with open(self.fm.localDepthPickle, 'wb') as f:
                pickle.dump(self.depth_data, f)

    def unpickle_data(self, dtype):
        print('unpickling {} data'.format(dtype))
        if dtype == 'cluster':
            with open(self.fm.localClusterPickle, 'rb') as f:
                self.cluster_data = pickle.load(f)
        elif dtype == 'depth':
            with open(self.fm.localDepthPickle, 'rb') as f:
                self.depth_data = pickle.load(f)

    def update_multiproject_data(self):
        df = pd.read_csv(self.fm.localMultiProjectData, index_col='project')


