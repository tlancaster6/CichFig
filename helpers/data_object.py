from helpers.legacy.FileManager import FileManager
from helpers.legacy.DepthAnalyzer import DepthAnalyzer as DA
from helpers.legacy.ClusterAnalyzer import ClusterAnalyzer as CA
from helpers.legacy.LogParser import LogParser as LP
from helpers.legacy.HMMAnalyzer import HMMAnalyzer as HA
from types import SimpleNamespace
from PIL import Image
import pandas as pd
import numpy as np
import os, pickle, re


class DataObject:

    def __init__(self, pid):
        self.pid = pid
        self.ca = None
        self.da = None
        self.lp = None
        self.ha = None
        self.fm = FileManager(self.pid)
        self.depth_data = SimpleNamespace()
        self.cluster_data = SimpleNamespace()
        self.hmm_data = SimpleNamespace()

    def load_data(self):
        if not os.path.exists(self.fm.localProjectDir):
            self.fm.downloadProjectData(dtype='Figures')

        self.da = DA(self.fm) if self.da is None else self.da
        self.ca = CA(self.fm) if self.ca is None else self.ca
        self.lp = LP(self.fm.localLogfile) if self.lp is None else self.lp

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

        if os.path.exists(self.fm.localHmmPickle):
            self.unpickle_data('hmm')
        else:
            print('no pkl file found. running hmm data prep')
            self.prep_hmm_data()

        self.update_multiproject_data()

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
        # prep data for the hmm_background and 3_fish_hmm_background plots
        box_size = 300
        boxed_fish = pd.read_csv(self.fm.localBoxedFishFile, index_col=0)
        if self.pid in boxed_fish.ProjectID.values:
            self.fm.downloadData(self.fm.localBoxedFishDir + self.pid, tarred=True)
            boxed_fish = boxed_fish[boxed_fish.ProjectID == self.pid]
            boxed_fish = boxed_fish[boxed_fish.Nfish >= 1]
            boxed_fish_3 = boxed_fish[boxed_fish.Nfish == 3]
            vid = re.search(r'\d+(?=_vid)', boxed_fish_3.iloc[0].Framefile).group(0)
            for df in boxed_fish, boxed_fish_3:
                df = df[[vid in fname for fname in df.Framefile.str.split('_').to_list()]]
                df.drop_duplicates(subset='Framefile', inplace=True)

            def square_box(box, size):
                box = tuple(map(int, box.strip('()').split(',')))
                return box[0] + box[2]//2 - size//2, box[1] + box[3]//2 - size//2, size, size

            boxed_fish.Box = boxed_fish.apply(lambda row: square_box(row.Box, box_size), axis=1)
            self.fm.downloadData(self.fm.localTroubleshootingDir + vid + '_vid.hmm.npy')
            self.fm.downloadData(self.fm.localTroubleshootingDir + vid + '_vid.hmm.txt')
            self.ha = HA(self.fm.returnVideoObject(int(vid) - 1).localHMMFile)

            self.hmm_data.originals = []
            self.hmm_data.backgrounds = []
            d = self.fm.localBoxedFishDir + self.pid + '/'
            for index, row in boxed_fish.iterrows():
                crop = np.s_[row.Box[1]: row.Box[1] + row.Box[3], row.Box[0]: row.Box[0] + row.Box[2]]
                original = np.asarray(Image.open(d + row.Framefile).convert('L'))[crop]
                background = self.ha.retImage(index)[crop]
                if original.shape == background.shape == (box_size, box_size):
                    self.hmm_data.originals.append(original)
                    self.hmm_data.backgrounds.append(background)

            self.hmm_data.three_fish_originals = []
            self.hmm_data.three_fish_backgrounds = []
            for index, row in boxed_fish_3.iterrows():
                self.hmm_data.three_fish_originals.append(np.asarray(Image.open(d + row.Framefile).convert('L')))
                self.hmm_data.three_fish_backgrounds.append(self.ha.retImage(index))

        # prep data for the hmm progressions plot
        self.hmm_data.hmm_progressions = []
        # find a day with a wide variety of cluster types (preferably all 10 types)
        best_day = self.ca.clusterData.groupby('videoID')['Model18_All_pred'].unique().apply(len).argmax()
        # download the hmm.npy and hmm.txt files associated with best_day
        self.fm.downloadData(self.fm.localTroubleshootingDir + '{:04d}'.format(best_day + 1) + '_vid.hmm.npy')
        self.fm.downloadData(self.fm.localTroubleshootingDir + '{:04d}'.format(best_day + 1) + '_vid.hmm.txt')
        # generate an hmm analyzer object for best_day
        self.ha = HA(self.fm.returnVideoObject(int(best_day)).localHMMFile)
        # slice the cluster data to only include clusters from best_day
        t0 = self.lp.movies[best_day].startTime.replace(hour=8, minute=0, second=0, microsecond=0)
        t1 = t0.replace(hour=18)
        df = self.ca.sliceDataframe(t0=t0, t1=t1, columns=['X', 'Y', 't', 'Model18_All_pred', 'N'])
        # further filter the cluster data to only include clusters with N > 1500, then pick 20 clusters at random
        df = df[df.N > 1500].sample(20)
        # for each of the 10 chosen clusters, get the associated hmm image and crop it to 50x50
        framerate = self.lp.movies[best_day].framerate
        for index, event in df.iterrows():
            crop = np.s_[int(event.X) - 25: int(event.X) + 25, int(event.Y) - 25: int(event.Y) + 25]
            frames = np.linspace((event.t - 20) * framerate,  (event.t + 20) * framerate, 5)
            self.hmm_data.hmm_progressions.append([self.ha.retImage(t)[crop] for t in frames])
        # pickle the hmm object
        self.pickle_data(dtype='hmm')

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
        lp = LP(self.fm.localLogfile) if self.lp is None else self.lp

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
        elif dtype == 'hmm':
            with open(self.fm.localHmmPickle, 'wb') as f:
                pickle.dump(self.hmm_data, f)

    def unpickle_data(self, dtype):
        print('unpickling {} data'.format(dtype))
        if dtype == 'cluster':
            with open(self.fm.localClusterPickle, 'rb') as f:
                self.cluster_data = pickle.load(f)
        elif dtype == 'depth':
            with open(self.fm.localDepthPickle, 'rb') as f:
                self.depth_data = pickle.load(f)
        elif dtype == 'hmm':
            with open(self.fm.localHmmPickle, 'rb') as f:
                self.hmm_data = pickle.load(f)

    def update_multiproject_data(self):
        df = pd.read_csv(self.fm.localMultiProjectData, index_col=0).to_dict('index')
        df.update({self.pid: {'n_transitions': 0, 'n_assigned_transitions': 0, 'n_clusters': 0}})
        df = pd.DataFrame(df).T if len(df) == 1 else pd.DataFrame(df)
        pd.DataFrame(df).to_csv(self.fm.localMultiProjectData)



