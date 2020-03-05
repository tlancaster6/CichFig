import subprocess, os
from datetime import datetime as dt


class FileManager:

    def __init__(self):
        self.pids = []
        self.log = self.read_analysis_log()
        self.uploads = []

    def required_files(self):
        return ['Logfile.txt',
                'MasterAnalysisFiles/AllLabeledClusters.csv',
                'MasterAnalysisFiles/smoothedDepthData.npy',
                'MasterAnalysisFiles/TransMFile.npy',
                'MasterAnalysisFiles/DepthCrop.txt']

    def get_local_path(self, *args):
        return os.path.join(os.getenv('HOME'), 'scratch', *args)

    def get_cloud_path(self, *args):
        return os.path.join('cichlidVideo:BioSci-McGrath/Apps/CichlidPiData/', *args)

    def identify_projects(self):
        required = self.required_files()
        cmnd = ['rclone', 'lsf', '--dirs-only', self.get_cloud_path()]
        projects = set(subprocess.run(cmnd, capture_output=True, encoding='utf-8').stdout.split())
        bad_projects = set()
        for proj in projects:
            for req in required:
                if not self.check_exists_cloud(os.path.join(proj, req)):
                    bad_projects.add(proj)
                    break
        projects = list(projects - bad_projects)
        return projects

    def check_exists_cloud(self, relative_path):
        cmnd = ['rclone', 'lsf', self.get_cloud_path(relative_path)]
        exists = subprocess.run(cmnd, capture_output=True, encoding='utf-8').stdout != ''
        return exists

    def add_project(self, pid):
        self.pids.append(pid)
        self.construct_filesystem(pid)
        if ~os.path.exists(self.get_local_path(pid, 'data_cache')):
            self.download_required_files(pid)

    def construct_filesystem(self, pid):
        for sub_dir in ['Figures', 'MasterAnalysisFiles']:
            path = self.get_local_path(pid, sub_dir)
            os.makedirs(path)

    def download_file(self, relative_path):
        source = self.get_cloud_path(relative_path)
        dest = os.path.dirname(self.get_local_path(relative_path))
        subprocess.run(['rclone', 'copy', source, dest])
        return 0 if os.path.exists(dest) else 1

    def download_required_files(self, pid):
        required = self.required_files()
        for file in required:
            self.download_file(os.path.join(pid, file))

    def read_analysis_log(self):
        relative_path = 'Logs/AnalysisLog.csv'
        if self.check_exists_cloud(relative_path):
            self.download_file(relative_path)
        log = {}
        local_path = self.get_local_path(relative_path)
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                for line in f.read():
                    pid, date = line.split(',')
                    log.update({pid: date})
        return log

    def mark_as_analyzed(self, pid):
        self.log.update({pid: dt.today()})

    def write_analysis_log(self):
        local_path = self.get_local_path('Logs/AnalysisLog.csv')
        with open(local_path, 'w') as f:
            for pid, date in self.log.items():
                f.write('{}, {}\n'.format(pid, date))

    def upload_results(self):
        for upload in self.uploads:
            source = self.get_local_path(upload)
            dest = os.path.dirname(self.get_cloud_path(upload))
            assert os.path.isfile(source)
            subprocess.run(['rclone', 'copy', source, dest])

