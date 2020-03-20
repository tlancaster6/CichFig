from cich_fig import CichFig as CF
from helpers.plotter import Plotter

cf = CF()
cf.add_projects('CV_fem_con1')
cf.plot('plot_all_hmm')
# cf.dh.load_projects(pid)
# plotter = Plotter(cf.dh.data_objects[pid])
# plotter.three_fish_hmm_background()

