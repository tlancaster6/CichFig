from cich_fig import CichFig as CF
from helpers.plotter import Plotter

# pid = 'CV10_3'
# cf = CF()
# cf.add_projects(pid)
# cf.plot('plot_all_depth_and_kde')

pid = 'MC6_5'
cf = CF()
cf.add_projects(pid)
cf.plot('hmm_progressions')
