from cich_fig import CichFig as CF
from helpers.plotter import Plotter

pid = 'TI_social_fem_con1'
cf = CF()
cf.add_projects(pid)
cf.dh.load_projects(pid)
plotter = Plotter(cf.dh.data_objects[pid])
plotter.hmm_background()

