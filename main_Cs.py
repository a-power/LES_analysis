import numpy as np
import numpy.ma as ma
import dynamic as dy
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from importlib import reload
from scipy.stats import gaussian_kde

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/'
path20f = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn_p/'
file20 = "BOMEX_m0020_g0800_all_14400_filter_"

data20_2D = Dataset(file20f+file20+str('ga00.nc'), mode='r')
data20_4D = Dataset(file20f+file20+str('ga01.nc'), mode='r')

