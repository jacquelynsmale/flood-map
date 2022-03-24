import glob
import numpy as np
import os
import pyproj
import urllib
from osgeo import gdal
from osgeo import osr
from osgeo.gdal_array import LoadFile
from pathlib import Path
from scipy import optimize


def readData(filename, ndtype=np.float64):
    '''
    z=readData('/path/to/file')
    '''
    if os.path.isfile(filename):
        return LoadFile(filename).astype(ndtype);
    else:
        return gdal.Open(filename, gdal.GA_ReadOnly).readAsArray()


def get_wesn(info):
    west, south = info['cornerCoordinates']['lowerLeft']
    east, north = info['cornerCoordinates']['upperRight']
    return west, south, east, north


def iterative(hand, extent, water_levels=range(15)):
    def _goal_ts(w):
        iterative_flood_extent = hand < w  # w=water level
        tp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 1))  # true positive
        fp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 0))  # False positive
        fn = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 1))  # False negative
        return 1 - tp / (tp + fp + fn)  # threat score #we will minimize goal func, hence 1-threat_score.

    class MyBounds(object):
        def __init__(self, xmax=[max(water_levels)], xmin=[min(water_levels)]):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    bounds = MyBounds()
    x0 = [np.mean(water_levels)]
    opt_res = optimize.basinhopping(_goal_ts, x0, niter=10000, niter_success=100, accept_test=bounds)
    if opt_res.message[0] == 'success condition satisfied' or opt_res.message[
        0] == 'requested number of basinhopping iterations completed successfully':
        best_water_level = opt_res.x[0]
    else:
        best_water_level = np.inf  # unstable solution.
    return best_water_level
