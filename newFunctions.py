import numpy as np
import util
import os
import warnings
from os import symlink
from osgeo import gdal
from osgeo import osr
from osgeo.gdal_array import LoadFile
from pathlib import Path
from scipy import ndimage
from scipy import stats


def check_coordinate_system(info):
    info = info['coordinateSystem']['wkt']
    return info.split('ID')[-1].split(',')[1].replace(']', '')


def get_wesn(info):
    west, south = info['cornerCoordinates']['lowerLeft']
    east, north = info['cornerCoordinates']['upperRight']
    return west, south, east, north


def readData(filename, ndtype=np.float64):
    '''
    z=readData('/path/to/file')
    '''
    if os.path.isfile(filename):
        return LoadFile(filename).astype(ndtype);
    else:
        return gdal.Open(filename, gdal.GA_ReadOnly).readAsArray()


def reproject_flood_mask(epsg, epsg_hand, filename, reprojected_filename, tiff_dir):
    if epsg != epsg_hand:
        gdal.Warp(reprojected_filename, filename, dstSRS=epsg_hand,
                  resampleAlg='cubicspline', format="GTiff")
    else:
        print('HAND and Flood Mask have the same projection. ')
        if reprojected_filename.exists():
            reprojected_filename.unlink()
        symlink(tiff_dir / filename, reprojected_filename)


def initial_mask_generation(change_map, known_water_mask, water_classes=[0, 1, 2, 3, 4, 5]):
    for c in water_classes:  # This allows more than a single water_class to be included in flood mask
        change_map[change_map == c] = 1

    mask = change_map == 1
    flood_mask = np.bitwise_or(mask, known_water_mask)  # add known water mask... #Added 20200921
    return flood_mask


def get_waterbody(input_info, ths=30):
    sw_path = Path.cwd() / f"S_WATER"
    epsg_code = 'EPSG:' + check_coordinate_system(input_info)

    west, south = input_info['cornerCoordinates']['lowerLeft']
    east, north = input_info['cornerCoordinates']['upperRight']
    width, height = input_info['size']

    water_extent_vrt = str(sw_path / 'water_extent.vrt')  # All Perennial Flood Data
    wimage_file = str(sw_path / 'surface_water_map_clip.tif')

    print(f"Known waterbodies written to: {wimage_file}")

    gdal.Warp(wimage_file, water_extent_vrt, dstSRS=epsg_code,
              outputBounds=[west, south, east, north],
              width=width, height=height, resampleAlg='lanczos', format='GTiff')

    wmask = readData(wimage_file) > ths  # higher than 30% possibility (present water)
    return wmask


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


def logstat(data, func=np.nanstd):
    """ stat=logstat(data, func=np.nanstd)
       calculates the statistic after taking the log and returns the statistic in linear scale.
       INF values inside the data array are set to nan.
       The func has to be able to handle nan values.
    """
    ld = np.log(data)
    ld[np.isinf(ld)] = np.nan
    st = func(ld)
    return np.exp(st)


def estimate_flood_depth(hand_array, flood_mask, estimator='nmad', water_level_sigma=3, iterative_bounds=[0, 15]):
    flood_mask_labels, num_labels = ndimage.label(flood_mask)
    object_slices = ndimage.find_objects(flood_mask_labels)
    print(f'Detected {num_labels} water bodies...')

    flood_depth = np.zeros(flood_mask.shape)

    for l in range(1, num_labels):  # Skip first, largest label.
        slices = object_slices[l - 1]  # osl label=1 is in object_slices[0]
        min0 = slices[0].start
        max0 = slices[0].stop
        min1 = slices[1].start
        max1 = slices[1].stop
        flood_mask_labels_clip = flood_mask_labels[min0: max0, min1: max1]

        flood_mask_clip = flood_mask[min0: max0, min1: max1].copy()
        flood_mask_clip[flood_mask_labels_clip != l] = 0  # Maskout other flooded areas (labels)
        hand_clip = hand_array[min0: max0, min1: max1]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            if estimator.lower() == "numpy":
                m = np.nanmean(hand_clip[flood_mask_labels_clip == l])
                s = np.nanstd(hand_clip[flood_mask_labels_clip == l])
                water_height = m + water_level_sigma * s
            elif estimator.lower() == "nmad":
                m = np.nanmean(hand_clip[flood_mask_labels_clip == l])
                s = stats.median_abs_deviation(hand_clip[flood_mask_labels_clip == l], scale='normal',
                                               nan_policy='omit')
                water_height = m + water_level_sigma * s
            elif estimator.lower() == "logstat":
                m = logstat(hand_clip[flood_mask_labels_clip == l], func=np.nanmean)
                s = logstat(hand_clip[flood_mask_labels_clip == l])
                water_height = m + water_level_sigma * s
            elif estimator.lower() == "iterative":
                water_height = iterative(hand_clip, flood_mask_labels_clip == l, water_levels=iterative_bounds)
            else:
                print("Unknown estimator selected for water height calculation.")
                raise ValueError

        flood_depth_clip = flood_depth[min0:max0, min1:max1]
        flood_depth_clip[flood_mask_labels_clip == l] = water_height - hand_clip[flood_mask_labels_clip == l]

    flood_depth[flood_depth < 0] = 0
    return flood_depth
