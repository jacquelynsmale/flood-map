import logging
import os
import warnings
from os import symlink
from pathlib import Path

import numpy as np
import pylab as pl
from osgeo import gdal
from scipy import ndimage
from scipy import stats

import util

log = logging.getLogger(__name__)
ß
def estimate_flood_depth(hand_dem, water_extent_tif, estimator='nmad', water_classes = [1, 2, 3, 4, 5], water_level_sigma=3, known_water_threshold=30,
                         iterative_bounds=[0,15]):
    return flood_depth


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--bucket', help='AWS bucket to upload product files to')
    parser.add_argument('--bucket-prefix', default='', help='AWS prefix (location in bucket) to add to product files')

    parser.add_argument('--hand-raster',
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                             'If not specified, HAND data will be extracted from a Copernicus GLO-30 DEM based HAND.')
    parser.add_argument('--water-level-sigma', type=float, default=3,
                        help='Estimate max water height for each object.')
    parser.add_argument('--known-water-threshold', type=float, default=30,
                        help='Threshold for extracting known water area in percent')
    parser.add_argument('--estimator', type=float, default='nmad',
                        help='Flood depth estimation approach (iterative, nmad, logstat, numpy). Default is nmad.')
    parser.add_argument('--iterative_bounds', type=float, default=[0, 15],
                        help='.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    estimate_flood_depth(hand_raster, water_extent_tif, estimator, water_level_sigma, known_water_threshold,
                         iterative_bounds)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
