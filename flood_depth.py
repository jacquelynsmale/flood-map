import logging
import os
import warnings
from os import symlink
from pathlib import Path

import numpy as np
import osgeo
import pylab as pl
from osgeo import gdal
from scipy import ndimage
from scipy import stats

import util
import newFunctions as nf
from asf_tools.hand.prepare import prepare_hand_for_raster
from asf_tools.composite import write_cog

"""Creates a flood depth map from a water extent map. 

Create a flood depth map from a single Hyp3-generated surface water extent map and
a HAND image. The HAND image must completely cover the water extent map. 

Known perennial Global Surface water data are produced under the Copernicus Programme, 
and are pulled to ensure this information is accounted for in the Flood Depth Map calculation.
This is added to the SAR-derived surface water detection maps to generate the 
final Flood Depth product. 

Flood depth maps are estimated using one of the approaches: 
*Iterative: Basin hopping optimization method to match flooded areas to flood depth 
estimates given by the HAND layer. This is the most accurate method, but also the 
most time-intensive. 
*Normalized Median Absolute Deviation (nmad): (Default) Uses a median operator to estimate
the variation to increase robustness in the presence of liars. 
*Logstat: Calculates the mean and standard deviation of HAND heights in the logarithmic 
domain to improve robustness for very non-Gaussian data distributions. 
*Numpy: Calculates statistics in a linear scale. Least robust to outliers and non-Gaussian
distributions. 

"""

log = logging.getLogger(__name__)

######################################################
# MANUALLY SET UP ARGS/PATHS
# DEFINE PARAMETERS
water_classes = [0, 1, 2, 3, 4, 5]  # 1 has to be a water class, 0 is no water Others are optional.
water_level_sigma = 3  # use 3*std to estimate max. water height (water level) for each object. Used for numpy, nmad,logstat
estimator = "nmad"  # iterative, numpy, nmad or logstat
iterative_bounds = [0, 15]  # only used for iterative
known_water_threshold = 30  # Threshold for extracting the known water area in percent.

tiff_dir = '/Users/jrsmale/projects/floodMap/BangledeshFloodMapping/tifs/'
tiff_path = tiff_dir + 'flooddays.tiff'
hand_dem = tiff_dir + 'Bangladesh_Training_DEM_hand.tif'

filename = Path(tiff_path).name
filenoext = Path(tiff_path).stem  # given vrt we want to force geotif output with tif extension
outfile = tiff_dir + 'NEWHAND_FloodDepth_' + estimator + filenoext + '.tif'

tiff_dir = Path(tiff_dir)
reprojected_flood_mask = tiff_dir / f"reproj_{filenoext}.tif"
#############################################################
# check coordinate systems
info_we = gdal.Info(str(tiff_path), options=['-json'])
info_hand = gdal.Info(str(hand_dem), options=['-json'])

epsg_we = nf.check_coordinate_system(info_we)
epsg_hand = nf.check_coordinate_system(info_hand)
print(f"Water extent map EPSG: {epsg_we}")
print(f"HAND EPSG: {epsg_hand}")

# Reproject Flood Mask
nf.reproject_tifs(epsg_we, epsg_hand, tiff_dir, filename, reprojected_flood_mask)

#Info for reprojected TIF
info = gdal.Info(str(reprojected_flood_mask), options=['-json'])
epsg = nf.check_coordinate_system(info)
gT = info['geoTransform']
width, height = info['size']
rfm_wesn = util.get_wesn(info)

# Clip HAND to the same size as the reprojected_flood_mask
print(f'Clipping HAND to be {width} by {height} pixels.')
hand_dem_bb = util.bounding_box(hand_dem)
gdal.Warp(str(tiff_dir) + '/clip_HAND.tif', hand_dem, outputBounds=rfm_wesn, width=width, height=height,
          resampleAlg='lanczos', format="GTiff")  # Missing -overwrite

hand_array = util.readData(f"{tiff_dir}/clip_HAND.tif")

# Get known Water Mask
print(f"RFM EPSG: {epsg}")
print('Fetching perennial flood data.')
known_water_mask = util.get_waterbody(str(reprojected_flood_mask), known_water_threshold)

# load and display change detection product from Hyp3
hyp3_map = gdal.Open(str(reprojected_flood_mask))
change_map = hyp3_map.ReadAsArray()

flood_mask = nf.initial_mask_generation(change_map, known_water_mask,
                                        water_classes=water_classes)

flood_depth = nf.estimate_flood_depth(hand_array, flood_mask, estimator=estimator,
                                      water_level_sigma=water_level_sigma,
                                      iterative_bounds=iterative_bounds)

util.writeTiff(flood_depth, gT, outfile, srs_proj4=epsg_we, nodata=0, options = ["TILED=YES","COMPRESS=LZW","INTERLEAVE=BAND","BIGTIFF=YES"])
