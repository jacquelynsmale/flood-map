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




######################################################
# MESSILY SET UP INPUTS (WILL USE ARGPARSE)
# DEFINE PARAMETERS
water_classes = [1, 2, 3, 4, 5]  # 1 has to be a water class, 0 is no water Others are optional.
pattern = "*_water_mask_combined.tiff"  # "filter_*_amp_Classified.tif"
show_plots = True  # turn this off for debugging with IPDB
water_level_sigma = 3  # use 3*std to estimate max. water height (water level) for each object. Used for numpy, nmad,logstat
estimator = "nmad"  # iterative, numpy, nmad or logstat
iterative_bounds = [0, 15]  # only used for iterative
output_prefix = ''  # Output file is created in the same folder as flood extent. A prefix can be added to the filename.
known_water_threshold = 30  # Threshold for extracting the known water area in percent.

tiff_dir = '/Users/jrsmale/projects/floodMap/BangledeshFloodMapping/tifs/'
tiff_path = tiff_dir + 'flooddaysBG.tif'
hand_dem = tiff_dir + 'Bangladesh_Training_DEM_hand.tif'

filename = Path(tiff_path).name
filenoext = Path(tiff_path).stem  # given vrt we want to force geotif output with tif extension
tiff_dir = Path(tiff_dir)
reprojected_flood_mask = tiff_dir / f"reproj_{filenoext}.tif"
#############################################################
# check coordinate systems
epsg_we = nf.check_coordinate_system(tiff_path)
epsg_hand = nf.check_coordinate_system(hand_dem)

# Reproject
nf.reproject_tifs(epsg_we, epsg_hand, tiff_dir, filename, reprojected_flood_mask)

# Get Size of reprojected flood mask
pixels, lines = util.get_size(str(reprojected_flood_mask))

# checking extent of the map
# info = (gdal.Info(str(reprojected_flood_mask), options = ['-json']))
rfm_wesn = util.get_wesn(str(reprojected_flood_mask))

# Clip HAND to the same size as the reprojected_flood_mask
hand_dem_bb = util.bounding_box(hand_dem)
gdal.Warp(str(tiff_dir) + '/clip_' + filename, hand_dem, outputBounds=rfm_wesn, width=pixels, height=lines,
          resampleAlg='lanczos', format="GTiff")  # Missing -overwrite

hand_array = util.readData(f"{tiff_dir}/clip_{filename}")

# Get known Water Mask
known_water_mask = util.get_waterbody(str(reprojected_flood_mask), known_water_threshold)

# load and display change detection product from Hyp3
hyp3_map = gdal.Open(str(reprojected_flood_mask))
change_map = hyp3_map.ReadAsArray()

flood_mask = nf.initial_mask_generation(change_map, known_water_mask, water_classes=water_classes)

flood_depth = nf.estimate_flood_depth(hand_array, flood_mask, estimator=estimator,
                                      water_level_sigma=water_level_sigma,
                                      iterative_bounds=iterative_bounds)
