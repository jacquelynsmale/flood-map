import logging
import matplotlib.pyplot as plt
import newFunctions as nf
import numpy as np
import os
import osgeo
import pylab as pl
import warnings
from asf_tools.composite import write_cog
from asf_tools.hand.prepare import prepare_hand_for_raster
from os import symlink
from osgeo import gdal
from pathlib import Path
from scipy import ndimage
from scipy import stats

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
water_classes = [1, 2, 3, 4, 5]  # 1 has to be a water class, 0 is no water Others are optional.
water_level_sigma = 3  # use 3*std to estimate max. water height (water level) for each object. Used for numpy, nmad,logstat
estimator = "nmad"  # iterative, numpy, nmad or logstat
iterative_bounds = [0, 15]  # only used for iterative
known_water_threshold = 30  # Threshold for extracting the known water area in percent.

water_extent = '/home/jrsmale/projects/floodMap/BangledeshFloodMapping/tifs/flooddaysBG.tif'
hand_raster = None
outfile = '/home/jrsmale/projects/floodMap/BangledeshFloodMapping/tifs/BG_FloodDepth.tif'

#############################################################
if hand_raster is None:
    hand_raster = str(water_extent).replace('.tif', '_HAND.tif')
    log.info(f'Extracting HAND data to: {hand_raster}')
    prepare_hand_for_raster(hand_raster, water_extent)

# check coordinate systems
info_we = gdal.Info(str(water_extent), options=['-json'])
info_hand = gdal.Info(str(hand_raster), options=['-json'])

epsg_we = nf.check_coordinate_system(info_we)
epsg_hand = nf.check_coordinate_system(info_hand)
log.info(f"Water extent map EPSG: {epsg_we}")
log.info(f"HAND EPSG: {epsg_hand}")

log.info('Project HAND and water extent map to same EPSG.')
reprojected_flood_mask = str(water_extent).replace('.tif', '_reprojected.tif')
nf.reproject_flood_mask(epsg_we, epsg_hand, water_extent, reprojected_flood_mask,
                        Path(water_extent).parent)

# Save Info for reprojected TIF
info = gdal.Info(str(reprojected_flood_mask), options=['-json'])
epsg = nf.check_coordinate_system(info)
gT = info['geoTransform']
width, height = info['size']
west, south, east, north = nf.get_wesn(info)

# Clip HAND to the same size as the reprojected_flood_mask
log.info(f'Clipping HAND to {width} by {height} pixels.')
gdal.Warp(str(hand_raster).replace('.tif', '_CLIP.tif'), hand_raster, outputBounds=[west, south, east, north],
          width=width,
          height=height,
          resampleAlg='lanczos', format="GTiff")  # Missing -overwrite

# Read in HAND array
hand_array = nf.readData(str(hand_raster).replace('.tif', '_CLIP.tif'))
log.info('Fetching perennial flood data.')
known_water_mask = nf.get_waterbody(info, ths=known_water_threshold)

# load change detection product from Hyp3
hyp3_map = gdal.Open(reprojected_flood_mask)
change_map = hyp3_map.ReadAsArray()

flood_mask = nf.initial_mask_generation(change_map, known_water_mask,
                                        water_classes=water_classes)

# Estimate flood depth (own function in NewFunctions.py)
flood_mask_labels, num_labels = ndimage.label(flood_mask)
object_slices = ndimage.find_objects(flood_mask_labels)
log.info(f'Detected {num_labels} water bodies...')

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
            m = nf.logstat(hand_clip[flood_mask_labels_clip == l], func=np.nanmean)
            s = nf.logstat(hand_clip[flood_mask_labels_clip == l])
            water_height = m + water_level_sigma * s
        elif estimator.lower() == "iterative":
            water_height = nf.iterative(hand_clip, flood_mask_labels_clip == l, water_levels=iterative_bounds)
        else:
            log.info("Unknown estimator selected for water height calculation.")
            raise ValueError

    flood_depth_clip = flood_depth[min0:max0, min1:max1]
    flood_depth_clip[flood_mask_labels_clip == l] = water_height - hand_clip[flood_mask_labels_clip == l]

flood_depth[flood_depth < 0] = 0

m = np.nanmean(flood_depth)
s = np.nanstd(flood_depth)
clim_min = max([m-2*s, 0])
clim_max = min([m+2*s, 5])
pl.matshow(flood_depth)
pl.colorbar()
pl.clim([clim_min, clim_max])
pl.title('Estimated Flood Depth')
pl.show()

write_cog(outfile, flood_depth, transform=gT, epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)

#Original script saves
#Flood depth as HAND_WaterDepth...
#Flood Mask as Flood_mask...

flood_mask[known_water_mask] = 0
flood_depth[np.bitwise_not(flood_mask)] = 0

#Flood Depth again as HAND_FloodDepth