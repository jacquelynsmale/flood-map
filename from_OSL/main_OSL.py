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

import convienence

log = logging.getLogger(__name__)

#DEFINE PARAMETERS
version="0.1.8"
water_classes = [1,2,3,4,5] # 1 has to be a water class, 0 is no water Others are optional.
pattern="*_water_mask_combined.tiff" #"filter_*_amp_Classified.tif"
show_plots=True #turn this off for debugging with IPDB
water_level_sigma=3 #use 3*std to estimate max. water height (water level) for each object. Used for numpy, nmad,logstat
estimator = "nmad" # iterative, numpy, nmad or logstat
iterative_bounds=[0,15] #only used for iterative
output_prefix='' # Output file is created in the same folder as flood extent. A prefix can be added to the filename.
known_water_threshold=30 #Threshold for extracting the known water area in percent.

tiff_dir = '/Users/jrsmale/projects/floodMap/BangledeshFloodMapping/tifs/'
tiff_path = tiff_dir + 'flooddaysBG.tif'
work_path = Path(tiff_path).parent
hand_dem = tiff_dir + 'Bangladesh_Training_DEM_hand.tif'

#Check coordinate systems
#Check coordinate systems
info = (gdal.Info(str(tiff_path), options = ['-json']))
info = info['coordinateSystem']['wkt']
epsg = info.split('ID')[-1].split(',')[1].replace(']', '')
print(f"EPSG code for Water Extent: {epsg}")

hand_dem = tiff_dir + 'Bangladesh_Training_DEM_hand.tif'
info_hand = (gdal.Info(str(hand_dem), options = ['-json']))
info_hand = info_hand['coordinateSystem']['wkt']
epsg_hand = info_hand.split('ID')[-1].split(',')[1].replace(']', '')
print(f'EPSG for HAND: {epsg_hand}')

filename = Path(tiff_path).name
filenoext = Path(tiff_path).stem #given vrt we want to force geotif output with tif extension
tiff_dir = Path(tiff_dir)
# Building the virtual raster for Change Detection product(tiff)
reprojected_flood_mask = tiff_dir/f"reproj_{filenoext}.tif"

#Reproject coordinate system
if epsg != epsg_hand:
    cmd_reproj=f"gdalwarp -overwrite -t_srs EPSG:{epsg_hand} -r cubicspline -of GTiff {tiff_dir}/{filename} {tiff_dir}/reproj_{filenoext}.tif"
    print(cmd_reproj)
    os.system(cmd_reproj)
else:
    if (tiff_dir/f'reproj_{filenoext}.tif').exists():
        (tiff_dir/f'reproj_{filenoext}.tif').unlink()
    symlink(tiff_dir/filename, tiff_dir/f'reproj_{filenoext}.tif')

# Building the virtual raster for Change Detection product(tiff)
reprojected_flood_mask = tiff_dir/f"reproj_{filenoext}.tif"
pixels, lines = convienence.get_size(str(reprojected_flood_mask))

#checking extent of the map
info = (gdal.Info(str(reprojected_flood_mask), options = ['-json']))
west, east, south, north = convienence.get_wesn(str(reprojected_flood_mask))

#Clip HAND to the same size as the reprojected_flood_mask
hand_dem_bb = convienence.bounding_box(hand_dem)
cmd_clip = f"gdalwarp -overwrite -te {west} {south} {east} {north} -ts {pixels} {lines} -r lanczos  -of GTiff {hand_dem} {tiff_dir}/clip_{filename}"
os.system(cmd_clip)

hand_array = convienence.readData(f"{tiff_dir}/clip_{filename}")

#Get known Water Mask
known_water_mask = convienence.get_waterbody(str(reprojected_flood_mask), known_water_threshold)

#load and display change detection product from Hyp3
hyp_map = gdal.Open(str(reprojected_flood_mask))
change_map = hyp_map.ReadAsArray()

#Initial mask layer generation
for c in water_classes: # This allows more than a single water_class to be included in flood mask
    change_map[change_map==c] = 1

mask = change_map == 1
flood_mask = np.bitwise_or(mask,known_water_mask) #add known water mask... #Added 20200921

# Calculate Flood Depth - Show Progress Bar
flood_mask_labels, num_labels = ndimage.label(flood_mask)
print(f'Detected {num_labels} water bodies...')
object_slices = ndimage.find_objects(flood_mask_labels)

flood_depth=np.zeros(flood_mask.shape)

print(f'Using estimator: {estimator}')
for l in range(1, num_labels):#Skip first, largest label.
    slices = object_slices[l-1] #osl label=1 is in object_slices[0]
    min0 = slices[0].start
    max0 = slices[0].stop
    min1 = slices[1].start
    max1 = slices[1].stop
    flood_mask_labels_clip = flood_mask_labels[min0: max0, min1: max1]

    flood_mask_clip = flood_mask[min0: max0, min1: max1].copy()
    flood_mask_clip[flood_mask_labels_clip != l] = 0 #Maskout other flooded areas (labels)
    hand_clip = hand_array[min0: max0, min1: max1]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        if estimator.lower() == "numpy": #BROKE
            m = np.nanmean(hand_clip[flood_mask_labels_clip == l])
            s = np.nanstd(hand_clip[flood_mask_labels_clip == l])
            water_height = m + water_level_sigma * s
        elif estimator.lower() == "nmad":
            m = np.nanmean(hand_clip[flood_mask_labels_clip == l])
            s = stats.median_abs_deviation(hand_clip[flood_mask_labels_clip == l], scale='normal', nan_policy='omit')
            water_height = m + water_level_sigma * s
        elif estimator.lower() == "logstat":
            m = convienence.logstat(hand_clip[flood_mask_labels_clip == l], func=np.nanmean)
            s = convienence.logstat(hand_clip[flood_mask_labels_clip == l])
            water_height = m + water_level_sigma * s
        elif estimator.lower() == "iterative":
            water_height = convienence.iterative(hand_clip, flood_mask_labels_clip == l, water_levels=iterative_bounds)
        else:
            print("Unknown estimator selected for water height calculation.")
            raise ValueError

    flood_depth_clip = flood_depth[min0:max0, min1:max1]
    flood_depth_clip[flood_mask_labels_clip==l] = water_height - hand_clip[flood_mask_labels_clip==l]

#remove negative depths:
flood_depth[flood_depth<0] = 0

m = np.nanmean(flood_depth)
s = np.nanstd(flood_depth)
clim_min = max([m-2*s, 0])
clim_max = min([m+2*s, 5])
pl.matshow(flood_depth)
pl.colorbar()
pl.clim([clim_min, clim_max])
pl.title('Estimated Flood Depth')
pl.show()

arrsave = estimator + filenoext
np.save(arrsave, flood_depth, allow_pickle=False)

#Saving Estimated FD to geotiff
geotiff_path = Path(work_path)/'geotiff'
print(geotiff_path)

if not geotiff_path.exists():
    geotiff_path.mkdir()

gT = convienence.get_geotransform(f"{tiff_dir}/clip_{filename}")

outfilename = str(tiff_path).split(str(tiff_dir))[1].split("/")[1]
srs_proj4 = convienence.gdal_get_projection(f"{tiff_dir}/clip_{filename}")
convienence.writeTiff(flood_depth, gT, filename = "_".join(filter(None, [output_prefix, f"{geotiff_path}/HAND_WaterDepth", estimator, version, outfilename])), srs_proj4=srs_proj4, nodata=0, options = ["TILED=YES","COMPRESS=LZW","INTERLEAVE=BAND","BIGTIFF=YES"])
convienence.writeTiff(flood_mask, gT, filename = "_".join(filter(None, [output_prefix, f"{geotiff_path}/Flood_mask", estimator, version, outfilename])), srs_proj4=srs_proj4, options = ["TILED=YES","COMPRESS=LZW","INTERLEAVE=BAND","BIGTIFF=YES"])

flood_mask[known_water_mask] = 0
flood_depth[np.bitwise_not(flood_mask)] = 0
convienence.writeTiff(flood_depth, gT, filename = "_".join(filter(None, [output_prefix, f"{geotiff_path}/HAND_FloodDepth", estimator, version, outfilename])), nodata=0, srs_proj4=srs_proj4, options = ["TILED=YES","COMPRESS=LZW","INTERLEAVE=BAND","BIGTIFF=YES"])
print('Export complete.')




