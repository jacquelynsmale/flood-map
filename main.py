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

import newFunctions as nf

log = logging.getLogger(__name__)


def estimate_flood_depth(out_raster, water_extent, hand_raster, estimator='nmad', water_level_sigma=3,
                         known_water_threshold=30, water_classes=[1, 2, 3, 4, 5], iterative_bounds=[0, 15]):
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
    nf.reproject_flood_mask(epsg_we, epsg_hand, filename, reprojected_flood_mask,
                            Path(water_extent).parent)

    # Save Info for reprojected TIF
    info = gdal.Info(str(reprojected_flood_mask), options=['-json'])
    epsg = nf.check_coordinate_system(info)
    gT = info['geoTransform']
    width, height = info['size']
    west, south, east, north = nf.get_wesn(info)

    # Clip HAND to the same size as the reprojected_flood_mask
    log.info(f'Clipping HAND to {width} by {height} pixels.')
    gdal.Warp(str(cwd) + '/clip_HAND.tif', hand_raster, outputBounds=[west, south, east, north], width=width,
              height=height,
              resampleAlg='lanczos', format="GTiff")  # Missing -overwrite

    # Read in HAND array
    hand_array = nf.readData(f"{cwd}/clip_HAND.tif")
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
                m = logstat(hand_clip[flood_mask_labels_clip == l], func=np.nanmean)
                s = logstat(hand_clip[flood_mask_labels_clip == l])
                water_height = m + water_level_sigma * s
            elif estimator.lower() == "iterative":
                water_height = iterative(hand_clip, flood_mask_labels_clip == l, water_levels=iterative_bounds)
            else:
                log.info("Unknown estimator selected for water height calculation.")
                raise ValueError

        flood_depth_clip = flood_depth[min0:max0, min1:max1]
        flood_depth_clip[flood_mask_labels_clip == l] = water_height - hand_clip[flood_mask_labels_clip == l]

    flood_depth[flood_depth < 0] = 0

    write_cog(str(out_raster).replace('.tif', f'_{estimator}_WaterDepth.tif'), flood_depth, transform=gT,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)
    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodMask.tif'), flood_mask, transform=gT,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)

    flood_mask[known_water_mask] = 0
    flood_depth[np.bitwise_not(flood_mask)] = 0

    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodDepth.tif'), flood_depth, transform=gT,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--water-extent',
                        help='Hyp3-Generated water extent raster file.')
    parser.add_argument('--hand-raster',
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                             'If not specified, HAND data will be extracted from a Copernicus GLO-30 DEM based HAND.')
    parser.add_argument('--water-level-sigma', type=float, default=3,
                        help='Estimate max water height for each object.')
    parser.add_argument('--known-water-threshold', type=float, default=30,
                        help='Threshold for extracting known water area in percent')
    parser.add_argument('--estimator', type=float, default='nmad',
                        help='Flood depth estimation approach (iterative, nmad, logstat, numpy). Default is nmad.')
    parser.add_argument('--iterative-bounds', type=float, default=[0, 15],
                        help='.')
    parser.add_argument('--water-classes', type=float, default=[1, 2, 3, 4, 5],
                        help='.')
    parser.add_argument('--out-raster',
                        help='File flood depth map will be saved to.')


    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    estimate_flood_depth(args.out-raster, args.water-extent, args.hand-raster, args.estimator, args.water-level-sigma,
                         args.known_water-threshold, args.water-classes, args.iterative-bounds, args.out-raster)

    info.log(f"Flood Map written to {args.out-raster}.")


# Press the green button in the gutter to run the script.git 
if __name__ == '__main__':
    main()
