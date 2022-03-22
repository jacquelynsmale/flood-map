import os
import urllib
import warnings
from pathlib import Path

import astropy
import numpy as np
import pyproj
import rasterio
from osgeo import gdal
from osgeo import osr
from scipy import optimize
from osgeo.gdal_array import LoadFile

import mygis as gis


def bounding_box_inside_bounding_box(small, big):
    s0 = np.array([p[0] for p in small])
    s1 = np.array([p[1] for p in small])
    b0 = np.array([p[0] for p in big])
    b1 = np.array([p[1] for p in big])
    inside = True
    if s0.min() < b0.min():
        inside = False
    if s0.max() > b0.max():
        inside = False
    if s1.min() < b1.min():
        inside = False
    if s1.max() > b1.max():
        inside = False
    return inside


def get_geotransform(filename):
    '''
    [top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution]=getGeoTransform('/path/to/file')
    '''
    # http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    ds = gdal.Open(filename)
    return ds.GetGeoTransform()

def writeTiff(ary, coord, filename='kgiAlos.tif', rescale=None, format=gdal.GDT_Float64, lon=None, lat=None,
              nodata=None, grid=False, cog=False, srs_proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
              options=[], gcps=None):
    '''writeTiff(ary, geoTransform, filename='kgiAlos.tif', rescale=None, format=gdal.GDT_Float64 ,lon=None, lat=None):
    ary: 2D array.
    geoTransform: [top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution]
    rescale: [min max]: If given rescale ary values between min and max.

    If lon lat is specified set coord to None

    '''

    if coord is None and gcps is None:
        import scipy
        import scipy.linalg
        s = [sk // 10 for sk in ary.shape]
        ary10 = ary[::s[0], ::s[1]];
        lon10 = lon[::s[0], ::s[1]];
        lat10 = lat[::s[0], ::s[1]];
        A = np.ones([np.multiply(*ary10.shape), 3])
        line, pixel = np.meshgrid(np.r_[0:ary.shape[0]:s[0]], np.r_[0:ary.shape[1]:s[1]])
        A[:, 1] = pixel.ravel()
        A[:, 2] = line.ravel()
        xlon = np.dot(scipy.linalg.pinv(A), lon10.ravel())
        xlat = np.dot(scipy.linalg.pinv(A), lat10.ravel())
        coord = [xlon[0], xlon[2], xlon[1], xlat[0], xlat[2], xlat[1]];
        print(coord)

        if grid:
            import scipy.interpolate
            LON, LAT = np.meshgrid(np.r_[lon.min():lon.max():abs(coord[1])], np.r_[lat.max():lat.min():-abs(coord[5])])
            ary = scipy.interpolate.griddata(np.array([lon.ravel(), lat.ravel()]).T, ary.ravel(), (LON, LAT),
                                             method='cubic');
            coord = [LON[0, 0], abs(coord[1]), 0, LAT[0, 0], 0, -abs(coord[5])];
            print(coord)

    if rescale:
        import basic
        ary = basic.rescale(ary, rescale);

    if ary.ndim == 2:
        Ny, Nx = ary.shape
        Nb = 1;
    elif ary.ndim == 3:
        Ny, Nx, Nb = ary.shape  # Nb: number of bands. #osgeo.gdal expects, (band, row, col), so this is a deviation from that.
    else:
        print("Input array has to be 2D or 3D.")
        return None
    driver = gdal.GetDriverByName("GTiff")
    if cog:
        options = ["TILED=YES", "COMPRESS=LZW", "INTERLEAVE=BAND", "BIGTIFF=YES"]
    ds = driver.Create(filename, Nx, Ny, Nb, gdal.GDT_Float64, options)

    srs = osr.SpatialReference()
    srs.ImportFromProj4(srs_proj4)
    ds.SetProjection(srs.ExportToWkt());
    if gcps is None:
        ds.SetGeoTransform(coord)
    else:
        if type(gcps[0]) == gdal.GCP:
            ds.SetGCPs(gcps, srs.ExportToWkt())
        elif type(gcps[0]) == np.int and len(gcps) == 2 and lat is not None:
            gcp_list = create_gcp_list(lon, lat, np.zeros(lat.shape), gcp_count=[gcps[0], gcps[1]])
            ds.SetGCPs(gcp_list, srs.ExportToWkt())
        else:
            print('unsupported type of GCPs. Skipping.')
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(nodata);
    if Nb == 1:
        ds.GetRasterBand(1).WriteArray(ary)
    else:
        for b in range(Nb):
            ds.GetRasterBand(b + 1).WriteArray(ary[:, :, b])
    if cog:
        ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128, 256])

    ds = None
    print("File written to: " + filename);


def readData(filename, ndtype=np.float64):
    '''
    z=readData('/path/to/file')
    '''
    if os.path.isfile(filename):
        return LoadFile(filename).astype(ndtype);
    else:
        return gdal.Open(filename, gdal.GA_ReadOnly).readAsArray()


def build_vrt(filename, input_file_list):
    vrt_options = gdal.BuildVRTOptions(resampleAlg='near', resolution='highest', separate=False,
                                       targetAlignedPixels=True)
    gdal.BuildVRT(filename, input_file_list, options=vrt_options)


#def get_tiff_paths(paths):
 #   tiff_paths = !ls $paths | sort - t_ - k5, 5
  #  return tiff_paths


def gdal_get_projection(filename, out_format='proj4'):
    """
    epsg_string=get_epsg(filename, out_format='proj4')
    """
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        srs = gdal.osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjectionRef())
    except:  # I am not sure if this is working for datasets without a layer. The first try block should work mostly.
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        ly = ds.GetLayer()
        if ly is None:
            print(f"Can not read projection from file:{filename}")
            return None
        else:
            srs = ly.GetSpatialRef()
    if out_format.lower() == 'proj4':
        return srs.ExportToProj4()
    elif out_format.lower() == 'wkt':
        return srs.ExportToWkt()
    elif out_format.lower() == 'epsg':
        crs = pyproj.crs.CRS.from_proj4(srs.ExportToProj4())
        return crs.to_epsg()


def get_size(filename):
    """(width, height) = get_size(filename)
    """
    ds = gdal.Open(filename)
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None
    return (width, height)


def get_proj4(filename):
    f = rasterio.open(filename)
    return pyproj.Proj(f.crs, preserve_units=True)  # used in pysheds


def clip_gT(gT, xmin, xmax, ymin, ymax, method='image'):
    '''calculate new geotransform for a clipped raster either using pixels or projected coordinates.
    clipped_gT=clip_gT(gT, xmin, xmax, ymin, ymax, method='image')
    method: 'image' | 'coord'
    '''
    if method == 'image':
        y, x = xy2coord(ymin, xmin, gT);  # top left, reference, coordinate
    if method == 'coord':
        # find nearest pixel
        yi, xi = coord2xy(ymin, xmin, gT)
        # get pixel coordinate
        y, x = xy2coord(yi, xi, gT)
    gTc = list(gT)
    gTc[0] = y
    gTc[3] = x
    return tuple(gTc)


def xy2coord(x, y, gT):
    '''
    lon,lat=xy2coord(x,y,geoTransform)
    projects pixel index to position based on geotransform.
    '''
    coord_x = gT[0] + x * gT[1] + y * gT[2]
    coord_y = gT[3] + x * gT[4] + y * gT[5]
    return coord_x, coord_y


def coord2xy(x, y, gT):
    '''
    x,y = coord2xy(lon, lat, geoTransform)
    calculates pixel index closest to the lon, lat.
    '''
    # ref: https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal/221430
    xOrigin = gT[0]
    yOrigin = gT[3]
    pixelWidth = gT[1]
    pixelHeight = -gT[5]

    col = np.array((x - xOrigin) / pixelWidth).astype(int)
    row = np.array((yOrigin - y) / pixelHeight).astype(int)

    return row, col


def fitSurface(x, y, z, X, Y):
    p0 = [0, 0.1, 0.1]
    fitfunc = lambda p, x, y: p[0] + p[1] * x + p[2] * y
    errfunc = lambda p, x, y, z: abs(fitfunc(p, x, y) - z)
    planefit, success = optimize.leastsq(errfunc, p0, args=(x, y, z))
    return fitfunc(planefit, X, Y)


def nonan(A, rows=False):
    if rows:
        return A[np.isnan(A).sum(1) == 0];
    else:
        return A[~np.isnan(A)];


def get_wesn(filename, t_srs=None):
    bb = bounding_box(filename, t_srs=t_srs)
    w = np.inf
    e = -np.inf
    n = -np.inf
    s = np.inf
    for p in bb:
        if p[0] < w:
            w = p[0]
        if p[0] > e:
            e = p[0]
        if p[1] < s:
            s = p[1]
        if p[1] > n:
            n = p[1]
    return [w, e, s, n]


def bounding_box(filename, t_srs=None):
    """
    ((lon1,lat1), (lon2,lat2), (lon3,lat3), (lon4,lat4))=bounding_box('/path/to/file', t_srs=None) #returns x,y in native coordinate system
    ((lon1,lat1), (lon2,lat2), (lon3,lat3), (lon4,lat4))=bounding_box('/path/to/file', t_srs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    """
    gT = get_geotransform(filename)
    width, height = get_size(filename)
    pts = (xy2coord(0, 0, gT), xy2coord(width, 0, gT), xy2coord(width, height, gT), xy2coord(0, height, gT))
    if t_srs is None:
        return pts
    else:
        pts_tsrs = []
        s_srs = gdal_get_projection(filename, out_format='proj4')
        for p in pts:
            pts_tsrs.append(transform_point(p[0], p[1], 0, s_srs=s_srs, t_srs=t_srs))
    return tuple(pts_tsrs)


def transform_point(x, y, z, s_srs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                    t_srs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'):
    '''get
    transform_point(x,y,z,s_srs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', t_srs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    Known Bugs: gdal transform may fail if a proj4 string can not be found for the EPSG or WKT formats.
    '''
    srs_cs = osr.SpatialReference()
    if "EPSG" == s_srs[0:4]:
        srs_cs.ImportFromEPSG(int(s_srs.split(':')[1]));
    elif "GEOCCS" == s_srs[0:6]:
        srs_cs.ImportFromWkt(s_srs);
    else:
        srs_cs.ImportFromProj4(s_srs);

    trs_cs = osr.SpatialReference()
    if "EPSG" == t_srs[0:4]:
        trs_cs.ImportFromEPSG(int(t_srs.split(':')[1]));
    elif "GEOCCS" == t_srs[0:6]:
        trs_cs.ImportFromWkt(t_srs);
    else:
        trs_cs.ImportFromProj4(t_srs);
    if int(gdal.VersionInfo()) > 2999999:  # 3010300
        # https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order
        # https://github.com/OSGeo/gdal/blob/master/gdal/MIGRATION_GUIDE.TXT
        srs_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        trs_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(srs_cs, trs_cs)

    if numel(x) > 1:
        return [transformPoint(x[k], y[k], z[k]) for k in range(numel(x))]
    else:
        try:
            return transform.TransformPoint((x, y, z));
        except:
            return transform.TransformPoint(x, y, z)


def get_waterbody(filename, ths):
    corners = bounding_box(filename)

    epsg = gdal_get_projection(filename, out_format='epsg')
    if epsg == "4326":
        corners = bounding_box(filename)
    else:
        srs = gdal_get_projection(filename, out_format='proj4')
        corners = bounding_box(filename, t_srs="EPSG:4326")
    west = corners[0][0]
    east = corners[1][0]
    south = corners[2][1]
    north = corners[0][1]

    # S_WATER directory is being created above working directory (i.e. Bangladesh, El_Salvador, etc.) to avoid clutter.
    # Modify here with "work_path" if you want S_WATER directory generated within your working directory.
    cwd = Path.cwd()
    sw_path = cwd / f"S_WATER"

    if not sw_path.exists():
        sw_path.mkdir()

    lon = np.floor(west / 10)
    lon = int(abs(lon * 10))
    lat = np.ceil(north / 10)
    lat = int(abs(lat * 10))

    if (west < 0 and north < 0):
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}W_{lat}Sv1_1_2019.tif",
            f"{cwd}/S_WATER/surface_water_{lon}W_{lat}S.tif")
        if (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon - 10}W_{lat}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon - 10}W_{lat}S.tif")
        if (np.floor(north / 10) != np.floor(south / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}W_{lat + 10}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon}W_{lat + 10}S.tif")
        if (np.floor(north / 10) != np.floor(south / 10)) and (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon - 10}W_{lat + 10}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon - 10}W_{lat + 10}S.tif")
        print(f"lon: {lon}-{lon - 10}W, lat: {lat}-{lat + 10}S ")

    elif (west < 0 and north >= 0):
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}W_{lat}Nv1_1_2019.tif",
            f"{cwd}/S_WATER/surface_water_{lon}W_{lat}N.tif")
        if (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon - 10}W_{lat}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon - 10}W_{lat}N.tif")
        if (np.floor(north / 10) != np.floor(south / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}W_{lat - 10}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon}W_{lat - 10}N.tif")
        if (np.floor(north / 10) != np.floor(south / 10)) and (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon - 10}W_{lat - 10}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon - 10}W_{lat - 10}N.tif")
        print(f"lon: {lon}-{lon - 10}W, lat: {lat}-{lat - 10}N ")


    elif (west >= 0 and north < 0):
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}E_{lat}Sv1_1_2019.tif",
            f"{cwd}/S_WATER/surface_water_{lon}E_{lat}S.tif")
        if (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon + 10}E_{lat}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon + 10}E_{lat}S.tif")
        if (np.floor(north / 10) != np.floor(south / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}E_{lat + 10}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon}E_{lat + 10}S.tif")
        if (np.floor(north / 10) != np.floor(south / 10)) and (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon + 10}E_{lat + 10}Sv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon + 10}E_{lat + 10}S.tif")
        print(f"lon: {lon}-{lon + 10}E, lat: {lat}-{lat + 10}S ")

    else:
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}E_{lat}Nv1_1_2019.tif",
            f"{cwd}/S_WATER/surface_water_{lon}E_{lat}N.tif")
        if (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon + 10}E_{lat}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon + 10}E_{lat}N.tif")
        if (np.floor(north / 10) != np.floor(south / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon}E_{lat - 10}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon}E_{lat - 10}N.tif")
        if (np.floor(north / 10) != np.floor(south / 10)) and (np.floor(west / 10) != np.floor(east / 10)):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/global-surface-water/downloads2019v2/occurrence/occurrence_{lon + 10}E_{lat - 10}Nv1_1_2019.tif",
                f"{cwd}/S_WATER/surface_water_{lon + 10}E_{lat - 10}N.tif")
        print(f"lon: {lon}-{lon + 10}E, lat: {lat}-{lat - 10}N ")

    # Building the virtual raster for Change Detection product(tiff)
    product_wpath = Path(cwd) / f"S_WATER/surface_water*.tif"

    # wildcard_path is not being used for now
    # wildcard_path = f"{cwd}/change_VV_20170818T122205_20170830T122203.tif"
    print(product_wpath)

    cmd_swm = f'gdalbuildvrt {product_wpath.parent}/surface_water_map.vrt $product_wpath'
    os.system(cmd_swm)
    #get_ipython().system(f'gdalbuildvrt {product_wpath.parent}/surface_water_map.vrt $product_wpath')

    # Clipping/Resampling Surface Water Map for AOI
    dim = get_size(filename)
    if epsg == "4326":
        cmd_resamp = f"gdalwarp -overwrite -te {west} {south} {east} {north} -ts {dim[0]} {dim[1]} -r lanczos {product_wpath.parent}/surface_water_map.vrt {product_wpath.parent}/surface_water_map_clip.tif"
    else:
        corners = bounding_box(filename)  # we now need corners in the non EPSG:4326 format.
        west = corners[0][0]
        east = corners[1][0]
        south = corners[2][1]
        north = corners[0][1]
        cmd_resamp = f"gdalwarp -overwrite -t_srs '{srs}' -te {west} {south} {east} {north} -ts {dim[0]} {dim[1]} -r nearest {product_wpath.parent}/surface_water_map.vrt {product_wpath.parent}/surface_water_map_clip.tif"
    print(cmd_resamp)
    os.system(cmd_resamp)

    # load resampled water map
    wimage_file = f"{product_wpath.parent}/surface_water_map_clip.tif"
    water_map = gdal.Open(wimage_file)

    swater_map = gis.readData(wimage_file)
    wmask = swater_map > ths  # higher than 30% possibility (present water)

    return wmask


def numel(x):
    if isinstance(x, np.int):
        return 1
    elif isinstance(x, np.double):
        return 1
    elif isinstance(x, np.float):
        return 1
    elif isinstance(x, str):
        return 1
    elif isinstance(x, list) or isinstance(x, tuple):
        return len(x)
    elif isinstance(x, np.ndarray):
        return x.size
    else:
        print('Unknown type {}.'.format(type(x)))
        return None


def yesno(yes_no_question="[y/n]"):
    while True:
        # raw_input returns the empty string for "enter"
        yes = {'yes', 'y', 'ye'}
        no = {'no', 'n'}

        choice = input(yes_no_question + "[y/n]").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with 'yes' or 'no'")


def fill_nan(arr):
    """
    filled_arr=fill_nan(arr)
    Fills Not-a-number values in arr using astropy.
    """
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)  # kernel x_size=8*stddev
    arr_type = arr.dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while np.any(np.isnan(arr)):
            arr = astropy.convolution.interpolate_replace_nans(arr.astype(float), kernel,
                                                               convolve=astropy.convolution.convolve)
    return arr.astype(arr_type)


def logstat(data, func=np.nanstd):
    """ stat=logstat(data, func=np.nanstd)
       calculates the statistic after taking the log and returns the statistic in linear scale.
       INF values inside the data array are set to nan.
       The func has to be able to handle nan values.
    """
    ld = np.log(data)
    ld[np.isinf(ld)] = np.nan  # requires func to handle nan-data.
    st = func(ld)
    return np.exp(st)


def iterative(hand, extent, water_levels=range(15)):
    # accuracy=np.zeros(len(water_levels))
    # for k,w in enumerate(water_levels):
    #    iterative_flood_extent=hand<w
    #    TP=np.nansum(np.logical_and(iterative_flood_extent==1, extent==1)) #true positive
    #    TN=np.nansum(np.logical_and(iterative_flood_extent==0, extent==0)) # True negative
    #    FP=np.nansum(np.logical_and(iterative_flood_extent==1, extent==0)) # False positive
    #    FN=np.nansum(np.logical_and(iterative_flood_extent==0, extent==1)) # False negative
    #    #accuracy[k]=(TP+TN)/(TP+TN+FP+FN) #accuracy
    #    accuracy[k]=TP/(TP+FP+FN) #threat score
    # best_water_level=water_levels[np.argmax(accuracy)]

    def _goal_ts(w):
        iterative_flood_extent = hand < w  # w=water level
        TP = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 1))  # true positive
        TN = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 0))  # True negative
        FP = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 0))  # False positive
        FN = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 1))  # False negative
        return 1 - TP / (TP + FP + FN)  # threat score #we will minimize goal func, hence 1-threat_score.

    # bounds=(min(water_levels), max(water_levels))
    # opt_res=optimize.minimize(_goal_ts, max(bounds),method='TNC',bounds=[bounds],options={'xtol':0.1, 'scale':1})

    class MyBounds(object):
        def __init__(self, xmax=[max(water_levels)], xmin=[min(water_levels)]):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    mybounds = MyBounds()
    x0 = [np.mean(water_levels)]
    opt_res = optimize.basinhopping(_goal_ts, x0, niter=10000, niter_success=100, accept_test=mybounds)
    if opt_res.message[0] == 'success condition satisfied' or opt_res.message[
        0] == 'requested number of basinhopping iterations completed successfully':
        best_water_level = opt_res.x[0]
    else:
        best_water_level = np.inf  # set to inf to mark unstable solution.
    return best_water_level



