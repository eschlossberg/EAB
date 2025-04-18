import os
import glob
import argparse
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from multiprocessing import Pool, cpu_count
from osgeo import gdal
import numpy as np
import gc

def list_subdatasets(img_file):
    """Return GDAL subdataset URIs and descriptions from an HDF file."""
    ds = gdal.Open(img_file)
    subs = ds.GetSubDatasets()  # list of (uri, description)
    if not subs:
        raise ValueError(f"No subdatasets found in {img_file}")
    return subs


def read_hdf(img_file, raster_vars=None):
    """
    Read subdatasets from an HDF file into a stacked xarray.DataArray.
    If raster_vars is provided, only subdatasets whose description or URI contain any of those names are included.
    """
    subs = list_subdatasets(img_file)
    if raster_vars:
        subs = [(uri, desc) for uri, desc in subs
                if any(var in uri or var in desc for var in raster_vars)]
    if not subs:
        raise ValueError(f"No subdatasets matching vars {raster_vars} in {img_file}")

    rasters = []
    band_names = []
    for uri, desc in subs:
        sd_ds = gdal.Open(uri)
        arr = sd_ds.ReadAsArray()
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        gt = sd_ds.GetGeoTransform()
        x_size, y_size = sd_ds.RasterXSize, sd_ds.RasterYSize
        xs = gt[0] + np.arange(x_size) * gt[1]
        ys = gt[3] + np.arange(y_size) * gt[5]
        da = xr.DataArray(arr, dims=('y', 'x'), coords={'x': xs, 'y': ys})
        rasters.append(da)
        band_names.append(desc)

    da_all = xr.concat(rasters, dim='band')
    da_all = da_all.assign_coords(band=band_names)
    proj = gdal.Open(subs[0][0]).GetProjection()
    da_all = da_all.rio.write_crs(proj)
    return da_all


def read_netcdf(img_file, raster_vars):
    """
    Read specified variables from a NetCDF file into a stacked xarray.DataArray.
    """
    ds = xr.open_dataset(img_file, engine='netcdf4')
    missing = [v for v in raster_vars if v not in ds.variables]
    if missing:
        raise ValueError(f"Missing variables in NetCDF: {missing}")
    da = ds[raster_vars].to_array(dim="band")
    return da


def process_image(args):
    img_file, stands, stand_id_field, target_crs, raster_vars, source, output_folder, overwrite = args
    filename = os.path.basename(img_file)
    date_str, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Read imagery
    try:
        if ext in ['.hdf', '.h5']:
            da = read_hdf(img_file, raster_vars)
            x_dim, y_dim = 'x', 'y'
        elif ext in ['.nc', '.nc4', '.cdf']:
            da = read_netcdf(img_file, raster_vars)
            x_dim, y_dim = 'lon', 'lat'
        else:
            print(f"[SKIP] Unrecognized format {ext} for {filename}")
            return
    except Exception as e:
        print(f"[ERROR] Reading {filename}: {e}")
        return

    # If MODIS MCD43A4 style (14 bands), mask QA and compute NDVI
    try:
        if 'band' in da.dims and da.sizes['band'] == 14:
            qa = da.isel(band=slice(0, 7))
            refl = da.isel(band=slice(7, 14))
            # mask reflectance where QA != 0
            mask = (qa.values == 0)
            refl_vals = refl.values
            refl_masked = np.where(mask, refl_vals, np.nan)
            refl = xr.DataArray(
                refl_masked,
                dims=refl.dims,
                coords=refl.coords,
                name='reflectance'
            )
            # compute NDVI
            red = refl.isel(band=0)
            nir = refl.isel(band=1)
            ndvi = (nir - red) / (nir + red)
            da = ndvi.rename('NDVI')
    except Exception as e:
        print(f"[ERROR] NDVI computation failed for {filename}: {e}")
        return

    # Ensure CRS
    if not da.rio.crs:
        da = da.rio.write_crs(target_crs)
    stands_proj = stands.to_crs(da.rio.crs)

    # Set spatial dims
    try:
        da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    except Exception:
        pass

    # Clip & save per stand
    for _, stand in stands_proj.iterrows():
        sid = stand[stand_id_field]
        geom = [stand.geometry]
        out_dir = os.path.join(output_folder, f'stand_{sid}', source)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'{date_str}.tif')

        # handle overwrite flag
        if os.path.exists(out_file) and not overwrite:
            continue

        try:
            clipped = da.rio.clip(geom, da.rio.crs, drop=True, all_touched=True)
            # check NDVI valid range
            arr = clipped.values
            valid = np.isfinite(arr) & (arr >= -1) & (arr <= 1)
            if not np.any(valid):
                print(f"[SKIP_SAVE] No valid NDVI for stand {sid}, {filename}")
                continue
            clipped.rio.to_raster(out_file)
            action = 'OVERWRITE' if overwrite and os.path.exists(out_file) else 'SAVED'
            print(f"[{action}] {sid} {filename}")
        except Exception as e:
            print(f"[CLIP ERROR] {filename}, stand {sid}: {e}")

    # Cleanup
    try:
        da.close()
    except:
        pass
    del da
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Clip multi-band imagery (HDF/NetCDF) by stands in parallel.")
    parser.add_argument('--shapefile', required=True, help='Path to stand shapefile')
    parser.add_argument('--imagery_sources', required=True, nargs='+', help='List of imagery source subfolders')
    parser.add_argument('--raw_imagery', required=True, help='Base raw imagery directory')
    parser.add_argument('--output_folder', required=True, help='Base output directory')
    parser.add_argument('--stand_id_field', default='stand_id', help='Field name for stand ID')
    parser.add_argument('--target_crs', default='EPSG:4326', help='Target CRS for clipping')
    parser.add_argument('--raster_vars', nargs='+', default=None, help='Raster vars/bands to include; omit for all')
    parser.add_argument('--num_workers', type=int, default=cpu_count()-2, help='Number of parallel processes')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()

    stands = gpd.read_file(args.shapefile).to_crs(args.target_crs)
    tasks = []
    for source in args.imagery_sources:
        files = sorted(glob.glob(os.path.join(args.raw_imagery, source, '*.*')))
        for f in files:
            tasks.append((f, stands, args.stand_id_field, args.target_crs, args.raster_vars, source, args.output_folder, args.overwrite))

    with Pool(processes=args.num_workers) as pool:
        pool.map(process_image, tasks)

if __name__ == '__main__':
    main()
