#!/usr/bin/env python
import os
import glob
import argparse

import xarray as xr
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from rasterio.enums import Resampling
from dask.distributed import Client, LocalCluster


def parse_date_from_filename(tif_path):
    """
    Extracts a datetime from the TIFF filename.
    - Non-MODIS: expects basename like <anything>_YYYYMMDD.tif
    - MODIS_MCD43A4: expects a sub-string like .AYYYYDDD. (year + Julian day)
    """
    basename = os.path.basename(tif_path).replace('.tif', '')
    if basename.startswith('MCD43A4'):
        # e.g. "MCD43A4.A2021001.006.2021010123456"
        _, datestr, _, _, _ = basename.split('.')
        return pd.to_datetime(datestr, format='A%Y%j')
    else:
        date_str = basename.split('_')[-1]
        return pd.to_datetime(date_str, format='%Y%m%d')


def process_stand(stand_id, tif_paths, store_path, output_path, mask_shapefile=None, variable_name='Reflectance'):
    """
    Reads all TIFFs for one stand, reprojects to a common grid,
    optionally masks out areas from a shapefile, concatenates into an xarray.Dataset,
    writes Zarr + NetCDF.
    """
    if not tif_paths:
        print(f"[{stand_id}] No TIFFs found → skipping.")
        return

    # Open first TIFF to get reference grid
    ref_da = rxr.open_rasterio(tif_paths[0], masked=True, chunks={'x':1024,'y':1024})
    ref_crs = ref_da.rio.crs
    ref_transform = ref_da.rio.transform()

    # Load and reproject mask shapefile if provided
    mask_gdf = None
    if mask_shapefile:
        try:
            mask_gdf = gpd.read_file(mask_shapefile)
            mask_gdf = mask_gdf.to_crs(ref_crs)
        except Exception as e:
            print(f"[{stand_id}] Error loading mask shapefile {mask_shapefile}: {e}")
            mask_gdf = None

    ds_list = []
    for tif in tif_paths:
        try:
            date = parse_date_from_filename(tif)
        except Exception as e:
            print(f"[{stand_id}] Skipping {tif}: date parse error: {e}")
            continue

        try:
            da = rxr.open_rasterio(tif, masked=True, chunks={'x':1024,'y':1024})
        except Exception as e:
            print(f"[{stand_id}] Error opening {tif}: {e}")
            continue

        # Reproject to reference
        da = da.rio.reproject(ref_crs, transform=ref_transform, resampling=Resampling.nearest)

        # Mask out water bodies if mask provided
        if mask_gdf is not None:
            da = da.rio.clip(mask_gdf.geometry, mask_gdf.crs, drop=False, invert=True)

        # Drop or rename band dim
        if 'band' in da.dims and da.sizes['band']==1:
            da = da.squeeze('band', drop=True)
        else:
            da = da.rename({'band':'bands'})

        # Add date dimension
        da = da.expand_dims({'date':[date]})

        ds = da.to_dataset(name=variable_name)
        ds = ds.drop_vars(['add_offset','scale_factor','_FillValue'], errors='ignore')
        ds_list.append(ds)

    if not ds_list:
        print(f"[{stand_id}] No valid slices → skipping.")
        return

    combined = xr.concat(ds_list, dim='date')
    combined = combined.chunk({'date':1,'x':512,'y':512})

    combined.to_zarr(store_path, mode='w', consolidated=True)
    print(f"[{stand_id}] Wrote Zarr → {store_path}")

    combined.attrs['stand_id'] = stand_id
    combined.to_netcdf(output_path)
    print(f"[{stand_id}] Wrote NetCDF → {output_path}")


def build_stand_xarray_parallel(base_dir, output_dir, num_workers, sources, mask_shapefile=None, variable_name='Reflectance'):
    """
    Scans base_dir for stands, builds mapping stand_id→TIFFs,
    then submits one Dask task per stand, passing mask_shapefile.
    """
    stands = {}
    for entry in os.listdir(base_dir):
        stand_path = os.path.join(base_dir, entry)
        if not os.path.isdir(stand_path):
            continue
        tif_list = []
        subdirs = sources or os.listdir(stand_path)
        for src in subdirs:
            src_dir = os.path.join(stand_path, src)
            if not os.path.isdir(src_dir):
                continue
            tif_list.extend(sorted(glob.glob(os.path.join(src_dir,'*.tif'))))
        stands[entry] = tif_list

    os.makedirs(output_dir, exist_ok=True)

    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, processes=True, memory_limit='8GB')
    client = Client(cluster)
    print(f"Dask dashboard available at {client.dashboard_link}")

    futures = []
    for stand_id, tif_paths in stands.items():
        store_path = os.path.join(output_dir, f"{stand_id}.zarr")
        output_path = os.path.join(output_dir, f"{stand_id}.nc")
        futures.append(client.submit(process_stand, stand_id, tif_paths, store_path, output_path, mask_shapefile, variable_name, pure=False))

    client.gather(futures)
    client.close()

if __name__=='__main__':
    p = argparse.ArgumentParser(description="Build per-stand time series via xarray + Dask")
    p.add_argument('--base_dir', required=True, help='Root folder containing stand_*/<source>/*.tif')
    p.add_argument('--output_dir', required=True, help='Where to write .zarr and .nc outputs')
    p.add_argument('--num_workers', type=int, default=None, help='Number of Dask workers (defaults to CPU count)')
    p.add_argument('--sources', nargs='+', default=None, help='Subset of subfolders to include')
    p.add_argument('--mask_shapefile', default=None, help='Optional shapefile to mask out areas (e.g., water bodies)')
    p.add_argument('--variable_name', default='Reflectance', help='Name of variable in output dataset')
    args = p.parse_args()

    if args.num_workers is None:
        import multiprocessing
        args.num_workers = max(1, multiprocessing.cpu_count()-1)

    build_stand_xarray_parallel(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        sources=args.sources,
        mask_shapefile=args.mask_shapefile,
        variable_name=args.variable_name
    )
