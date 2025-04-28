#!/usr/bin/env python3
"""
map_diff.py

Generate per-year GeoTIFFs of per-pixel annual statistics from a forest-wide
NetCDF timeseries, with an optional --diff flag to subtract the forest-wide scalar
annual mean from every pixel.

Usage:
  python map_diff.py --forest forest.nc --variable NDVI --stat mean --months 6-8 [--diff] [--outdir output] [--procs N]
"""
import argparse
import os
from multiprocessing import Pool, cpu_count

import xarray as xr
import rioxarray  # enables GeoTIFF IO via rasterio integration


def parse_months(month_str):
    if '-' in month_str:
        start, end = map(int, month_str.split('-', 1))
        return list(range(start, end + 1))
    return [int(month_str)]


def compute_pixel_stats(ds, var, stat, months):
    da = ds[var].where(ds['date'].dt.month.isin(months), drop=True)
    grp = da.groupby('date.year')
    if stat == 'mean':
        return grp.mean(dim='date')
    elif stat == 'median':
        return grp.median(dim='date')
    elif stat.endswith('th'):
        p = int(stat[:-2])
        return grp.quantile(p / 100, dim='date', skipna=True)
    else:
        raise ValueError(f"Unsupported stat: {stat}")


def compute_forest_stats(ds, var, stat, months):
    da = ds[var].where(ds['date'].dt.month.isin(months), drop=True)
    grp = da.groupby('date.year')
    if stat == 'mean':
        ts = grp.mean(dim=['date', 'y', 'x'])
    elif stat == 'median':
        ts = grp.median(dim=['date', 'y', 'x'])
    elif stat.endswith('th'):
        p = int(stat[:-2])
        ts = da.groupby('date.year') \
            .quantile(p / 100, dim='date', skipna=True) \
            .mean(dim=['y', 'x'])
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return ts


def worker_init(forest_path, variable, stat, months, outdir, crs, transform, diff_flag):
    global pixel_ts, forest_stats_ts, var, st, mos, odir, _crs, _transform, do_diff
    ds = xr.open_dataset(forest_path).sortby('date')
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)
    ds = ds.rio.write_crs(crs, inplace=False)
    pixel_ts = compute_pixel_stats(ds, variable, stat, months)
    forest_stats_ts = compute_forest_stats(ds, variable, stat, months) if diff_flag else None
    var = variable
    st = stat
    mos = months
    odir = outdir
    _crs = crs
    _transform = transform
    do_diff = diff_flag


def process_year(year_val):
    year = int(year_val)
    arr = pixel_ts.sel(year=year)
    if do_diff:
        scalar = float(forest_stats_ts.sel(year=year).values)
        result = arr - scalar
    else:
        result = arr
    # write GeoTIFF
    result = result.rio.write_crs(_crs, inplace=False)
    result = result.rio.write_transform(_transform, inplace=False)
    prefix = 'diff_' if do_diff else ''
    out_fname = f"{prefix}{var}_{year}.tif"
    out_path = os.path.join(odir, out_fname)
    result.rio.to_raster(out_path)
    print(f"Written {out_path}")


def main():
    p = argparse.ArgumentParser(description="Generate per-year GeoTIFFs of pixel stats with optional diff.")
    p.add_argument('--forest', '-f', required=True, help='Forest NetCDF file (with dims date,y,x)')
    p.add_argument('--variable', '-v', required=True, help='Variable name in NetCDF (e.g. NDVI, SIF)')
    p.add_argument('--stat', '-s', choices=['mean','median','90th','95th'], default='mean',
                   help='Statistic to compute annually')
    p.add_argument('--months', '-m', default='1-12',
                   help='Month or month-range to aggregate (e.g. 6-8)')
    p.add_argument('--diff', action='store_true',
                   help='Subtract forest-wide annual mean (a scalar) from each pixel')
    p.add_argument('--outdir', '-o', default=None,
                   help='Output directory for GeoTIFFs (defaults to forest file directory)')
    p.add_argument('--procs', type=int, default=None,
                   help='Number of parallel processes (default: CPU count)')
    args = p.parse_args()

    months = parse_months(args.months)
    forest_path = args.forest
    variable = args.variable
    stat = args.stat
    outdir = args.outdir or os.path.dirname(os.path.abspath(forest_path))
    os.makedirs(outdir, exist_ok=True)

    ds0 = xr.open_dataset(forest_path).sortby('date')
    ds0 = ds0.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)
    crs = ds0.rio.crs or 'EPSG:4326'
    ds0 = ds0.rio.write_crs(crs, inplace=False)
    transform = ds0.rio.transform()

    # determine years
    full_ts = compute_pixel_stats(ds0, variable, stat, months)
    years = full_ts['year'].values

    procs = args.procs or cpu_count()
    init_args = (forest_path, variable, stat, months, outdir, crs, transform, args.diff)
    with Pool(processes=min(procs, len(years)), initializer=worker_init, initargs=init_args) as pool:
        pool.map(process_year, years)

if __name__ == '__main__':
    main()
