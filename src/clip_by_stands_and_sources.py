import os
import glob
import argparse
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from multiprocessing import Pool, cpu_count
import subprocess

def get_subdataset_names(img_file):
    result = subprocess.run(['gdalinfo', img_file], capture_output=True, text=True)
    lines = result.stdout.splitlines()
    subdatasets = []
    for line in lines:
        if 'SUBDATASET_' in line and '_NAME=' in line:
            name = line.split('=')[1].strip()
            subdatasets.append(name)
    return subdatasets

def process_single_image(args_tuple):
    img_file, stands, stand_id_field, target_crs, raster_vars, source, output_folder = args_tuple
    filename = os.path.basename(img_file)
    date_str, ext = os.path.splitext(filename)
    img_file_abs = os.path.abspath(img_file).replace("\\", "/")

    # Try to determine format by extension
    is_hdf = ext.lower() in ['.hdf', '.h5']
    is_netcdf = ext.lower() in ['.nc', '.nc4', '.cdf']

    # ========================
    # HANDLE HDF FILES
    # ========================
    if is_hdf:
        subdatasets = get_subdataset_names(img_file_abs)
        if not subdatasets:
            print(f"[ERROR] No subdatasets found in {filename}. Skipping.")
            return

        selected_subdatasets = []
        for var in raster_vars:
            matches = [s for s in subdatasets if var in s]
            if matches:
                selected_subdatasets.extend(matches)
            else:
                print(f"[WARN] Variable '{var}' not found in {filename}. Skipping variable.")

        if not selected_subdatasets:
            print(f"[SKIP] No matching variables found in {filename}.")
            return

        raster_bands = []
        for subdataset in selected_subdatasets:
            subdataset_fixed = subdataset.replace("\\", "/")
            try:
                da = rxr.open_rasterio(subdataset_fixed, masked=True)
                raster_bands.append(da)
            except Exception as e:
                print(f"[ERROR] Opening subdataset {subdataset_fixed} failed: {e}")
                return

        try:
            raster = xr.concat(raster_bands, dim='band')
        except Exception as e:
            print(f"[ERROR] Failed to stack HDF bands: {e}")
            return

    # ========================
    # HANDLE NETCDF FILES
    # ========================
    elif is_netcdf:
        try:
            ds = xr.open_dataset(img_file, engine='netcdf4')
        except Exception as e:
            print(f"[ERROR] Could not open {filename}: {e}")
            return

        missing_vars = [var for var in raster_vars if var not in ds.variables]
        if missing_vars:
            print(f"[SKIP] Variables {missing_vars} missing in {filename}.")
            return

        try:
            raster = ds[raster_vars].to_array(dim='band')
        except Exception as e:
            print(f"[ERROR] Failed to stack NetCDF bands: {e}")
            return

    else:
        print(f"[SKIP] Unrecognized file format for {filename}.")
        return

    # ========================
    # CLIP RASTER TO STANDS
    # ========================
    try:
        if not raster.rio.crs:
            raster = raster.rio.write_crs(target_crs, inplace=True)
        stands_proj = stands.to_crs(raster.rio.crs)
    except Exception as e:
        print(f"[ERROR] CRS handling failed: {e}")
        return

    for idx, stand in stands_proj.iterrows():
        stand_id = stand[stand_id_field]
        geom = [stand.geometry]

        stand_source_dir = os.path.join(output_folder, f'stand_{stand_id}', source)
        os.makedirs(stand_source_dir, exist_ok=True)
        output_path = os.path.join(stand_source_dir, f'{date_str}.tif')

        if os.path.isfile(output_path):
            continue

        try:
            raster.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)  # Just in case
        except Exception:
            pass  # No harm if already set

        try:
            clipped = raster.rio.clip(geom, raster.rio.crs, drop=True, invert=False, all_touched=True)
        except Exception as e:
            print(f"[CLIP ERROR] {filename}, stand {stand_id}: {e}")
            continue

        try:
            clipped.rio.to_raster(output_path)
            print(f"[SAVED] {stand_id}, Source: {source}, Date: {date_str}")
        except Exception as e:
            print(f"[SAVE ERROR] {stand_id} ({source}): {e}")

def clip_to_stands(shapefile_path, imagery_sources, raw_imagery_folder, output_folder, stand_id_field, target_crs, raster_vars, num_workers):
    stands = gpd.read_file(shapefile_path).to_crs(target_crs)

    for source in imagery_sources:
        imagery_path = os.path.join(raw_imagery_folder, source)
        imagery_files = sorted(glob.glob(os.path.join(imagery_path, '*.*')))
        print(f"\n[PROCESSING SOURCE] {source} with {len(imagery_files)} files.")

        args_list = [
            (img_file, stands, stand_id_field, target_crs, raster_vars, source, output_folder)
            for img_file in imagery_files
        ]

        with Pool(processes=num_workers) as pool:
            pool.map(process_single_image, args_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel clip multi-band NetCDF/HDF imagery by stands.")
    parser.add_argument('--shapefile', required=True, help='Path to shapefile')
    parser.add_argument('--imagery_sources', required=True, nargs='+', help='Imagery sources')
    parser.add_argument('--raw_imagery_folder', required=True, help='Raw imagery directory')
    parser.add_argument('--output_folder', required=True, help='Processed imagery directory')
    parser.add_argument('--stand_id_field', default='stand_id', help='Stand ID field name')
    parser.add_argument('--target_crs', default='EPSG:4326', help='CRS for data')
    parser.add_argument('--raster_vars', required=True, nargs='+', help='Raster variable names (bands)')
    parser.add_argument('--num_workers', type=int, default=cpu_count()-1, help='Number of parallel workers')

    args = parser.parse_args()

    clip_to_stands(
        shapefile_path=args.shapefile,
        imagery_sources=args.imagery_sources,
        raw_imagery_folder=args.raw_imagery_folder,
        output_folder=args.output_folder,
        stand_id_field=args.stand_id_field,
        target_crs=args.target_crs,
        raster_vars=args.raster_vars,
        num_workers=args.num_workers
    )
