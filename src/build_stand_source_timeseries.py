import xarray as xr
import rioxarray as rxr
import pandas as pd
import glob
import os
import argparse
from multiprocessing import Pool, cpu_count

def process_single_stand(args):
    stand, output_dir, sources = args
    stand_id = os.path.basename(stand)
    if not sources:
        sources = os.listdir(stand)

    dataarrays = []

    for source in sources:
        source_dir = os.path.join(stand, source)
        tifs = sorted(glob.glob(os.path.join(source_dir, '*.tif')))

        if not tifs:
            print(f"No imagery for {stand_id}, source {source}. Skipping.")
            continue
        for tif in tifs:
            basename = os.path.basename(tif).replace('.tif', '')
            try:
                if source != 'MODIS_MCD43A4':
                    source_name, date_part = basename.split('_')
                    date_str = pd.to_datetime(date_part, format='%Y%m%d')
                else:
                    _, date_part, _, _, _ = basename.split('.')
                    date_str = pd.to_datetime(date_part, format='A%Y%j')
            except Exception as e:
                print(f"[{stand_id}] Filename format error ({basename}): {e}")
                continue

            try:
                da = rxr.open_rasterio(tif, masked=True)
            except Exception as e:
                print(f"[{stand_id}] Error reading {tif}: {e}")
                continue

            if da.sizes['band'] == 1:
                da = da.squeeze('band')
            else:
                da = da.rename({'band': 'bands'})

            da = da.expand_dims({'date': [date_str], 'source': [source]})

            dataarrays.append(da)

    if not dataarrays:
        print(f"No valid imagery found for {stand_id}.")
        return
    combined = xr.concat(dataarrays, dim='date')
    dataset = combined.to_dataset(name='Reflectance')
    dataset.set_coords(['date', 'source'])

    dataset.attrs['stand_id'] = stand_id

    output_path = os.path.join(output_dir, f'{stand_id}_timeseries.nc')
    try:
        dataset.to_netcdf(output_path)
        print(f"[{stand_id}] Saved combined dataset.")
    except Exception as e:
        print(f"[{stand_id}] Error saving dataset: {e}")

def build_stand_xarray_parallel(base_dir, output_dir, num_workers, sources):
    stands = glob.glob(os.path.join(base_dir, '*'))
    os.makedirs(output_dir, exist_ok=True)

    args_list = [(stand, output_dir, sources) for stand in stands]

    with Pool(processes=num_workers) as pool:
        pool.map(process_single_stand, args_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel aggregation into xarray datasets per stand.")
    parser.add_argument('--base_dir', required=True, help='Base directory with processed stand imagery')
    parser.add_argument('--output_dir', required=True, help='Directory to save aggregated xarray datasets')
    parser.add_argument('--num_workers', type=int, default=cpu_count()-1, help='Number of parallel processes')
    parser.add_argument('--sources', required=False, default=None, nargs='+', help='Name of sources to process')

    args = parser.parse_args()

    build_stand_xarray_parallel(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        sources=args.sources
    )
