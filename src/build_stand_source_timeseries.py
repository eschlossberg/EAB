import numpy as np
import rasterio
import glob
import os
import argparse

def build_time_series(base_dir, source, bands):
    stands = glob.glob(os.path.join(base_dir, 'stand_*'))

    for stand in stands:
        stand_id = os.path.basename(stand)
        source_dir = os.path.join(stand, source)
        
        if not os.path.isdir(source_dir):
            print(f"Source {source} not found for {stand_id}. Skipping.")
            continue

        tifs = sorted(glob.glob(os.path.join(source_dir, '*.tif')))

        if not tifs:
            print(f"No imagery found for {stand_id} from source {source}. Skipping.")
            continue

        stack_list = []

        for tif in tifs:
            with rasterio.open(tif) as src:
                img = src.read(bands)  # bands is a list
                stack_list.append(img)

        # Convert to NumPy array with shape (time, bands, rows, cols)
        time_series = np.stack(stack_list, axis=0)

        # Save array
        bands_str = '_'.join([str(b) for b in bands])
        npy_path = os.path.join(source_dir, f'{source}_bands_{bands_str}_timeseries.npy')
        np.save(npy_path, time_series)

        print(f"Saved time series: {stand_id} ({source}), bands: {bands_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NumPy time series arrays for specified imagery source and bands.")
    parser.add_argument('--base_dir', required=True, help='Base directory with processed stand imagery')
    parser.add_argument('--source', required=True, help='Imagery source folder name (e.g., MODIS, PACE)')
    parser.add_argument('--bands', required=True, nargs='+', type=int, help='Band numbers to extract (e.g., 1 2 3)')

    args = parser.parse_args()

    build_time_series(
        base_dir=args.base_dir,
        source=args.source,
        bands=args.bands
    )

