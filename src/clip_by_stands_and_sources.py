import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os
import glob
import argparse

def clip_imagery(shapefile_path, imagery_sources, raw_imagery_folder, output_folder, stand_id_field, target_crs):
    stands = gpd.read_file(shapefile_path).to_crs(target_crs)

    for source in imagery_sources:
        imagery_path = os.path.join(raw_imagery_folder, source)
        imagery_files = sorted(glob.glob(os.path.join(imagery_path, '*.tif')))

        print(f"\nProcessing imagery source: {source}")

        for img_file in imagery_files:
            date_str = os.path.basename(img_file).split('.')[0]

            with rasterio.open(img_file) as src:
                stands_proj = stands.to_crs(src.crs)

                for idx, stand in stands_proj.iterrows():
                    stand_geom = [stand.geometry]
                    stand_id = stand[stand_id_field]

                    try:
                        out_image, out_transform = mask(src, stand_geom, crop=True)
                    except Exception as e:
                        print(f"Stand {stand_id} skipped ({e})")
                        continue

                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    stand_source_dir = os.path.join(output_folder, f'stand_{stand_id}', source)
                    os.makedirs(stand_source_dir, exist_ok=True)

                    output_path = os.path.join(stand_source_dir, f'{date_str}.tif')

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                    print(f"Saved: Stand {stand_id}, Source {source}, Date {date_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip imagery by stand polygons and imagery source.")
    parser.add_argument('--shapefile', required=True, help='Path to tree stands shapefile')
    parser.add_argument('--imagery_sources', required=True, nargs='+', help='List imagery source folder names')
    parser.add_argument('--raw_imagery_folder', required=True, help='Path to raw imagery directory')
    parser.add_argument('--output_folder', required=True, help='Output directory for processed imagery')
    parser.add_argument('--stand_id_field', default='stand_id', help='Field name for stand ID in shapefile')
    parser.add_argument('--target_crs', default='EPSG:4326', help='Target CRS for processing')

    args = parser.parse_args()

    clip_imagery(
        shapefile_path=args.shapefile,
        imagery_sources=args.imagery_sources,
        raw_imagery_folder=args.raw_imagery_folder,
        output_folder=args.output_folder,
        stand_id_field=args.stand_id_field,
        target_crs=args.target_crs
    )

