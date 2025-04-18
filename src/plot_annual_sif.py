import os
import glob
import math
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Tile reflectance time‐series plots for all .nc files in a directory"
)
parser.add_argument(
    '--input_dir', 
    required=True, 
    help='Directory containing .nc files to visualize'
)
parser.add_argument(
    '--ncols', 
    type=int, 
    default=None,
    help='Number of subplot columns (defaults to ceil(sqrt(N)))'
)
args = parser.parse_args()
input_dir = args.input_dir

# -----------------------------------------------------------------------------
# Discover files and compute grid layout
# -----------------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(input_dir, '*.nc')))
if not files:
    raise FileNotFoundError(f"No .nc files found in {input_dir!r}")

n_files = len(files)
ncols = args.ncols or math.ceil(math.sqrt(n_files))
nrows = math.ceil(n_files / ncols)

# -----------------------------------------------------------------------------
# Prepare month ticks
# -----------------------------------------------------------------------------
tick_locs   = pd.date_range("2001-01-01", "2001-12-31", freq="MS")
tick_pos    = tick_locs.dayofyear
tick_labels = tick_locs.strftime("%b")

# -----------------------------------------------------------------------------
# Create figure and axes grid
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4 * ncols, 3 * nrows),
    squeeze=False
)
axes_flat = axes.flatten()

# -----------------------------------------------------------------------------
# Loop over each file & axis
# -----------------------------------------------------------------------------
for ax, fp in zip(axes_flat, files):
    # Extract stand title from filename
    fname = os.path.basename(fp)  # e.g. "stand_FOO_timeseries.nc"
    stand_title = fname.removeprefix("stand_").removesuffix("_timeseries.nc")
    
    ds = xr.open_dataset(fp)
    avg_ref = ds['Reflectance'].mean(dim=['y','x','source'])
    
    df = pd.DataFrame({
        "date": pd.to_datetime(avg_ref.date.values),
        "reflectance": avg_ref.values
    })
    df["year"] = df["date"].dt.year
    
    for year, grp in df.groupby("year"):
        ax.plot(grp["date"].dt.dayofyear, grp["reflectance"], label=str(year))
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_title(stand_title, fontsize=10)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Reflectance")
    ax.grid(True)
    ax.legend(fontsize='small', title='Year', loc='upper right')

# -----------------------------------------------------------------------------
# Turn off any unused axes
# -----------------------------------------------------------------------------
for ax in axes_flat[n_files:]:
    ax.set_visible(False)

fig.tight_layout()
plt.show()
