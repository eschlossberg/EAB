import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import argparse

# Load dataset
# TODO: better description
parser = argparse.ArgumentParser(description="For making figure 1 figs")
parser.add_argument('--filepath', required=True, help='Path to the .nc file to be visualized')
args = parser.parse_args()

filepath = args.filepath

ds = xr.open_dataset(filepath)

# Average over spatial dimensions
avg_reflectance = ds['Reflectance'].mean(dim=['y', 'x', 'source'])

# Convert to DataFrame
df = pd.DataFrame({
    "date": pd.to_datetime(avg_reflectance.date.values),
    "reflectance": avg_reflectance.values
})
df["year"] = df["date"].dt.year
df["day_of_year"] = df["date"].dt.dayofyear

# Plot
plt.figure(figsize=(12, 6))

# One line per year
for year, group in df.groupby("year"):
    plt.plot(group["date"].dt.dayofyear, group["reflectance"], label=str(year))

# Format x-axis as months
# Create dummy dates from a non-leap year for label placement
tick_locs = pd.date_range("2001-01-01", "2001-12-31", freq="MS")
tick_pos = tick_locs.dayofyear
tick_labels = tick_locs.strftime("%b")

plt.xticks(ticks=tick_pos, labels=tick_labels)
plt.title(f"Daily Reflectance Over Time (Average over pixels)")
plt.xlabel("Month")
plt.ylabel("Mean Reflectance")
plt.grid(True)
plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()