#!/usr/bin/env python3
'''
timeseries_aggregator.py

Compute a yearly statistic over a given month or month-range
from one or more NetCDF timeseries files (with a 'date' coordinate),
collapsing spatial (and any other) dims to a global average,
and plot them for comparison, optionally shading ±1 standard deviation.

Now accepts a dedicated forest-wide NetCDF for comparison and
an optional --normalize flag to plot differences from the forest average.
'''

import argparse
import glob
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def parse_months(month_str):
    '''
    Parse a month or month-range string (e.g. '7' or '6-8') into a list of ints.
    '''
    if '-' in month_str:
        start, end = map(int, month_str.split('-', 1))
        return list(range(start, end + 1))
    else:
        return [int(month_str)]


def gather_netcdf_inputs(inputs):
    '''
    Given a list of paths (files or dirs), return a sorted list of .nc files.
    '''
    nc_files = []
    for path in inputs:
        if os.path.isdir(path):
            nc_files.extend(glob.glob(os.path.join(path, '*.nc')))
        elif os.path.isfile(path) and path.lower().endswith('.nc'):
            nc_files.append(path)
        else:
            print(f'Warning: {path!r} is not a directory or .nc file, skipping.')
    return sorted(nc_files)


def compute_yearly_stats(ds, var, stat, months):
    '''
    Mask to selected months, collapse dims except 'date', group by calendar year,
    and compute the chosen statistic and std per year.
    Returns two 1D DataArrays indexed by 'year'.
    '''
    da = ds[var].where(ds['date'].dt.month.isin(months), drop=True)
    other_dims = [d for d in da.dims if d != 'date']
    if other_dims:
        da = da.mean(dim=other_dims)
    grp = da.groupby('date.year')
    if stat == 'mean':
        stat_ts = grp.mean(dim='date')
    elif stat == 'median':
        stat_ts = grp.median(dim='date')
    elif stat.endswith('th'):
        p = int(stat[:-2])
        stat_ts = da.groupby('date.year').quantile(p / 100, dim='date', skipna=True)
    else:
        raise ValueError(f'Unsupported statistic: {stat}')
    std_ts = grp.std(dim='date')
    return stat_ts, std_ts


def main():
    p = argparse.ArgumentParser(
        description='Plot yearly statistics (mean/median/percentile) over a month-range from NetCDF timeseries.'
    )
    p.add_argument(
        '--forest', '-f', required=True,
        help='Path to NetCDF file covering the whole forest'
    )
    p.add_argument(
        '--normalize', action='store_true',
        help='If set, plot each stand series as the difference from the forest average (forest line will be zero)'
    )
    p.add_argument(
        'inputs', nargs='+',
        help='Paths to NetCDF files or directories containing them'
    )
    p.add_argument(
        '--variable', '-v', required=True,
        help='Name of variable in the NetCDF to plot (e.g. Reflectance, NDVI, SIF)'
    )
    p.add_argument(
        '--stat', '-s', choices=['mean', 'median', '90th', '95th'], default='mean',
        help='Statistic to compute per year'
    )
    p.add_argument(
        '--months', '-m', default='1-12',
        help='Month or month-range to aggregate (e.g. 7 or 6-8). Default is all months'
    )
    p.add_argument(
        '--labels', '-l', nargs='*',
        help='Optional labels for each input (in same order as the gathered files)'
    )
    p.add_argument(
        '--ci', action='store_true',
        help='Shade ±1 standard deviation envelope around each line'
    )
    p.add_argument(
        '--show', action='store_true',
        help='Show plot interactively instead of saving to file'
    )
    p.add_argument(
        '--out', '-o', default='timeseries_comparison.png',
        help='Output filename for the plot (ignored if --show is set)'
    )
    args = p.parse_args()

    months = parse_months(args.months)
    # load forest stats
    forest_ds = xr.open_dataset(args.forest)
    forest_stat_ts, forest_std_ts = compute_yearly_stats(forest_ds, args.variable, args.stat, months)

    files = gather_netcdf_inputs(args.inputs)
    if not files:
        print('No NetCDF files found. Exiting.')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    shades = []

    # plot each stand
    for idx, filepath in enumerate(files):
        ds = xr.open_dataset(filepath)
        stat_ts, std_ts = compute_yearly_stats(ds, args.variable, args.stat, months)
        years = stat_ts['year'].values
        if args.normalize:
            fsel = forest_stat_ts.sel(year=stat_ts['year'])
            vals = stat_ts.values - fsel.values
        else:
            vals = stat_ts.values
        label = args.labels[idx] if (args.labels and idx < len(args.labels)) else os.path.splitext(os.path.basename(filepath))[0]
        line, = ax.plot(years, vals, marker='o', label=label)
        lines.append(line)
        if args.ci:
            std_vals = std_ts.values
            lower = vals - std_vals
            upper = vals + std_vals
            shade = ax.fill_between(years, lower, upper, alpha=0.2)
            shades.append(shade)
        else:
            shades.append(None)

    # plot forest series
    forest_years = forest_stat_ts['year'].values
    if args.normalize:
        zeros = np.zeros_like(forest_years)
        forest_line, = ax.plot(
            forest_years, zeros,
            linestyle='--', linewidth=2.5,
            color='k', label='Forest Average'
        )
        lines.append(forest_line)
        if args.ci:
            stdf = forest_std_ts.values
            lower = -stdf
            upper = stdf
            forest_shade = ax.fill_between(forest_years, lower, upper, color='k', alpha=0.2)
            shades.append(forest_shade)
        else:
            shades.append(None)
    else:
        forest_vals = forest_stat_ts.values
        forest_line, = ax.plot(
            forest_years, forest_vals,
            linestyle='--', linewidth=2.5,
            color='k', label='Forest Average'
        )
        lines.append(forest_line)
        if args.ci:
            lower = forest_vals - forest_std_ts.values
            upper = forest_vals + forest_std_ts.values
            forest_shade = ax.fill_between(forest_years, lower, upper, color='k', alpha=0.2)
            shades.append(forest_shade)
        else:
            shades.append(None)

    ax.set_xlabel('Year')
    if args.normalize:
        ax.set_ylabel(f'Difference of {args.stat.capitalize()} {args.variable} from Forest Average')
        ax.set_title(
            f'Difference of {args.stat.capitalize()} {args.variable} '
            f'for months {months[0]}–{months[-1]}'
        )
    else:
        ax.set_ylabel(f'{args.stat.capitalize()} of {args.variable}')
        ax.set_title(
            f'{args.stat.capitalize()} of {args.variable} '
            f'for months {months[0]}–{months[-1]}'
        )

    # interactive legend toggling
    leg = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)
        legline._origline = origline

    def on_pick(event):
        legline = event.artist
        origline = legline._origline
        vis = not origline.get_visible()
        origline.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        idx = lines.index(origline)
        shade = shades[idx]
        if shade is not None:
            shade.set_visible(vis)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if args.show:
        plt.show()
    else:
        fig.savefig(args.out, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved plot to {args.out}')

if __name__ == '__main__':
    main()
