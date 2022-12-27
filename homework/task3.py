import copy
import datetime
import itertools
from typing import Tuple

import helpers
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def find_nearest_in_netcdf(data, lat: float, lon: float) -> Tuple[int]:
    """Find indices of nearest latitude and longitude in NetCDF dataset.

    Args:
        data (netCDF4.Dataset): netCDF4 Dataset
        lat (float): Latitude coordinate
        lon (float): Longitutde coordinate

    Returns:
        Tuple[int]: (latitude index, longitude index)
    """
    lons = data.variables['longitude'][:].data
    slon = data.dimensions['longitude'].size

    lats = data.variables['latitude'][::-1].data
    slat = data.dimensions['latitude'].size

    lon_idx, _ = helpers.binary_search(lons, 0, slon - 1, lon)
    lat_idx, _ = helpers.binary_search(lats, 0, slat - 1, lat)

    return (slat - 1 - lat_idx, lon_idx)


def find_region_in_netcdf(data, coords: Tuple[float]) -> Tuple[int]:
    """Find indices of nearest latitude and longitude coordinates covering coordinate box specified by coords,

    llcrnrlat - Lower left corner latitude
    llcrnrlon - Lower left corner longitude
    urcrnrlat - Upper right corner latitude
    urcrnrlon - Upper right corner longitude

    Args:
        data (netCDF4.Dataset): netCDF dataset
        coords (Tuple[float]): Lower left corner and upper right corner lat and lon coordinates

    Returns:
        Tuple[int]: Indices of lower left corner and upper right cornet lats and lons in dataset
    """
    llcrnrlat, llcrnrlon, urcrnrlat, urcrnrlon = coords

    llcrnrlat_idx, llcrnrlon_idx = find_nearest_in_netcdf(data, llcrnrlat, llcrnrlon)
    urcrnrlat_idx, urcrnrlon_idx = find_nearest_in_netcdf(data, urcrnrlat, urcrnrlon)

    return (llcrnrlat_idx, llcrnrlon_idx, urcrnrlat_idx, urcrnrlon_idx)


def subtask1():

    data_path = helpers.get_data_path('ECMWF_prognozes.nc')
    cfg = helpers.load_yaml_config_file('netcdf_region_ssr.yaml')

    with Dataset(data_path, 'r') as ds:

        box = cfg['region']

        coords = (
            box['llcrnrlat'],
            box['llcrnrlon'],
            box['urcrnrlat'],
            box['urcrnrlon'],
        )

        lat1, lon1, lat2, lon2 = find_region_in_netcdf(ds, coords)

        ssr = ds.variables['ssr'][:, lat2:lat1, lon1:lon2]
        hours = tuple(cfg['hours'])

        lats = ds.variables['latitude'][lat2:lat1]
        longs = ds.variables['longitude'][lon1:lon2]

        abs_min = np.asarray(ssr[hours, :]).min()
        abs_max = np.asarray(ssr[hours, :]).max()

        fig = plt.figure()
        num_plots = len(hours)
        plt_cfg = cfg['plot']

        hours_since = ds.variables['time'].units.split('hours since ')[1][:-5]
        ref_date = datetime.datetime.strptime(hours_since, '%Y-%m-%d %H:%M')

        cmap = plt.cm.tab10
        cmaplist = [cmap(i) for i in range(cmap.N)]
        levels = cmap.N

        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, levels)

        for idx, hour in enumerate(hours):

            plot_num = int('{}1{}'.format(num_plots, idx + 1))
            ax = fig.add_subplot(plot_num)

            bmap = Basemap(
                llcrnrlat=ds.variables['latitude'][lat1 - 1],
                llcrnrlon=ds.variables['longitude'][lon1],
                urcrnrlat=ds.variables['latitude'][lat2],
                urcrnrlon=ds.variables['longitude'][lon2 - 1],
                resolution='i',
            )

            data = np.asarray(ssr[hour]) - np.asarray(ssr[0])

            bmap.drawcountries()
            bmap.drawcoastlines()
            bmap.drawmapboundary()

            lat, lon = np.meshgrid(lats, longs)
            normalize = mpl.colors.Normalize(vmin=abs_min, vmax=abs_max)

            bmap.contourf(
                lon,
                lat,
                data.T,
                latlon=True,
                levels=levels,
                cmap=cmap,
                norm=normalize,
            )

            hours_from = ds.variables['time'][hour].data
            dt = datetime.timedelta(hours=int(hours_from))
            date = ref_date + dt

            if idx == 0:
                ax.set_title(f'{date.year}-{date.month}-{date.day} {date.hour}:00')
            else:
                ax.set_title(f'{date.year}-{date.month}-{date.day} {date.hour}:00 (+{hour}h)')

        fig.subplots_adjust(right=0.9)
        fig.tight_layout()
        cbar_ax = fig.add_axes([0.8, 0.15, 0.05, 0.7])
        cbar_ax.set_xlabel(r"SSR, $\frac{J}{m^2}$")
        fig.colorbar(
            mpl.cm.ScalarMappable(normalize, cmap),
            cax=cbar_ax,
            ticks=np.linspace(abs_min, abs_max, levels + 1, endpoint=True),
        )

        main_dir = helpers.get_file_dir()
        fname = f"{plt_cfg['fname']}.{plt_cfg['format']}"
        savepath = main_dir / 'results' / fname

        fig.savefig(savepath, bbox_inches='tight', format='svg')


def subtask2():

    data_path = helpers.get_data_path('ECMWF_prognozes.nc')
    cfg = helpers.load_yaml_config_file('netcdf_plot_average.yaml')

    locs = cfg['locs']

    data_template = {}
    for i in range(24):
        data_template[i] = []

    for info in locs.values():
        info['data'] = copy.deepcopy(data_template)
        info['avg_data'] = copy.deepcopy(data_template)

    with Dataset(data_path, 'r') as ds:

        for info in locs.values():
            info['lat_idx'], info['lon_idx'] = find_nearest_in_netcdf(ds, info['lat'], info['lon'])

        hours_since = ds.variables['time'].units.split('hours since ')[1][:-5]
        ref_date = datetime.datetime.strptime(hours_since, '%Y-%m-%d %H:%M')

        fig, ax = plt.subplots()
        ax.set_xticks(list(range(0, 24, 2)))
        markers = itertools.cycle(('o', '^', 's', '*'))
        plt_cfg = cfg['plot']

        for loc, info in locs.items():

            data = ds.variables[plt_cfg['variable']]
            lat = info['lat_idx']
            lon = info['lon_idx']

            for idx, hours_from in enumerate(ds.variables['time'][1:]):

                idx += 1
                val = data[idx, lat, lon] - data[idx - 1, lat, lon]
                valW = val / 3600
                dt = datetime.timedelta(hours=int(hours_from))
                date = ref_date + dt
                info['data'][date.hour].append(valW)

            for hour, values in info['data'].items():
                info['avg_data'][hour] = sum(values) / len(values)

            x = list(info['avg_data'].keys())
            y = list(info['avg_data'].values())
            ax.plot(x, y, label=loc, marker=next(markers), linestyle=plt_cfg['linestyle'])

        ax.set_xlabel(r'{}'.format(plt_cfg['xlabel']))
        ax.set_ylabel(r'{}'.format(plt_cfg['ylabel']))
        ax.legend(loc=plt_cfg['legend_position'])
        ax.grid()

        main_dir = helpers.get_file_dir()
        fname = f"{plt_cfg['fname']}.{plt_cfg['format']}"
        savepath = main_dir / 'results' / fname
        fig.savefig(savepath, bbox_inches='tight', format=plt_cfg['format'])


if __name__ == '__main__':
    subtask1()
    subtask2()
