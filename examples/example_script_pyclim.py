##############################################################
#### Example script of the pyClim package ####################
#### Author: Alejandro Rodríguez Sánchez #####################
#### Contact: ars.rodriguezs@gmail.com #######################
##############################################################

import numpy as np
import matplotlib
import pandas as pd
import datetime as dt

import os

# ruff: noqa

# Import pyclim
from pyClim import (
    quality_control,
    compute_climate,
    compute_records,
    compute_and_plot_exceedances,
    plot_anomalies,
    plot_accumulated_anomalies,
    plot_data_vs_climate,
    plot_data_vs_climate_withrecords,
    plot_data_vs_climate_withrecords_multivar,
    plot_data_and_accum_anoms,
    plot_data_and_annual_cycle,
    plot_periodaverages,
    plot_records_count,
    plot_timeseries,
    plot_variable_trends,
    plot_annual_cycles,
    annual_meteogram,
    categories_evolution,
    timeseries_extremevalues,
    get_annual_cycle,
)

matplotlib.use("Agg")

# light_jet = cmap_map(lambda x: x*0.85, mpl.cm.jet)


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""

    ### From: https://github.com/CJP123/continuous_cmap/

    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""

    ### From: https://github.com/CJP123/continuous_cmap/

    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None, N_colors=256):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""

    ### From: https://github.com/CJP123/continuous_cmap/

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap(
        "my_cmp", segmentdata=cdict, N=N_colors
    )
    return cmp


# Create metadata
metadata = pd.DataFrame(["example", "example", 361.00, -361.00, 461]).T
metadata.columns = ["Name", "ID", "Latitude", "Longitude", "Altitude"]

# Some useful variables
year_to_plot = 2025
climate_normal_period = [1991, 2020]
variables = ["Tmin", "Tmean", "Tmax", "Rainfall", "WindSpeed"]
database = "Example"

# Mapping variables
units_list = {}  # ['ºC','ºC','ºC', 'm/s'] #,'mm']
units_list["Tmin"] = "ºC"
units_list["Tmean"] = "ºC"
units_list["Tmax"] = "ºC"
units_list["Rainfall"] = "mm"
units_list["WindSpeed"] = "m/s"

wd_map = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 125,
    "SSE": 147.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}
# colormap = matplotlib.cm.get_cmap('RdBu_r')


metadatos_sta = metadata[metadata.ID == "example"]
codigo_sta = metadatos_sta.ID.values[0]
# nombres_mod[listanombres] = nombres_mod[listanombres].replace('/','-')
station_name = str(metadatos_sta.Name.values[0])
station_name = station_name.replace("/", "-")

plotdir = os.path.join(os.getcwd(), "plots/%s/%s" % (database, "example"))
plotdir = plotdir.replace("\\", "/")
if os.path.isdir(plotdir) is False:
    try:
        os.mkdir(plotdir)
    except OSError:
        print("Creation of the directory %s failed" % plotdir)
        print("Setting plotdir to the same directory of the script...")
        plotdir = os.path.join(os.getcwd())
    else:
        print("Successfully created the directory %s " % plotdir)


# Read data
input_file = os.path.join(os.getcwd(), "data/example_data.txt")
df1 = pd.read_csv(input_file, sep=";", decimal=".", header=0, encoding="latin-1")

df1["Date"] = pd.to_datetime(df1.iloc[:, 0])
df1 = df1.set_index("Date")

df1["Day"] = df1.index.day
df1["Month"] = df1.index.month
df1["Year"] = df1.index.year

yeartoday = int(df1["Year"].values[-1])
iniyear = int(df1.Year.values[0])


df1["DayofYear"] = df1.index.dayofyear

df1["Accum.Rainfall"] = df1.groupby(df1.index.year)["Rainfall"].cumsum()

climate_vars = [
    "Tmax",
    "Tmean",
    "Tmin",
    "Rainfall",
    "Accum.Rainfall",
    "WindSpeed",
    "WindMaxSpeed",
]

df1_complete = df1.reindex(
    pd.date_range("%i-01-01" % iniyear, "%i-12-31" % yeartoday, freq="1D")
)  # Fill data in case there is any missing date

df1_complete["Day"] = df1_complete.index.day
df1_complete["Month"] = df1_complete.index.month
df1_complete["Year"] = df1_complete.index.year
df1_complete["DayofYear"] = df1_complete.index.dayofyear

# Pass quality control
df1_complete = quality_control(
    df1_complete, ["Tmean", "WindSpeed"], t_units="C", wind_units="km/h"
)
# diff = df1_complete1[df1_complete.index.year == 2025]['Tmean'] - df1_complete[df1_complete.index.year == 2025]['Tmean']

climate_df = compute_climate(
    df1_complete.loc[
        df1_complete.index.isin(df1.index), df1_complete.columns.isin(df1.columns)
    ],
    climate_vars,
    [1991, 2020],
    separate_df=False,
)
climate_df_sep = compute_climate(
    df1_complete.loc[:, df1_complete.columns.isin(df1.columns)],
    climate_vars,
    [1991, 2020],
)


### Anomalies
plot_anomalies(
    df1_complete,
    "Tmean",
    "ºC",
    climate_normal_period,
    database,
    station_name,
    plotdir + "/daily_anomalies_Tmean.png",
    window=12,
    freq="1D",
)
plot_anomalies(
    df1_complete,
    "Tmean",
    "ºC",
    climate_normal_period,
    database,
    station_name,
    plotdir + "/monthly_anomalies_Tmean.png",
    window=12,
    freq="1ME",
)

#### Accumulated anomalies
plot_accumulated_anomalies(
    df1_complete,
    "Tmean",
    "ºC",
    2025,
    climate_normal_period,
    database,
    station_name,
    plotdir + "/Tmean_accum_anoms_daily.png",
    freq="1D",
)
plot_accumulated_anomalies(
    df1_complete,
    "Tmean",
    "ºC",
    2025,
    climate_normal_period,
    database,
    station_name,
    plotdir + "/Tmean_accum_anoms_weekly.png",
    freq="1W",
)
plot_accumulated_anomalies(
    df1_complete,
    "Tmean",
    "ºC",
    2025,
    climate_normal_period,
    database,
    station_name,
    plotdir + "/Tmean_accum_anoms_monthly.png",
    freq="1ME",
)


colors_anom_bars = ["#34b1eb", "#eb4034"]
levels_anom_bars = [0, 1]
cmap_anom_bars = get_continuous_cmap(
    colors_anom_bars, levels_anom_bars, 2
)  # Only works with HEX colors

# Plot data from a certain period versus the climatological normal
variable = "Tmean"
units = "ºC"
inidate = dt.datetime(2025, 1, 1)  # df1.index[-1] - dt.timedelta(days=ndays)
enddate = dt.datetime(2025, 12, 31)
plot_data_vs_climate(
    df1_complete,
    climate_df_sep,
    variable,
    units,
    inidate,
    enddate,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir + "/%speriodtimeseries_climatemedian19912020.png" % variable,
    kind="bar",
    fillcolor_gradient=False,
)

### Plot mean or median value of a certain period for every year with data
for var in list(set(df1_complete.columns) & set(variables)):
    units = units_list[var]
    plot_periodaverages(
        df1_complete,
        climate_df_sep,
        var,
        units,
        dt.datetime(2025, 6, 1),
        dt.datetime(2025, 9, 1),
        station_name,
        database,
        plotdir + "/%speriodaverages.png" % var,
        stat="median",
        window=10,
    )


records_df_allvars = pd.DataFrame()
multiyearrecords_df_allvars = (
    pd.DataFrame()
)  # For saving multiple variable records' DataFrames

varis = ["Tmax", "Tmean", "Rainfall"]
units_varis = []
for i in range(len(varis)):
    units_varis.append(units_list[varis[i]])

    multiyearrecords_df = compute_records(
        df1_complete, varis[i], df1_complete.index.year.unique()
    )  # Compute records for variable

    multiyearrecords_df_allvars = pd.concat(
        [multiyearrecords_df_allvars, multiyearrecords_df], axis=1
    )


# Plot annual records
plot_records_count(
    multiyearrecords_df_allvars,
    "Tmean",
    database,
    station_name,
    plotdir + "/annual_records_Tmean.png",
    freq="day",
)  # Plot number of days exceeding daily records

plot_records_count(
    multiyearrecords_df_allvars,
    "Tmean",
    database,
    station_name,
    plotdir + "/monthly_records_Tmean.png",
    freq="month",
)  # Plot number of days exceeding daily records

# Multiyear records
plot_data_vs_climate_withrecords(
    df1_complete,
    climate_df_sep,
    multiyearrecords_df_allvars,
    "Tmean",
    "ºC",
    inidate,
    enddate,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir + "/%stimeseries_climatemedian19912020_withrecords.png" % varis[i],
    kind="bar",
    fillcolor_gradient=False,
)

plot_data_vs_climate_withrecords_multivar(
    df1_complete,
    climate_df_sep,
    multiyearrecords_df_allvars,
    varis,
    units_varis,
    inidate,
    enddate,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir
    + "/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords.png"
    % varis[i],
    kind="bar",
    fillcolor_gradient=False,
)
plot_data_vs_climate_withrecords_multivar(
    df1_complete,
    climate_df_sep,
    multiyearrecords_df_allvars,
    varis,
    units_varis,
    inidate,
    enddate,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir
    + "/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords_std.png"
    % varis[i],
    kind="bar",
    fillcolor_gradient=False,
    use_std=True,
)
plot_data_vs_climate_withrecords_multivar(
    df1_complete,
    climate_df_sep,
    multiyearrecords_df_allvars,
    varis,
    units_varis,
    inidate,
    enddate,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir
    + "/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords_std_line.png"
    % varis[i],
    kind="line",
    fillcolor_gradient=False,
    use_std=True,
)

# Plot variable and anomalies
varis = ["Tmin", "Tmax", "Tmean"]
units_varis = []
for i in range(len(varis)):
    units_varis.append(units_list[varis[i]])

# Plot variable and accumulated anomaly
plot_data_and_accum_anoms(
    df1_complete,
    climate_df_sep,
    year_to_plot,
    varis,
    units_varis,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir,
    secondplot_type="accum",
    w=7,
)
plot_data_and_accum_anoms(
    df1_complete,
    climate_df_sep,
    year_to_plot,
    varis,
    units_varis,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir,
    secondplot_type="moving",
    w=7,
)
# Plot variable and accumulated mean or value
plot_data_and_annual_cycle(
    df1_complete,
    climate_df_sep,
    year_to_plot,
    varis,
    units_varis,
    cmap_anom_bars,
    database,
    climate_normal_period,
    station_name,
    plotdir,
    fillcolor_gradient=True,
)


# Plot annual meteogram
df1_complete["Temp"] = df1_complete["Tmean"]
climate_df_sep["Temp"] = climate_df_sep["Tmean_median"]
climate_df_sep["WindSpeed"] = climate_df_sep["WindSpeed_median"]
annual_meteogram(
    df1_complete,
    climate_df_sep,
    year_to_plot,
    climate_normal_period,
    database,
    station_name,
    plotdir + "/%i_meteogram.png" % year_to_plot,
)
#    df1_complete['Temp'] = get_yearly_cycle(df1_complete, climate_df_sep, ['Tmean'])['Tmean']
#    climate_df_sep['Temp_median'] = get_yearly_cycle(df1_complete, climate_df_sep, ['Tmean'])['Tmean_median']
annual_meteogram(
    df1_complete,
    climate_df_sep,
    year_to_plot,
    climate_normal_period,
    database,
    station_name,
    plotdir + "/%i_meteogram_bars.png" % year_to_plot,
    plot_anoms=True,
)

# Timeseries
plot_timeseries(
    df1_complete,
    climate_df_sep,
    varis[-1],
    units_varis[-1],
    climate_normal_period,
    database,
    station_name,
    plotdir + "/%s_timeseries.png" % varis[-1],
    plot_MA=False,
    climate_stat="median",
    window=30,
)
plot_timeseries(
    df1_complete,
    climate_df_sep,
    varis[-1],
    units_varis[-1],
    climate_normal_period,
    database,
    station_name,
    plotdir + "/%s_timeseries_MA.png" % varis[-1],
    plot_MA=True,
    climate_stat="median",
    window=365,
)


### Yearly cycles

tmax_accum_anom = get_annual_cycle(
    df1_complete, climate_df, varis
)  # Get yearly cycle of accumulated anomalies

colores_calidos = ["#800008", "#B80101", "#ff949b"]
colores_frios = ["#2205A8", "#1542F5", "#7397fb"]
# colores_calidos = ['#ff949b',"#B80101",'#800008']
plot_annual_cycles(
    df1_complete,
    "Tmax",
    "ºC",
    year_to_plot,
    climate_normal_period,
    database,
    station_name,
    colores_frios,
    plotdir + "/Tmax_yearlycycle_colorlowest.png",
    yearly_cycle=True,
    criterion="lowest",
)
plot_annual_cycles(
    tmax_accum_anom,
    "Tmax",
    "ºC",
    year_to_plot,
    climate_normal_period,
    database,
    station_name,
    colores_calidos,
    plotdir + "/Tmax_yearlycycle_colorhighest.png",
    yearly_cycle=True,
    criterion="highest",
)


### Seasonal plots
for var in sorted(
    list(set(df1_complete.columns) & set(variables)), key=lambda x: variables.index(x)
):
    units = units_list[var]
    # Extreme values
    timeseries_extremevalues(
        df1_complete,
        var,
        units,
        climate_normal_period,
        database,
        station_name,
        plotdir + "/Annualextremevalues_%s_lines.png" % var,
        time_scale="Year",
    )
    timeseries_extremevalues(
        df1_complete,
        var,
        units,
        climate_normal_period,
        database,
        station_name,
        plotdir + "/seasonalextremevalues_%s_lines.png" % var,
        time_scale="season",
    )

    # Plot variables evolution in time
    plot_variable_trends(
        df1_complete,
        var,
        units,
        database,
        station_name,
        plotdir + "/%s_withmean.png" % var,
        averaging_period=5,
        grouping="year",
        grouping_stat="mean",
        rain_limit=1,
    )

    plot_variable_trends(
        df1_complete,
        var,
        units,
        database,
        station_name,
        plotdir + "/%s_sum_withmean.png" % var,
        averaging_period=5,
        grouping="year",
        grouping_stat="sum",
        rain_limit=1,
    )

    plot_variable_trends(
        df1_complete,
        var,
        units,
        database,
        station_name,
        plotdir + "/%s_withmean_season.png" % var,
        averaging_period=5,
        grouping="season",
        grouping_stat="mean",
        rain_limit=1,
    )
    plot_variable_trends(
        df1_complete,
        var,
        units,
        database,
        station_name,
        plotdir + "/%s_withmean_month.png" % var,
        averaging_period=5,
        grouping="month",
        grouping_stat="mean",
        rain_limit=1,
    )

    # Exceedances of threshold values
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_annualexceedances_withmeans_2b.png" % var,
        threshold=20,
        time_scale="year",
        upwards=True,
        plot_means=True,
        averaging_period=10,
    )
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_annualexceedances_b.png" % var,
        threshold=20,
        time_scale="year",
        upwards=True,
    )
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_monthlyexceedances.png" % var,
        threshold=20,
        time_scale="month",
        upwards=True,
    )
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_monthlyexceedances_withmeans.png" % var,
        threshold=20,
        time_scale="month",
        upwards=True,
        plot_means=True,
        averaging_period=10,
    )
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_seasonalexceedances.png" % var,
        threshold=20,
        time_scale="season",
        upwards=True,
    )
    compute_and_plot_exceedances(
        df1_complete,
        var,
        database,
        station_name,
        plotdir + "/%s_seasonalexceedances_withmeans.png" % var,
        threshold=20,
        time_scale="season",
        upwards=True,
        plot_means=True,
        averaging_period=10,
    )

    # Categorize data
    categories = [0, 5, 10, 15, 20, 25, 30]
    colors = ["blue", "cyan", "green", "yellow", "orange", "red"]
    categories_evolution(
        df1_complete,
        "Tmean",
        "ºC",
        categories,
        [str(x) for x in range(1, len(categories))],
        colors,
        database,
        station_name,
        plotdir + "/categories_Tmean.png",
        time_scale="year",
    )
    categories_evolution(
        df1_complete,
        "Tmean",
        "ºC",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_Tmean_default.png",
        time_scale="year",
    )
    categories_evolution(
        df1_complete,
        "Tmean",
        "ºC",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_Tmean_season_default.png",
        time_scale="season",
    )
    categories_evolution(
        df1_complete,
        "Tmean",
        "ºC",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_Tmean_month_default.png",
        time_scale="month",
    )

    categories = [0, 1, 5, 10, 15, 20, 25, 30]
    colors = ["lightgray", "blue", "cyan", "green", "yellow", "orange", "red"]
    categories_evolution(
        df1_complete,
        "Rainfall",
        "mm",
        categories,
        [str(x) for x in range(1, len(categories))],
        colors,
        database,
        station_name,
        plotdir + "/categories_rainfall.png",
        time_scale="year",
    )
    categories_evolution(
        df1_complete,
        "Rainfall",
        "mm",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_rainfall_default.png",
        time_scale="year",
    )
    categories_evolution(
        df1_complete,
        "Rainfall",
        "mm",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_rainfall_season_default.png",
        time_scale="season",
    )
    categories_evolution(
        df1_complete,
        "Rainfall",
        "mm",
        categories,
        [],
        colors,
        database,
        station_name,
        plotdir + "/categories_rainfall_month_default.png",
        time_scale="month",
    )
