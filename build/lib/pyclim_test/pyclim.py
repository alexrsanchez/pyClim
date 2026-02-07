# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 13:27:10 2025

@author: aleja
"""

import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import warnings
from scipy import stats

##############################################################
#### MÓDULO CON FUNCIONES PARA GRAFICAR SERIES TEMPORALES ####
#### Autor: Alejandro Rodríguez Sánchez ######################
#### Contacto: ars.rodriguezs@gmail.com ######################
##############################################################

def quality_control(df,vars_to_check, t_units='C', wind_units='m/s'):
    '''
    This function allows to perform a quality control in a DataFrame and returns it without suspicious or bad data.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    vars_to_check: list
        List of columns to check
    t_units: str
        The initial letter of the temperature scale (C for Celsius; F for Fahrenheit; K for Kelvin)
    wind_units: str
        The wind units (km/h for kilometers per hour; m/s for meters per second; mph for miles per hour; or kn for knots)

    Return
    ------
    cured_df: DataFrame
    '''

    upper_t_limit = {}
    upper_wind_limit = {}
    lower_t_limit = {}

    upper_t_limit['C'] = 50
    upper_t_limit['F'] = 1.8
    upper_t_limit['K'] = 323.15

    lower_t_limit['C'] = -50
    lower_t_limit['F'] = -50*1.8 + 32
    lower_t_limit['K'] = 223.15

    upper_wind_limit['km/h'] = 200
    upper_wind_limit['m/s'] = 200/3.6
    upper_wind_limit['mph'] = 200/1.609344
    upper_wind_limit['kn'] = 20/1.852

    for var in vars_to_check:
        #### QUALITY CONTROL 1: REMOVE unrealistic values (outliers)
        #### Criteria for selecting outliers: Temp > 50ºC or < -50ºC; WindSpeed > 200 km/h.
        if var.lower() in ['temp', 'tmean', 'tmin', 'tmax', 't']:
            df[var] = df[var].where(df[var] <= upper_t_limit[t_units], np.nan) # Keep only values lower or equal than upper limit
            df[var] = df[var].where(df[var] >= lower_t_limit[t_units], np.nan)

        elif var in ['windspeed', 'wspd']:
            df[var] = df[var].where(df[var] <= upper_wind_limit[wind_units], np.nan)
        
        #### QUALITY CONTROL 2: REMOVE unrealistic values (above or below 5 times the standard deviation)
        df_std = df.groupby(['Month', 'Day']).std(numeric_only=True).rename(columns={var:var+'_std'})[var+'_std']
        df_mean = df.groupby(['Month', 'Day']).mean(numeric_only=True).rename(columns={var:var+'_mean'})[var+'_mean']
        
        df_var = df[[var, 'Month', 'Day']].merge(df_std, on=['Month', 'Day'])
        df_var = df_var.merge(df_mean, on=['Month', 'Day'])
        df_var.index = df.index
        df_var['mean+5std'] = df_var[var+'_mean'] + 5*df_var[var+'_std']
        df_var['mean-5std'] = df_var[var+'_mean'] - 5*df_var[var+'_std']


        df_var[var] = df_var[var].where((df_var[var] <= df_var['mean+5std']) & (df_var[var] >= df_var['mean-5std']) , np.nan)
        df[var] = df_var[var]
    
    return df

def compute_climate(df, variables, climate_period, separate_df=True):
    '''
    This function allows to compute the climatology of a certain timeseries.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    variables: list
        List of the variables for which the climatology is going to be computed
    climate_period: list
        List of the first and last year of the desired reference climate period
    separate_df: boolean
        If true, creates a new DataFrame with the values of the climatology. If false, appends those values to the input DataFrame.

    Return
    ------
    climate_df: DataFrame
        Returns a DataFrame with the same data of the input data, but with additional columns representing the climatology of each selected variable
    '''
    ##### Step 1. Compute daily climatologies for the period defined by "climate_period"
    #if df.index.month[-1] != 12 or df.index.month[-1] != 31:
    #    df = df.reindex(pd.date_range('%i-01-01' %df.index.year.min(), '%i-12-31' %df.index.year.max(), freq='1D'))

    climate_df = df[(df.index.year >= climate_period[0]) & (df.index.year <= climate_period[1])]
    all_stats = pd.DataFrame()
    for var in variables:
        daily_statistics = pd.DataFrame()
        daily_statistics['%s_p005' %var] = climate_df.groupby(['Month','Day'])[var].quantile(0.05)
        daily_statistics['%s_p010' %var] = climate_df.groupby(['Month','Day'])[var].quantile(0.1)
        daily_statistics['%s_p090' %var] = climate_df.groupby(['Month','Day'])[var].quantile(0.9)
        daily_statistics['%s_p095' %var] = climate_df.groupby(['Month','Day'])[var].quantile(0.95)
        daily_statistics['%s_mean' %var] = climate_df.groupby(['Month','Day'])[var].mean()
        daily_statistics['%s_median' %var] = climate_df.groupby(['Month','Day'])[var].median()
        daily_statistics['%s_min' %var] = climate_df.groupby(['Month','Day'])[var].min()
        daily_statistics['%s_max' %var] = climate_df.groupby(['Month','Day'])[var].max()
        daily_statistics['%s_std' %var] = climate_df.groupby(['Month','Day'])[var].std()

        all_stats = pd.concat([all_stats, daily_statistics], axis=1)

    if separate_df == True:
        climate_df = df[['Month','Day']].join(all_stats, on=['Month','Day'])
    else:
        climate_df = df.join(all_stats, on=['Month','Day'])

    return climate_df


def fill_between_colormap(x,y1,y2,cmap, **kwargs):
    cmap = matplotlib.cm.get_cmap('RdBu_r')
    
    xx = x.values
    yy1 = y1.values
    yy2 = y2.values
    
    yy = yy1 - yy2
    
    extreme_value = max(abs(np.nanmin(yy)),abs(np.nanmax(yy)))

    normalize = matplotlib.colors.Normalize(vmin=-extreme_value, vmax=extreme_value)
    npts = len(xx)
    for i in range(npts - 1):
        a = plt.fill_between([xx[i], xx[i+1]],
                             [yy1[i], yy1[i+1]],
                             [yy2[i], yy2[i+1]],
                             color=cmap(normalize(yy[i])),
                             edgecolor=None, **kwargs)

    return cmap, normalize

def compute_daily_records_oneyear(df, variable, year_to_compute):
    '''
    This function allows to compute the exceedances of the extreme values.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    variable: list
        Variable for which the exceedances are going to be computed
    year_to_compute: int
        Year of the data for which the exceedances are going to be computed

    Return
    ------
    c: DataFrame
        Returns a DataFrame the records occurred during "year_to_compute"
    '''


    ###########################################################################
    #### COMPUTES DAILY, MONTHLY AND ABSOLUTE RECORDS UP TO "year_to_plot" ####
    ###########################################################################
    if 'Day' not in df.columns:
        df['Day'] = df.index.day
    if 'Month' not in df.columns:
        df['Month'] = df.index.month


    # Compute maximum and minimum values by day, previous to "year_to_plot"
    amax = df[df.index.year < year_to_compute].groupby(['Month','Day'])[variable].max()
    amin = df[df.index.year < year_to_compute].groupby(['Month','Day'])[variable].min()

    # Compute maximum and minimum values by month, previous to "year_to_plot"
    mmax = df[df.index.year < year_to_compute].groupby(['Month'])[variable].max()
    mmin = df[df.index.year < year_to_compute].groupby(['Month'])[variable].min()

    # Compute absolute maximum and minimum valuesprevious to "year_to_plot"
    absmax = df[df.index.year < year_to_compute][variable].max()
    absmin = df[df.index.year < year_to_compute][variable].min()

    amax_1 = amax.reset_index()
    mmax_1 = mmax.reset_index()
    amax_concat = pd.merge(amax_1,mmax_1, on='Month', how='left')
    amax_concat.columns = ['Month','Day',variable+'_maxdaily',variable+'_maxmonthly']
    amax_concat = amax_concat.set_index(['Month','Day'])

    amin_1 = amin.reset_index()
    mmin_1 = mmin.reset_index()

    amin_concat = pd.merge(amin_1,mmin_1, on='Month')
    amin_concat.columns = ['Month','Day',variable+'_mindaily',variable+'_minmonthly']
    amin_concat = amin_concat.set_index(['Month','Day'])

    # Current year records
    b = df.loc[df.index.year == year_to_compute, ['Month','Day',variable]].groupby(['Month','Day']).max()
    #bmin = df_climate.loc[df.index.year == year_to_plot, ['Month','Day',variable]].groupby(['Month','Day']).min()
    #c = pd.concat([amax,amin,,bmax,bmin], axis=1)
    c = pd.concat([amax_concat,amin_concat,b], axis=1)
    c = c[~((c.index.get_level_values(0) == 2) & (c.index.get_level_values(1) == 29))] # Remove leap day
    c.columns = [variable+'_Daymax', variable+'_Monmax', variable+'_Daymin', variable+'_Monmin',
                 variable+'_%i' %year_to_compute]
    
    # Fill columns with daily, monthly and absolute records
    c[variable+'_dayrmax'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] > c['%s_Daymax' %variable])
    c[variable+'_dayrmin'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] < c['%s_Daymin' %variable])
    c[variable+'_monrmax'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] > c['%s_Monmax' %variable])
    c[variable+'_monrmin'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] < c['%s_Monmin' %variable])
    c[variable+'_absrmax'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] > absmax)
    c[variable+'_absrmin'] = c[variable+'_%i' %year_to_compute].where(c[variable+'_%i' %year_to_compute] < absmin)
    c[variable+'_dayrmax_diff'] = c[variable+'_%i' %year_to_compute] - c['%s_Daymax' %variable]
    c[variable+'_dayrmin_diff'] = c[variable+'_%i' %year_to_compute] - c['%s_Daymin' %variable]
    c[variable+'_monrmax_diff'] = c[variable+'_%i' %year_to_compute] - c['%s_Monmax' %variable]
    c[variable+'_monrmin_diff'] = c[variable+'_%i' %year_to_compute] - c['%s_Monmin' %variable]
    c[variable+'_absrmax_diff'] = c[variable+'_%i' %year_to_compute] - absmax
    c[variable+'_absrmin_diff'] = c[variable+'_%i' %year_to_compute] - absmin
    dates = pd.date_range(pd.to_datetime('01-01-%i' %year_to_compute, format='%d-%m-%Y'), pd.to_datetime('31-12-%i' %year_to_compute, format='%d-%m-%Y'), freq='1D')
    dates = dates[~((dates.day ==29) & (dates.month == 2))]
    c = c.reset_index().set_index(dates)
    #c[variable+'_%irecords' %year_to_compute] = c[[variable+'_%idayrmax' %year_to_compute,variable+'_%idayrmin' %year_to_compute]].max(axis=1)
    #c[variable+'_%irecords_diff' %year_to_compute] = c[[variable+'_%idayrmax_diff' %year_to_compute,variable+'_%idayrmin_diff' %year_to_compute]].max(axis=1)
    return c

def compute_daily_records(df, variable, years):
    '''
    This function allows to compute the exceedances of the extreme values.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    variable: list
        Variable for which the exceedances are going to be computed
    years: list
        Years of the data for which the exceedances are going to be computed

    Return
    ------
    c: DataFrame
        Returns a DataFrame the records occurred during each year included in "years"
    '''

    ###########################################################################
    #### COMPUTES DAILY, MONTHLY AND ABSOLUTE RECORDS UP TO "y" ####
    ###########################################################################
    c1 = pd.DataFrame()
    if 'Day' not in df.columns:
        df['Day'] = df.index.day
    if 'Month' not in df.columns:
        df['Month'] = df.index.month

    for y in years:

        # Compute maximum and minimum values by day
        amax = df[df.index.year < y].groupby(['Month','Day'])[variable].max()
        amin = df[df.index.year < y].groupby(['Month','Day'])[variable].min()

        # Compute maximum and minimum values by month
        mmax = df[df.index.year < y].groupby(['Month'])[variable].max()
        mmin = df[df.index.year < y].groupby(['Month'])[variable].min()

        # Compute absolute maximum and minimum valuesprevious to "y"
        absmax = df[df.index.year < y][variable].max()
        absmin = df[df.index.year < y][variable].min()

        amax_1 = amax.reset_index()
        mmax_1 = mmax.reset_index()
        amax_concat = pd.merge(amax_1,mmax_1, on='Month', how='left')
        amax_concat.columns = ['Month','Day',variable+'_maxdaily',variable+'_maxmonthly']
        amax_concat = amax_concat.set_index(['Month','Day'])

        amin_1 = amin.reset_index()
        mmin_1 = mmin.reset_index()

        amin_concat = pd.merge(amin_1,mmin_1, on='Month')
        amin_concat.columns = ['Month','Day',variable+'_mindaily',variable+'_minmonthly']
        amin_concat = amin_concat.set_index(['Month','Day'])

        # Current year records
        b = df.loc[df.index.year == y, ['Month','Day',variable]].groupby(['Month','Day']).max()
        c = pd.concat([amax_concat,amin_concat,b], axis=1)
        c = c[~((c.index.get_level_values(0) == 2) & (c.index.get_level_values(1) == 29))] # Remove leap day
        c.columns = [variable+'_Daymax', variable+'_Monmax', variable+'_Daymin', variable+'_Monmin',
                     variable+'_%i' %y]
           
        # Fill columns with daily, monthly and absolute records
        c[variable+'_dayrmax'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] > c['%s_Daymax' %variable])
        c[variable+'_dayrmin'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] < c['%s_Daymin' %variable])
        c[variable+'_monrmax'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] > c['%s_Monmax' %variable])
        c[variable+'_monrmin'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] < c['%s_Monmin' %variable])
        c[variable+'_absrmax'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] > absmax)
        c[variable+'_absrmin'] = c[variable+'_%i' %y].where(c[variable+'_%i' %y] < absmin)

        c[variable+'_dayrmax_diff'] = c[variable+'_%i' %y] - c['%s_Daymax' %variable]
        c[variable+'_dayrmin_diff'] = c[variable+'_%i' %y] - c['%s_Daymin' %variable]
        c[variable+'_monrmax_diff'] = c[variable+'_%i' %y] - c['%s_Monmax' %variable]
        c[variable+'_monrmin_diff'] = c[variable+'_%i' %y] - c['%s_Monmin' %variable]
        c[variable+'_absrmax_diff'] = c[variable+'_%i' %y] - absmax
        c[variable+'_absrmin_diff'] = c[variable+'_%i' %y] - absmin

        
        dates = pd.date_range(pd.to_datetime('01-01-%i' %y, format='%d-%m-%Y'), pd.to_datetime('31-12-%i' %y, format='%d-%m-%Y'), freq='1D')
        dates = dates[~((dates.day ==29) & (dates.month == 2))]
        c = c.reset_index(drop=True).set_index(dates)
        if y == years[0]:
            c1 = pd.concat([c1,c])
        else:
            c1 = pd.concat([c1,c], axis=0)

    return c1

def plot_records_count(records_df, variable, database, station_name, filename, freq='day'):

    if freq not in ['day', 'month', 'year']:
        raise ValueError("'freq' must be 'day', 'month' or 'year'. Your 'freq' is '%s'" %freq)

    var = {}
    title_var = {}

    var['day'] = 'day'
    var['month'] = 'mon'
    var['year'] = 'abs'

    title_var['day'] = 'Daily'
    title_var['month'] = 'Monthly'
    title_var['year'] = 'Annual'

    records_df = records_df[['%s_%srmax' %(variable,var[freq]), '%s_%srmin' %(variable, var[freq])]].groupby(records_df.index.year).count()
    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(10,7))
    # plot the same data on both axes
    records_df.plot(kind='line', ax=ax1, color={"%s_%srmax" %(variable, var[freq]): "red", "%s_%srmin" %(variable, var[freq]): 'blue'}, markeredgecolor='black', markersize=80)
    
    ax1.grid(color='black', alpha=0.3)
    ax1.set_ylabel('Frequency (days)', fontsize=15)
    ax1.set_xlabel('')
    plt.legend(['Number of days below previous low record', 'Number of days above previous high record'], fontsize=14)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(records_df.index.min(),records_df.index.max()), fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
    plt.tight_layout()
    fig.suptitle('%s records for %s in %s' %(title_var[freq], variable, station_name), fontsize=19, y=0.95)
    fig.savefig(filename,dpi=300)



def compute_and_plot_exceedances(df, variable, database, station_name, filename, threshold=0, time_scale='year', upwards=True, plot_means=False, averaging_period=5):
    '''
    This function allows to compute the number of times a threshold value has been exceeded.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    variable: str
        Variable to be analysed
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    threshold: int or float
        Threshold value
    time_scale: str
        Time scale to which aggregate the number of exceedances
    upwards: str
        If True, computes exceedances above the threshold value. If False, computes exceedances below the threshold value
    '''

    if time_scale.lower() == 'year':
        if upwards == True:
            exceedances=pd.DataFrame(df[variable].gt(threshold))
            exceedances['Year'] = exceedances.index.year

            exceedances = exceedances.groupby('Year').sum(numeric_only=True).reset_index()

        else:
            exceedances=pd.DataFrame(df[variable].lt(threshold))
            exceedances['Year'] = exceedances.index.year

            exceedances = exceedances.groupby('Year').sum(numeric_only=True).reset_index()


        fig, ax = plt.subplots(figsize=(15,7))
        ax.bar(exceedances.Year, exceedances[variable], color="#00ad5c")
        if plot_means == True:
            if len(exceedances) % averaging_period == 1:
                for i in range(0,len(exceedances)-1,averaging_period):
                    if i == 0:
                        ax.hlines(exceedances.iloc[i:min(i+averaging_period, len(exceedances)),variable].mean(), xmin=exceedances.Year[i], xmax=exceedances.Year[min(i+averaging_period, len(exceedances)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                    else:
                        ax.hlines(exceedances.iloc[i:min(i+averaging_period, len(exceedances)),variable].mean(), xmin=exceedances.Year[i], xmax=exceedances.Year[min(i+averaging_period, len(exceedances)-1)], color="#000000", lw=1.2, label='_nolegend_')

            else:
                for i in range(0,len(exceedances),averaging_period):
                    if i == 0:
                        ax.hlines(exceedances.loc[i:min(i+averaging_period, len(exceedances)),variable].mean(), xmin=exceedances.Year[i], xmax=exceedances.Year[min(i+averaging_period, len(exceedances)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                    else:
                        ax.hlines(exceedances.loc[i:min(i+averaging_period, len(exceedances)),variable].mean(), xmin=exceedances.Year[i], xmax=exceedances.Year[min(i+averaging_period, len(exceedances)-1)], color="#000000", lw=1.2, label='_nolegend_')

        ax.grid(color='grey')
        ax.tick_params(axis='both', labelsize=15)
        if plot_means == True:
            ax.legend(fontsize=14,ncol=1)
        ax.set_ylabel('exceedances', fontsize=15)
        plt.text(0.03, 0.935, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.965, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.93, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.subplots_adjust(bottom=0.05,left=0.07, right=0.98, hspace=0.1, wspace=0.1)

        fig.suptitle('Annual exceedances of %s=%1.f' %(variable,threshold), fontsize=24, y=0.97)
        fig.savefig(filename,dpi=300)



    elif time_scale.lower() == 'season':
        if upwards == True:
            exceedances=pd.DataFrame(df[variable].gt(threshold))
            # Group to season
            month_to_season_lu = np.array([
                None,
                'DJF', 'DJF',
                'MAM', 'MAM', 'MAM',
                'JJA', 'JJA', 'JJA',
                'SON', 'SON', 'SON',
                'DJF'
            ])
            grp_ary = month_to_season_lu[df.index.month]
            exceedances['season'] = grp_ary
            exceedances['Year'] = exceedances.index.year

            exceedances = exceedances.groupby(['season','Year']).sum(numeric_only=True).reset_index()

        else:
            exceedances=pd.DataFrame(df[variable].lt(threshold))
            # Group to season
            month_to_season_lu = np.array([
                None,
                'DJF', 'DJF',
                'MAM', 'MAM', 'MAM',
                'JJA', 'JJA', 'JJA',
                'SON', 'SON', 'SON',
                'DJF'
            ])
            grp_ary = month_to_season_lu[df.index.month]
            exceedances['season'] = grp_ary
            exceedances['Year'] = exceedances.index.year

            exceedances=exceedances.groupby(['season', 'Year']).sum(numeric_only=True).reset_index()
    
        season_names = ['December-January-February', 'March-April-May', 'June-July-August', 'September-October-November']# ['DJF', 'MAM', 'JJA', 'SON']
        fig, ax = plt.subplots(figsize=(15,10), ncols=2, nrows=2, sharex=True)
        ax= ax.flatten()
        for j in range(4):
            df_plot = exceedances[exceedances.season == ['DJF', 'MAM', 'JJA', 'SON'][j]].reset_index(drop=True)
            ax[j].bar(df_plot.Year, df_plot[variable], color="#00ad5c")
            if plot_means == True:
                if len(df_plot) % averaging_period == 1:
                    for i in range(0,len(df_plot)-1,averaging_period):
                        if i == 0:
                            ax[j].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                        else:
                            ax[j].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='_nolegend_')

                else:
                    for i in range(0,len(df_plot),averaging_period):
                        if i == 0:
                            ax[j].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                        else:
                            ax[j].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='_nolegend_')
            ax[j].grid(color='grey')
            ax[j].set_title(season_names[j], fontsize=17)
            ax[j].tick_params(axis='both', labelsize=15)
        
        if plot_means == True:
            ax[3].legend(fontsize=14,ncol=1)
        ax[0].set_ylabel('exceedances', fontsize=15)           
        ax[2].set_ylabel('exceedances', fontsize=15)           


        plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.subplots_adjust(bottom=0.07, left=0.07, right=0.98, hspace=0.12, wspace=0.12)

        fig.suptitle('Seasonal exceedances of %s=%1.f' %(variable,threshold), fontsize=24)
        fig.savefig(filename,dpi=300)


    elif time_scale.lower() == 'month':
        if upwards == True:
            exceedances=pd.DataFrame(df[variable].gt(threshold))
            exceedances['Year'] = exceedances.index.year
            exceedances['Month'] = exceedances.index.month

            exceedances = exceedances.groupby(['Month', 'Year']).sum(numeric_only=True).reset_index()

        else:
            exceedances=pd.DataFrame(df[variable].lt(threshold))
            exceedances['Year'] = exceedances.index.year
            exceedances['Month'] = exceedances.index.month

            exceedances = exceedances.groupby(['Month', 'Year']).sum(numeric_only=True).reset_index()


        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        fig, ax = plt.subplots(figsize=(16.5,11.5), ncols=3, nrows=4, sharex=True, sharey=True)
        ax= ax.flatten()
        for j in range(1,13):
            df_plot = exceedances[exceedances.Month == j].reset_index(drop=True)
            ax[j-1].bar(df_plot.Year, df_plot[variable], color="#00ad5c")
            if plot_means == True:
                if len(df_plot) % averaging_period == 1:
                    for i in range(0,len(df_plot)-1,averaging_period):
                        if i == 0:
                            ax[j-1].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                        else:
                            ax[j-1].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='_nolegend_')

                else:
                    for i in range(0,len(df_plot),averaging_period):
                        if i == 0:
                            ax[j-1].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='%i-period mean' %averaging_period)
                        else:
                            ax[j-1].hlines(df_plot.loc[i:min(i+averaging_period, len(df_plot)),variable].mean(), xmin=df_plot.Year[i], xmax=df_plot.Year[min(i+averaging_period, len(df_plot)-1)], color="#000000", lw=1.2, label='_nolegend_')
            ax[j-1].grid(color='grey')
            ax[j-1].set_title(month_names[j-1], fontsize=17)
            ax[j-1].tick_params(axis='both', labelsize=15)
        if plot_means == True:
            ax[11].legend(fontsize=14,ncol=1)

        ax[0].set_ylabel('exceedances', fontsize=16)
        ax[3].set_ylabel('exceedances', fontsize=16)
        ax[6].set_ylabel('exceedances', fontsize=16)
        ax[9].set_ylabel('exceedances', fontsize=16)

        
        plt.text(0.03, 0.94, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.96, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.94, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.subplots_adjust(bottom=0.03, top=0.91, left=0.07, right=0.98, hspace=0.15, wspace=0.13)

        fig.suptitle('Monthly exceedances of %s=%1.f' %(variable,threshold), fontsize=24)
        fig.savefig(filename,dpi=300)
       
def plot_variable_trends(df, var, units, filename, database, station_name, averaging_period=5, grouping='year', grouping_stat='mean', rain_limit=1, plot_kind='line'):
    '''
    This function allows to plot and compare the annual meteogram of a certain year (year_to_plot) with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Name of the variable to be plotted
    units: str
        Units of the variable to be plotted
    station_name: str
        Name of the location of the data
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    plot_kind: str
        Desired plot type (line or bar)
    '''

    if units == 'days':
        # Set a limit on rainfall for computing rainy days
        df['Rainfall'] = (df['Rainfall'] >= rain_limit).astype(int)

    if grouping.lower() == 'year':
        df = df.groupby(pd.Grouper(freq='AS')).apply(grouping_stat, numeric_only=True)

        fig, ax = plt.subplots(figsize=(15,7), sharex=True)
        if plot_kind == 'bar':
            ax.bar(df.index, df[var], color="#008b4a", edgecolor='k', width=100, label=var)
        else:
            ax.plot(df.index, df[var], color="#008b4a", lw=1.4, label=var)

        #if len(df) % averaging_period == 1:
        for i in range(0,len(df)-1,averaging_period):
            if i == 0:
                ax.hlines(df.iloc[i:min(i+averaging_period, len(df)),:][var].mean(), xmin=df.index[i], xmax=df.index[min(i+averaging_period, len(df)-1)], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
            else:
                ax.hlines(df.iloc[i:min(i+averaging_period, len(df)),:][var].mean(), xmin=df.index[i], xmax=df.index[min(i+averaging_period, len(df)-1)], color="#000000", lw=1.3, label='_nolegend_')

        #else:
        #    for i in range(0,len(df),averaging_period):
        #        if i == 0:
        #            ax.hlines(df.iloc[i:i+averaging_period,:][var].mean(), xmin=df.index[i], xmax=df.index[i+averaging_period], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
        #        else:
        #            ax.hlines(df.iloc[i:i+averaging_period,:][var].mean(), xmin=df.index[i], xmax=df.index[i+averaging_period], color="#000000", lw=1.3, label='_nolegend_')

        text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.185),
                        bbox_transform=ax.transAxes, prop={'size': 12}, frameon=True)
        text.patch.set_alpha(0.5)
        plt.text(0.031, 0.955, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.78, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.78, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.suptitle('%s evolution and %i-%s mean' %(var, averaging_period, grouping), fontsize=22, y=0.97)
        plt.subplots_adjust(left=0.07, right=0.98, hspace=0.1, wspace=0.1,bottom=0.05, top=0.9)
        #fig.autofmt_xdate()

        ax.tick_params(labelsize=16)
        ax.set_ylabel(units, fontsize=16)
        ax.grid(color='grey')
        ax.legend(fontsize=14)
        ax.set_xlim(df.index.min()-dt.timedelta(days=3), df.index.max()+dt.timedelta(days=3))

        fig.savefig(filename,dpi=300)

    elif grouping.lower() == 'season':
        # Group to season
        month_to_season_lu = np.array([
            None,
            'DJF', 'DJF',
            'MAM', 'MAM', 'MAM',
            'JJA', 'JJA', 'JJA',
            'SON', 'SON', 'SON',
            'DJF'
        ])
        grp_ary = month_to_season_lu[df.index.month]
        df['season'] = grp_ary 
        df_winter = df[df.season == 'DJF'].groupby(['Year']).apply(grouping_stat, numeric_only=True)
        df_spring = df[df.season == 'MAM'].groupby(['Year']).apply(grouping_stat, numeric_only=True)
        df_summer = df[df.season == 'JJA'].groupby(['Year']).apply(grouping_stat, numeric_only=True)
        df_autumn = df[df.season == 'SON'].groupby(['Year']).apply(grouping_stat, numeric_only=True)

        df_list = [df_winter, df_spring, df_summer, df_autumn]


        fig, ax = plt.subplots(figsize=(15.75,7.75), ncols=2, nrows=2, sharex=True)
        ax= ax.flatten()
        for j in range(4):
            if plot_kind == 'bar':
                ax[j].bar(df_list[j][var].index, df_list[j][var].values, color="#008b4a", edgecolor='k', width=90, label=var)
            else:
                ax[j].plot(df_list[j][var].index, df_list[j][var].values, color="#008b4a", lw=1.4, label=var)
            #if len(df_list[j]) % averaging_period == 1:
            for i in range(0,len(df_list[j])-1,averaging_period):
                if i == 0:
                    ax[j].hlines(df_list[j].iloc[i:min(i+averaging_period, len(df_list[j])),:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[min(i+averaging_period, len(df_list[j])-1)], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
                else:
                    ax[j].hlines(df_list[j].iloc[i:min(i+averaging_period, len(df_list[j])),:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[min(i+averaging_period, len(df_list[j])-1)], color="#000000", lw=1.3, label='_nolegend_')

            #else:
            #    for i in range(0,len(df_list[j]),averaging_period):
            #        if i == 0:
            #            ax[j].hlines(df_list[j].iloc[i:i+averaging_period,:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[i+averaging_period], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
            #        else:
            #            ax[j].hlines(df_list[j].iloc[i:i+averaging_period,:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[i+averaging_period], color="#000000", lw=1.3, label='_nolegend_')

            ax[j].set_title(['December-January-February', 'March-April-May', 'June-July-August', 'September-October-November'][j], fontsize=16)
            if j in [0,2]:
                ax[j].set_ylabel(units, fontsize=16)
            ax[j].tick_params(labelsize=16)
            ax[j].grid(color='grey')
        plt.text(0.03, 0.965, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=13, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.965, 'Database: %s' %database, fontsize=13, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.935, 'Location: %s' %station_name, fontsize=13, transform=plt.gcf().transFigure)

        plt.suptitle('%s evolution and %i-%s mean' %(var, averaging_period, grouping), fontsize=22, y=0.98)
        #plt.subplots_adjust(left=0.05, right=0.95, hspace=0.15, wspace=0.15,bottom=0.05, top=0.9)
        plt.subplots_adjust(bottom=0.05, top=0.9, left=0.07, right=0.98, hspace=0.13, wspace=0.13)

        ax[0].legend(fontsize=14)
        fig.savefig(filename,dpi=300)


    elif grouping.lower() == 'month':
        df_list = {}
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        for i in range(1,13):
            df_list[i] = df[df.index.month == i].groupby(['Year']).apply(grouping_stat, numeric_only=True)


        fig, ax = plt.subplots(figsize=(16.25,10.25), ncols=3, nrows=4, sharex=True)
        ax= ax.flatten()
        for j in range(1,13):
            if plot_kind == 'bar':
                ax[j-1].bar(df_list[j][var].index, df_list[j][var].values, color="#008b4a", edgecolor='k', width=28, label=var)
            else:
                ax[j-1].plot(df_list[j][var].index, df_list[j][var].values, color="#008b4a", lw=1.4, label=var)
            #if len(df_list[j]) % averaging_period == 1:
            for i in range(0,len(df_list[j])-1,averaging_period):
                if i == 0:
                    ax[j-1].hlines(df_list[j].iloc[i:min(i+averaging_period, len(df_list[j])),:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[min(i+averaging_period, len(df_list[j])-1)], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
                else:
                    ax[j-1].hlines(df_list[j].iloc[i:min(i+averaging_period, len(df_list[j])),:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[min(i+averaging_period, len(df_list[j])-1)], color="#000000", lw=1.3, label='_nolegend_')

            #else:
            #    for i in range(0,len(df_list[j]),averaging_period):
            #        if i == 0:
            #            ax[j-1].hlines(df_list[j].iloc[i:i+averaging_period,:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[i+averaging_period], color="#000000", lw=1.3, label='%i-period mean' %averaging_period)
            #        else:
            #            ax[j-1].hlines(df_list[j].iloc[i:i+averaging_period,:][var].mean(), xmin=df_list[j][var].index[i], xmax=df_list[j][var].index[i+averaging_period], color="#000000", lw=1.3, label='_nolegend_')

            ax[j-1].set_title(month_names[j-1], fontsize=18)
            ax[j-1].tick_params(labelsize=16)
            ax[j-1].grid(color='grey')

            if j in [1,4,7,10]:
                ax[j-1].set_ylabel(units, fontsize=16)

        plt.text(0.03, 0.935, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.965, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.935, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)

        plt.suptitle('%s evolution [%s] and %i-%s mean' %(var, units, averaging_period, grouping), fontsize=20, y=0.97)
        plt.subplots_adjust(left=0.05, right=0.975, hspace=0.15, wspace=0.125,bottom=0.04, top=0.89)

        ax[0].legend(fontsize=14)
        fig.savefig(filename,dpi=300)


def plot_data_vs_climate(df,df_climate,variable,units,inidate,enddate,colormap,database,climate_normal_period, station_name, filename, kind='line', climate_stat='median', fillcolor_gradient=False, use_std=True, show_bands=True, show_seasons=True):
    '''
    This function allows to plot climatological data against the climatological mean or median of one variable.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    variable: str
        Variable to be plotted
    units: str
        Units of the variable to be plotted
    inidate: datetime.datetime
        First date to be plotted
    enddate: datetime.datetime
        Last date to be plotted
    colormap: matplotlib colormap
        Colormap to be used in plotting
    database: str
        Name of the database from which the plotted data comes
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    kind: str
        Kind of plot (line or bar)
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    fillcolor_gradient: boolean
        Parameter that controls the way the colormap is employed. If true, the colormap is continue
    use_std : boolean
        If True, use the std of the data for shading. Else, it uses the 10th and 90th percentiles. Only works if show_bands=True
    show_bands: boolean
        If True, shows bands representing the standard deviation or the range of percentiles 10th to 90th
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''

    if climate_stat.lower() not in ['median', 'mean']:
        raise ValueError('"climate_stat" should be equal to "median" or "mean"')
    locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)

    fig, ax = plt.subplots(figsize=(15,7))
    if kind == 'line':
        ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],color="#47fe60",label='Climate %s' %climate_stat)

        ax.plot(df.loc[inidate:enddate, variable].index, df.loc[inidate:enddate, variable],color='black')
        ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.5,label="10%-90%")
        ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_p095' %variable],df_climate.loc[inidate:enddate,'%s_p005' %variable],color='grey',alpha=0.25,label="5%-95%")
        if fillcolor_gradient == False:
            ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],where=df.loc[inidate:enddate, variable] >= df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],facecolor='red', interpolate=True)
            ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable,climate_stat)],where=df.loc[inidate:enddate, variable] <= df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],facecolor='blue', interpolate=True)
        else:
            fill_between_colormap(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index, df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)], cmap=colormap,alpha=0.9)

        if show_bands == True:
            if use_std == False:
                ax.plot(df_climate.loc[inidate:enddate, '%s_p010' %variable].index,df_climate.loc[inidate:enddate,'%s_p010' %variable], linestyle='-', lw=1, color='grey',label="_nolegend")
                ax.plot(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable], linestyle='-', lw=1, color='grey', label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %variable],
                                df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %variable],color='grey',alpha=0.3,label="+-1std")


    elif kind == 'bar':
        ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],color='k',label='Climate %s' %climate_stat)
        diff_var = df.loc[inidate:enddate, variable] - df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]
        mask1 = diff_var < 0
        mask2 = diff_var >= 0
        if len(mask1[mask1 == True]) > 0:
            ax.bar(df.loc[inidate:enddate, '%s' %(variable)].index[mask1], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
        if len(mask2[mask2 == True]) > 0:
            ax.bar(df.loc[inidate:enddate, '%s' %(variable)].index[mask2], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)

        if show_bands == True:
            if use_std == False:
                ax.plot(df_climate.loc[inidate:enddate, '%s_p010' %variable].index,df_climate.loc[inidate:enddate,'%s_p010' %variable], linestyle='-', lw=1, color='grey',label="_nolegend")
                ax.plot(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable], linestyle='-', lw=1, color='grey', label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %variable],
                                df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %variable],color='grey',alpha=0.3,label="+-1std")


    else:
        raise ValueError('kind must be "line" or "bar".')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s (%s)' %(variable, units), fontsize=17)
    ax.set_title('Daily %s during period: %s to %s' %(variable, df.loc[inidate:enddate].index[0].strftime('%d-%m-%Y'), df.loc[inidate:enddate].index[-1].strftime('%d-%m-%Y')),fontsize=16)
    ax.legend(loc='upper left').set_visible(True)
    ax.tick_params(labelsize=14)
    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.095),
                           bbox_transform=ax.transAxes, prop={'size': 12}, frameon=True)
    text.patch.set_alpha(0.5)
    ax.add_artist(text)
    if type(df.loc[inidate:enddate, variable].idxmin()) != float:
        plt.text(0.65, 0.0425, 'Period min.: %.2f%s' %(df.loc[inidate:enddate, variable].min(), units) +' ['+str(df.loc[inidate:enddate, variable].idxmin().strftime('%d-%m-%Y'))+']', fontsize=13, transform=plt.gcf().transFigure)
    if type(df.loc[inidate:enddate, variable].idxmax()) != float:
        plt.text(0.15, 0.0425, 'Period max.: %.2f%s' %(df.loc[inidate:enddate, variable].max(), units) +' ['+str(df.loc[inidate:enddate, variable].idxmax().strftime('%d-%m-%Y'))+']', fontsize=13, transform=plt.gcf().transFigure)
#    fig.autofmt_xdate()
    ax.set_xlim([inidate - dt.timedelta(days=1), enddate + dt.timedelta(days=1)])

    # Show seasons
    if show_seasons == True:
        season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
        for y in df.index.year.unique():
            ax.axvspan(mdates.date2num(dt.datetime(int(y),1,1)), mdates.date2num(dt.datetime(int(y),3,1)), color="#4696db", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),3,1)), mdates.date2num(dt.datetime(int(y),6,1)), color="#32a852", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),6,1)), mdates.date2num(dt.datetime(int(y),9,1)), color="#da5757", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),9,1)), mdates.date2num(dt.datetime(int(y),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),12,1)), mdates.date2num(dt.datetime(int(y)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)

    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig(filename,dpi=300)

    
def plot_data_vs_climate_withrecords(df,df_climate,records_df,variable,units,inidate,enddate,colormap,database,climate_normal_period, station_name, filename, kind='line', climate_stat='median', fillcolor_gradient=False, use_std=True, show_bands = True, show_seasons=True):
    '''
    This function allows to plot climatological data against the climatological mean or median of one variable, and shows when a record value has been broken.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    records_df: DataFrame
        DataFrame containing the records values
    variable: str
        Variable to be plotted
    units: str
        Units of the variable to be plotted
    inidate: datetime.datetime
        First date to be plotted
    enddate: datetime.datetime
        Last date to be plotted
    colormap: matplotlib colormap
        Colormap to be used in plotting
    database: str
        Name of the database from which the plotted data comes
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    kind: str
        Kind of plot (line or bar)
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    fillcolor_gradient: boolean
        Parameter that controls the way the colormap is employed. If true, the colormap is continue
    use_std : boolean
        If True, use the std of the data for shading. Else, it uses the 10th and 90th percentiles. Only works if show_bands=True
    show_bands: boolean
        If True, shows bands representing the standard deviation or the range of percentiles 10th to 90th
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''

    if climate_stat.lower() not in ['median', 'mean']:
        raise ValueError('"climate_stat" should be equal to "median" or "mean"')
    locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    fig, ax = plt.subplots(figsize=(15,7)) 
    if kind == 'line':
        ax.plot(df.loc[inidate:enddate, variable].index, df.loc[inidate:enddate, variable],color='black')
        median, = ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],color="#000000",label='Climate %s' %climate_stat)
        #ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.4,label="10%-90%")
        #ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_p095' %variable],df_climate.loc[inidate:enddate,'%s_p005' %variable],color='grey',alpha=0.25,label="5%-95%")
        if fillcolor_gradient == False:
            ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],where=df.loc[inidate:enddate, variable] >= df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],facecolor='red', interpolate=True, alpha=0.9)
            ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],where=df.loc[inidate:enddate, variable] <= df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],facecolor='blue', interpolate=True, alpha=0.9)
        else:
            fill_between_colormap(df_climate.loc[inidate:enddate, variable].index, df.loc[inidate:enddate, variable],df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)], cmap=colormap,alpha=0.9)

        if show_bands == True:
            if use_std == False:
                ax.plot(df_climate.loc[inidate:enddate, '%s_p010' %variable].index,df_climate.loc[inidate:enddate,'%s_p010' %variable], linestyle='-', lw=1, color='grey',label="_nolegend")
                ax.plot(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable], linestyle='-', lw=1, color='grey', label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %variable],
                                df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %variable],color='grey',alpha=0.3,label="+-1std")


    elif kind == 'bar':
        median,= ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)],color='k',label='Climate %s' %climate_stat)
        diff_var = df.loc[inidate:enddate, variable] - df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]
        #print(diff_var)
        mask1 = diff_var < 0
        mask2 = diff_var >= 0
        if len(mask1[mask1 == True]) > 0:
            ax.bar(df.loc[inidate:enddate, variable].index[mask1], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
        if len(mask2[mask2 == True]) > 0:
            ax.bar(df.loc[inidate:enddate, variable].index[mask2], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)

        if show_bands == True:
            if use_std == False:
                ax.plot(df_climate.loc[inidate:enddate, '%s_p010' %variable].index,df_climate.loc[inidate:enddate,'%s_p010' %variable], linestyle='-', lw=1, color='grey',label="_nolegend")
                ax.plot(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable], linestyle='-', lw=1, color='grey', label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_p090' %variable].index,df_climate.loc[inidate:enddate,'%s_p090' %variable],df_climate.loc[inidate:enddate,'%s_p010' %variable],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax.plot(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(variable)],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax.fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(variable, climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %variable],
                                df_climate.loc[inidate:enddate,'%s_%s' %(variable, climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %variable],color='grey',alpha=0.3,label="+-1std")

    else:
        raise ValueError('kind must be "line" or "bar".')
    
    # Daily records
    records_df = records_df.loc[inidate:enddate,:]
    drmax = ax.scatter(records_df[records_df['%s_monrmax' %variable].isnull()].index, records_df[records_df['%s_monrmax' %variable].isnull()]['%s_dayrmax' %(variable)], color='red', marker='o', edgecolor='black', label='Days with high record: %i' %records_df['%s_dayrmax' %(variable)].count())
    drmin = ax.scatter(records_df[records_df['%s_monrmin' %variable].isnull()].index, records_df[records_df['%s_monrmin' %variable].isnull()]['%s_dayrmin' %(variable)], color='blue', marker='o', edgecolor='black', label='Days with low record: %i' %records_df['%s_dayrmin' %(variable)].count())
    # Monthly records
    mrmax = ax.scatter(records_df[records_df['%s_absrmax' %variable].isnull()].index, records_df[records_df['%s_absrmax' %variable].isnull()]['%s_monrmax' %(variable)], color='red', marker='^', s=60, edgecolor='black', label='Days above monthly high record: %i' %records_df['%s_monrmax' %(variable)].count())
    mrmin = ax.scatter(records_df[records_df['%s_absrmin' %variable].isnull()].index, records_df[records_df['%s_absrmin' %variable].isnull()]['%s_monrmin' %(variable)], color='blue', marker='^', s=60, edgecolor='black', label='Days below monthly low record: %i' %records_df['%s_monrmin' %(variable)].count())
    # Absolute records
    absrmax = ax.scatter(records_df.index, records_df['%s_absrmax' %(variable)], color='red', marker='*', s=70, edgecolor='black', label='Days above absolute high record: %i' %records_df['%s_absrmax' %(variable)].count())
    absrmin = ax.scatter(records_df.index, records_df['%s_absrmin' %(variable)], color='blue', marker='*', s=70, edgecolor='black', label='Days below absolute low record: %i' %records_df['%s_absrmin' %(variable)].count())

    
    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s (%s)' %(variable, units), fontsize=17)
    ax.set_title('Daily %s during period: %s to %s' %(variable, df.loc[inidate:enddate].index[0].strftime('%d-%m-%Y'), df.loc[inidate:enddate].index[-1].strftime('%d-%m-%Y')),fontsize=16)
    ax.legend(loc='upper left',ncol=3).set_visible(True)
    ax.tick_params(labelsize=14)
    if type(df.loc[inidate:enddate, variable].idxmin()) != float:
        plt.text(0.65, 0.0425, 'Period min.: %.2f%s' %(df.loc[inidate:enddate, variable].min(), units) +' ['+str(df.loc[inidate:enddate, variable].idxmin().strftime('%d-%m-%Y'))+']', fontsize=13, transform=plt.gcf().transFigure)
    if type(df.loc[inidate:enddate, variable].idxmax()) != float:
        plt.text(0.15, 0.0425, 'Period max.: %.2f%s' %(df.loc[inidate:enddate, variable].max(), units) +' ['+str(df.loc[inidate:enddate, variable].idxmax().strftime('%d-%m-%Y'))+']', fontsize=13, transform=plt.gcf().transFigure)

    # Legends
    # Create a legend for the median.
    if show_bands == True:
        first_legend = ax.legend(handles=[median, std], loc='lower right', ncol=2)
    else:
        first_legend = ax.legend(handles=[median], loc='lower right')

    # Add the legend manually to the Axes.
    ax.add_artist(first_legend)

    # Create another legend for the records.
    ax.legend(handles=[drmax, drmin, mrmax, mrmin, absrmax, absrmin], loc='upper center', ncol=3)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim([inidate - dt.timedelta(days=1), enddate + dt.timedelta(days=1)])
    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.095),
                           bbox_transform=ax.transAxes, prop={'size': 12}, frameon=True)
    text.patch.set_alpha(0.5)
    ax.add_artist(text)

    # Show seasons
    if show_seasons == True:
        season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
        for y in df.index.year.unique():
            ax.axvspan(mdates.date2num(dt.datetime(int(y),1,1)), mdates.date2num(dt.datetime(int(y),3,1)), color="#4696db", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),3,1)), mdates.date2num(dt.datetime(int(y),6,1)), color="#32a852", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),6,1)), mdates.date2num(dt.datetime(int(y),9,1)), color="#da5757", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),9,1)), mdates.date2num(dt.datetime(int(y),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
            ax.axvspan(mdates.date2num(dt.datetime(int(y),12,1)), mdates.date2num(dt.datetime(int(y)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)

    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)

    plt.savefig(filename,dpi=300)


def plot_data_vs_climate_withrecords_multivar(df,df_climate,records_df,vars_list,units_list,inidate,enddate,colormap,database,climate_normal_period, station_name, filename, kind='line', climate_stat='median', fillcolor_gradient=False, use_std=True, show_bands=True, show_seasons=True):
    '''
    This function allows to plot climatological data of up to two variables against the climatological mean or median, and shows when a record value has been broken.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    records_df: DataFrame
        DataFrame containing the records values
    vars_list: str
        Variable(s) to be plotted
    units: str
        Units of the variable(s) to be plotted
    inidate: datetime.datetime
        First date to be plotted
    enddate: datetime.datetime
        Last date to be plotted
    colormap: matplotlib colormap
        Colormap to be used in plotting
    database: str
        Name of the database from which the plotted data comes
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    kind: str
        Kind of plot (line or bar)
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    fillcolor_gradient: boolean
        Parameter that controls the way the colormap is employed. If true, the colormap is continue
    use_std : boolean
        If True, use the std of the data for shading. Else, it uses the 10th and 90th percentiles. Only works if show_bands=True
    show_bands: boolean
        If True, shows bands representing the standard deviation or the range of percentiles 10th to 90th
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''

    if len(vars_list) < 2:
        raise ValueError('len(vars_list) must be greater or equal than 2. Your vars_list has a length of: %i' %len(vars_list))
    elif len(vars_list) > 2:
        warnings.warn('Your vars_list has a length of %i, but only the two first elements will be read.' %len(vars_list))

    if climate_stat.lower() not in ['median', 'mean']:
        raise ValueError('"climate_stat" should be equal to "median" or "mean"')


    records_df = records_df.loc[inidate:enddate,:]
    
    locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(15,7),sharex=True)
    ax = axs.flatten()
    if kind == 'line':
        ax[0].plot(df.loc[inidate:enddate, vars_list[0]].index, df.loc[inidate:enddate, vars_list[0]],color="#ffffff", lw=1.1)
        median, = ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],color="#000000",label='Climate %s' %climate_stat)
        if fillcolor_gradient == False:
            ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df.loc[inidate:enddate, vars_list[0]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],where=df.loc[inidate:enddate, vars_list[0]] >= df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],facecolor='red', interpolate=True, alpha=0.9)
            ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df.loc[inidate:enddate, vars_list[0]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],where=df.loc[inidate:enddate, vars_list[0]] <= df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],facecolor='blue', interpolate=True, alpha=0.9)
        else:
            fill_between_colormap(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index, df_climate.loc[inidate:enddate, vars_list[0]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)], cmap=colormap,alpha=0.9)

        if show_bands == True:
            if use_std == False:
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_p010' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p010' %vars_list[0]], linestyle='-', lw=1, color='grey',label="_nolegend")
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_p090' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p090' %vars_list[0]], linestyle='-', lw=1, color='grey', label="_nolegend_")
                std = ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_p090' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p090' %vars_list[0]],df_climate.loc[inidate:enddate,'%s_p010' %vars_list[0]],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(vars_list[0])],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(vars_list[0])],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %vars_list[0]],
                                df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %vars_list[0]],color='grey',alpha=0.3,label="+-1std")

        # Daily records
        drmax = ax[0].scatter(records_df[records_df['%s_monrmax' %(vars_list[0])].isnull()].index, records_df[records_df['%s_monrmax' %(vars_list[0])].isnull()]['%s_dayrmax' %(vars_list[0])], color='red', marker='o', s=20, edgecolor='black', label='Days with high record: %i' %records_df['%s_dayrmax' %(vars_list[0])].count(), zorder=100)
        drmin = ax[0].scatter(records_df[records_df['%s_monrmin' %(vars_list[0])].isnull()].index, records_df[records_df['%s_monrmin' %(vars_list[0])].isnull()]['%s_dayrmin' %(vars_list[0])], color='blue', marker='o', s=20, edgecolor='black', label='Days with low record: %i' %records_df['%s_dayrmin' %(vars_list[0])].count(), zorder=100)
        # Monthly records
        mrmax = ax[0].scatter(records_df[records_df['%s_absrmax' %(vars_list[0])].isnull()].index, records_df[records_df['%s_absrmax' %(vars_list[0])].isnull()]['%s_monrmax' %(vars_list[0])], color='red', marker='^', s=30, edgecolor='black', label='Days above monthly high record: %i' %records_df['%s_monrmax' %(vars_list[0])].count(), zorder=101)
        mrmin = ax[0].scatter(records_df[records_df['%s_absrmin' %(vars_list[0])].isnull()].index, records_df[records_df['%s_absrmin' %(vars_list[0])].isnull()]['%s_monrmin' %(vars_list[0])], color='blue', marker='^', s=30, edgecolor='black', label='Days below monthly low record: %i' %records_df['%s_monrmin' %(vars_list[0])].count(), zorder=101)
        # Absolute records
        absrmax = ax[0].scatter(records_df.index, records_df['%s_absrmax' %(vars_list[0])], color='red', marker='*', s=50, edgecolor='black', label='Days above absolute high record: %i' %records_df['%s_absrmax' %(vars_list[0])].count(), zorder=102)
        absrmin = ax[0].scatter(records_df.index, records_df['%s_absrmin' %(vars_list[0])], color='blue', marker='*', s=50, edgecolor='black', label='Days below absolute low record: %i' %records_df['%s_absrmin' %(vars_list[0])].count(), zorder=102)

        ax[0].grid(color='black',alpha=0.5)
        ax[0].set_ylabel('%s (%s)' %(vars_list[0], units_list[0]), fontsize=17)

        # Second variable
        if vars_list[1] != 'Rainfall':
            ax[1].plot(df.loc[inidate:enddate, vars_list[1]].index, df.loc[inidate:enddate, vars_list[1]],color='black')
            median1, = ax[1].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],color="#47fe60",label='Climate %s' %climate_stat)
            ax[1].fill_between(df_climate.loc[inidate:enddate, '%s_p090' %(vars_list[1])].index,df_climate.loc[inidate:enddate,'%s_p090' %vars_list[1]],df_climate.loc[inidate:enddate,'%s_p010' %vars_list[1]],color='grey',alpha=0.5,label="10%-90%")
            #ax[1].fill_between(df_climate.loc[inidate:enddate, '%s_p095'].index,df_climate.loc[inidate:enddate,'%s_p095' %vars_list[1]],df_climate.loc[inidate:enddate,'%s_p005' %vars_list[1]],color='grey',alpha=0.25,label="5%-95%")
            if fillcolor_gradient == False:
                ax[1].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index,df.loc[inidate:enddate, vars_list[1]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],where=df.loc[inidate:enddate, vars_list[1]] >= df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],facecolor='red', interpolate=True)
                ax[1].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index,df.loc[inidate:enddate, vars_list[1]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],where=df.loc[inidate:enddate, vars_list[1]] <= df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],facecolor='blue', interpolate=True)
            else:
                fill_between_colormap(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index, df_climate.loc[inidate:enddate, vars_list[1]],df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)], cmap=colormap,alpha=0.9)

        else:
            diff_var1 = df.loc[inidate:enddate, vars_list[1]] - df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)]
            mask1 = diff_var1 > 0
            mask2 = diff_var1 < 0
            mask3 = diff_var1 == 0
            if len(mask1[mask1 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask1], df.loc[inidate:enddate, vars_list[1]][mask1], color='#32a852', edgecolor='black',linewidth=1)
            if len(mask2[mask2 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask2], df.loc[inidate:enddate, vars_list[1]][mask2], color='#996100', edgecolor='black',linewidth=1)
            if len(mask3[mask3 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask3], df.loc[inidate:enddate, vars_list[1]][mask3], color='black', edgecolor='black',linewidth=1)
               
        # Daily records
        drmax1 = ax[1].scatter(records_df[records_df['%s_monrmax' %(vars_list[1])].isnull()].index, records_df[records_df['%s_monrmax' %(vars_list[1])].isnull()]['%s_dayrmax' %(vars_list[1])], color='red', marker='o', s=20, edgecolor='black', label='Days with high record: %i' %records_df['%s_dayrmax' %(vars_list[1])].count(), zorder=100)
        drmin1 = ax[1].scatter(records_df[records_df['%s_monrmin' %(vars_list[1])].isnull()].index, records_df[records_df['%s_monrmin' %(vars_list[1])].isnull()]['%s_dayrmin' %(vars_list[1])], color='blue', marker='o', s=20, edgecolor='black', label='Days with low record: %i' %records_df['%s_dayrmin' %(vars_list[1])].count(), zorder=100)
        # Monthly records
        mrmax1 = ax[1].scatter(records_df[records_df['%s_absrmax' %(vars_list[1])].isnull()].index, records_df[records_df['%s_absrmax' %(vars_list[1])].isnull()]['%s_monrmax' %(vars_list[1])], color='red', marker='^', s=30, edgecolor='black', label='Days above monthly high record: %i' %records_df['%s_monrmax' %(vars_list[1])].count(), zorder=101)
        mrmin1 = ax[1].scatter(records_df[records_df['%s_absrmin' %(vars_list[1])].isnull()].index, records_df[records_df['%s_absrmin' %(vars_list[1])].isnull()]['%s_monrmin' %(vars_list[1])], color='blue', marker='^', s=30, edgecolor='black', label='Days below monthly low record: %i' %records_df['%s_monrmin' %(vars_list[1])].count(), zorder=101) 
        # Absolute records
        absrmax1 = ax[1].scatter(records_df.index, records_df['%s_absrmax' %(vars_list[1])], color='red', marker='*', s=50, edgecolor='black', label='Days above absolute high record: %i' %records_df['%s_absrmax' %(vars_list[1])].count(), zorder=102)
        absrmin1 = ax[1].scatter(records_df.index, records_df['%s_absrmin' %(vars_list[1])], color='blue', marker='*', s=50, edgecolor='black', label='Days below absolute low record: %i' %records_df['%s_absrmin' %(vars_list[1])].count(), zorder=102)

        ax[1].grid(color='black',alpha=0.5)
        ax[1].set_ylabel('%s (%s)' %(vars_list[1], units_list[1]), fontsize=17)


    elif kind == 'bar':
        median, = ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)],color="#000000",label='Climate %s' %climate_stat)
        diff_var = df.loc[inidate:enddate, vars_list[0]] - df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)]
        #print(diff_var)
        mask1 = diff_var < 0
        mask2 = diff_var >= 0

        if len(mask1[mask1 == True]) > 0:
            ax[0].bar(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index[mask1], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
        if len(mask2[mask2 == True]) > 0:
            ax[0].bar(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index[mask2], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)
        
        if show_bands == True:
            if use_std == False:
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_p010' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p010' %vars_list[0]],linestyle='-',lw=1,color='grey',label="_nolegend")
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_p090' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p090' %vars_list[0]],linestyle='-',lw=1,color='grey', label="_nolegend_")
                std = ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_p090' %vars_list[0]].index,df_climate.loc[inidate:enddate,'%s_p090' %vars_list[0]],df_climate.loc[inidate:enddate,'%s_p010' %vars_list[0]],color='grey',alpha=0.3,label="10%-90%")
            else:
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %(vars_list[0])],linestyle='-', lw=1, color='grey',label="_nolegend_")
                ax[0].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)] - df_climate.loc[inidate:enddate, '%s_std' %(vars_list[0])],linestyle='-', lw=1, color='grey',label="_nolegend_")
                std = ax[0].fill_between(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[0], climate_stat)].index,df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)] + df_climate.loc[inidate:enddate, '%s_std' %vars_list[0]],
                                df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[0], climate_stat)]  - df_climate.loc[inidate:enddate, '%s_std' %vars_list[0]],color='grey',alpha=0.3,label="+-1std")

        # Daily records
        drmax = ax[0].scatter(records_df[records_df['%s_monrmax' %(vars_list[0])].isnull()].index, records_df[records_df['%s_monrmax' %(vars_list[0])].isnull()]['%s_dayrmax' %(vars_list[0])], color='red', marker='o', s=20, edgecolor='black', label='Days with high record: %i' %records_df['%s_dayrmax' %(vars_list[0])].count())
        drmin = ax[0].scatter(records_df[records_df['%s_monrmin' %(vars_list[0])].isnull()].index, records_df[records_df['%s_monrmin' %(vars_list[0])].isnull()]['%s_dayrmin' %(vars_list[0])], color='blue', marker='o', s=20, edgecolor='black', label='Days with low record: %i' %records_df['%s_dayrmin' %(vars_list[0])].count())
        # Monthly records
        mrmax = ax[0].scatter(records_df[records_df['%s_absrmax' %(vars_list[0])].isnull()].index, records_df[records_df['%s_absrmax' %(vars_list[0])].isnull()]['%s_monrmax' %(vars_list[0])], color='red', marker='^', s=30, edgecolor='black', label='Days above monthly high record: %i' %records_df['%s_monrmax' %(vars_list[0])].count())
        mrmin = ax[0].scatter(records_df[records_df['%s_absrmin' %(vars_list[0])].isnull()].index, records_df[records_df['%s_absrmin' %(vars_list[0])].isnull()]['%s_monrmin' %(vars_list[0])], color='blue', marker='^', s=30, edgecolor='black', label='Days below monthly low record: %i' %records_df['%s_monrmin' %(vars_list[0])].count())
        # Absolute records
        absrmax = ax[0].scatter(records_df.index, records_df['%s_absrmax' %(vars_list[0])], color='red', marker='*', s=50, edgecolor='black', label='Days above absolute high record: %i' %records_df['%s_absrmax' %(vars_list[0])].count())
        absrmin = ax[0].scatter(records_df.index, records_df['%s_absrmin' %(vars_list[0])], color='blue', marker='*', s=50, edgecolor='black', label='Days below absolute low record: %i' %records_df['%s_absrmin' %(vars_list[0])].count())

        
        ax[0].grid(color='black',alpha=0.5)
        ax[0].set_ylabel('%s (%s)' %(vars_list[0], units_list[0]), fontsize=17)
        
        if vars_list[1] != 'Rainfall':
            median1, = ax[1].plot(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index, df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],color="#47fe60",label='Climate %s' %climate_stat)
#            median1, = ax[1].plot(df_climate.loc[inidate:enddate, vars_list[1]].index, df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)],color="#47fe60",label='Climate %s' %climate_stat)
            diff_var = df.loc[inidate:enddate, vars_list[1]] - df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)]
            #print(diff_var)
            mask1 = diff_var < 0
            mask2 = diff_var >= 0
            if len(mask1[mask1 == True]) > 0:
                ax[1].bar(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index[mask1], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
            if len(mask2[mask2 == True]) > 0:
                ax[1].bar(df_climate.loc[inidate:enddate, '%s_%s' %(vars_list[1], climate_stat)].index[mask2], bottom=df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)
                        
        else:
            diff_var1 = df.loc[inidate:enddate, vars_list[1]] - df_climate.loc[inidate:enddate,'%s_%s' %(vars_list[1], climate_stat)]
            mask1 = diff_var1 > 0
            mask2 = diff_var1 < 0
            mask3 = diff_var1 == 0
    
            if len(mask1[mask1 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask1], df.loc[inidate:enddate, vars_list[1]][mask1], color='#32a852', edgecolor='black',linewidth=0.8)
            if len(mask2[mask2 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask2], df.loc[inidate:enddate, vars_list[1]][mask2], color='#996100', edgecolor='black',linewidth=0.8)
            if len(mask3[mask3 == True]) > 0:
                ax[1].bar(df.loc[inidate:enddate, vars_list[1]].index[mask3], df.loc[inidate:enddate, vars_list[1]][mask3], color='black', edgecolor='black',linewidth=0.8)
        
        # Daily records
        drmax1 = ax[1].scatter(records_df[records_df['%s_monrmax' %(vars_list[1])].isnull()].index, records_df[records_df['%s_monrmax' %(vars_list[1])].isnull()]['%s_dayrmax' %(vars_list[1])], color='red', marker='o', s=20, edgecolor='black', label='Days with high record: %i' %records_df['%s_dayrmax' %(vars_list[1])].count())
        drmin1 = ax[1].scatter(records_df[records_df['%s_monrmin' %(vars_list[1])].isnull()].index, records_df[records_df['%s_monrmin' %(vars_list[1])].isnull()]['%s_dayrmin' %(vars_list[1])], color='blue', marker='o', s=20, edgecolor='black', label='Days with low record: %i' %records_df['%s_dayrmin' %(vars_list[1])].count())
        # Monthly records
        mrmax1 = ax[1].scatter(records_df[records_df['%s_absrmax' %(vars_list[1])].isnull()].index, records_df[records_df['%s_absrmax' %(vars_list[1])].isnull()]['%s_monrmax' %(vars_list[1])], color='red', marker='^', s=30, edgecolor='black', label='Days above monthly high record: %i' %records_df['%s_monrmax' %(vars_list[1])].count())
        mrmin1 = ax[1].scatter(records_df[records_df['%s_absrmin' %(vars_list[1])].isnull()].index, records_df[records_df['%s_absrmin' %(vars_list[1])].isnull()]['%s_monrmin' %(vars_list[1])], color='blue', marker='^', s=30, edgecolor='black', label='Days below monthly low record: %i' %records_df['%s_monrmin' %(vars_list[1])].count())       
        # Absolute records
        absrmax1 = ax[1].scatter(records_df.index, records_df['%s_absrmax' %(vars_list[1])], color='red', marker='*', s=50, edgecolor='black', label='Days above absolute high record: %i' %records_df['%s_absrmax' %(vars_list[1])].count())
        absrmin1 = ax[1].scatter(records_df.index, records_df['%s_absrmin' %(vars_list[1])], color='blue', marker='*', s=50, edgecolor='black', label='Days below absolute low record: %i' %records_df['%s_absrmin' %(vars_list[1])].count())


        ax[1].grid(color='black',alpha=0.5)
        ax[1].set_ylabel('%s (%s)' %(vars_list[1], units_list[1]), fontsize=17)

    else:
        raise ValueError('kind must be "line" or "bar".')

    fig.suptitle(x=0.5,y=0.95, t='%s and %s during period %s to %s' %(vars_list[0], vars_list[1], df.loc[inidate:enddate].index[0].strftime('%d-%b-%Y'), df.loc[inidate:enddate].index[-1].strftime('%d-%b-%Y')),fontsize=16)
    
    # Legends
    # Create a legend for the median.
    if show_bands == True:
        first_legend = ax[0].legend(handles=[median, std], loc='lower right', ncol=2)
    else:
        first_legend = ax[0].legend(handles=[median], loc='lower right')


    # Add the legend manually to the Axes.
    ax[0].add_artist(first_legend)

    # Create another legend for the records.
    ax[0].legend(handles=[drmax, drmin, mrmax, mrmin, absrmax, absrmin], loc='upper center', ncol=3)
    
    #ax[0].legend(loc='upper center',ncol=4).set_visible(True)
    if vars_list[1] != 'Rainfall':
        # Create a legend for the median.
        first_legend = ax[1].legend(handles=[median1], loc='lower right')
        # Add the legend manually to the Axes.
        ax[1].add_artist(first_legend)

        # Create another legend for the records.
        ax[1].legend(handles=[drmax1, drmin1, mrmax1, mrmin1, absrmax1, absrmin1], loc='upper center', ncol=3)    
    else:
        ax[1].legend(loc='upper center', ncol=3).set_visible(True)

    ax[0].tick_params(labelsize=14)
    ax[1].tick_params(labelsize=14)
    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.185),
                           bbox_transform=ax[0].transAxes, prop={'size': 12}, frameon=True)
    text.patch.set_alpha(0.5)

    ax[0].add_artist(text)   
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(formatter)
    ax[0].set_xlim([inidate - dt.timedelta(days=1), enddate + dt.timedelta(days=1)])
    ax[1].set_xlim([inidate - dt.timedelta(days=1), enddate + dt.timedelta(days=1)])

    # Show seasons
    if show_seasons == True:
        season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
        for y in df.index.year.unique():
            for i in range(2):
                ax[i].axvspan(mdates.date2num(dt.datetime(int(y),1,1)), mdates.date2num(dt.datetime(int(y),3,1)), color="#4696db", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(y),3,1)), mdates.date2num(dt.datetime(int(y),6,1)), color="#32a852", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(y),6,1)), mdates.date2num(dt.datetime(int(y),9,1)), color="#da5757", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(y),9,1)), mdates.date2num(dt.datetime(int(y),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(y),12,1)), mdates.date2num(dt.datetime(int(y)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)


    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    plt.subplots_adjust(hspace=0.1, bottom=0.06, left=0.1, right=0.9)
    fig.savefig(filename,dpi=300)

    

def plot_periodaverages(df, df_climate, var, units, inidate, enddate, station_name, database, plotdir, kind='line', stat='median', window=10, use_std=False):
    '''
    This function allows to plot a timeseries of the mean (or median) value of a variable within a defined period. It also plots a moving average with a period defined by the parameter "window".

    Parameters
    ----------
    df : DataFrame
        DataFrame with all data of the days to be analysed
    df_climate : DataFrame
        DataFrame with annual averages of all data of the days to be analysed
    var : str
        Variable to plot
    units : str
        Units of the variable to plot
    inidate: datetime.datetime
        First date to be selected (only day and month are used, unless month of enddate is lower than month of inidate)
    enddate: datetime.datetime
        Last date to be selected (only day and month are used, unless month of enddate is lower than month of inidate)
    station_name : str
        Location of the data
    database : str
        Source of the data
    plotdir : str
        Plotting directory
    stat : str
        Statistic to apply to the data
    window : int
        Window of the moving average
    use_std : boolean
        If True, use the std of the data for shading. Else, it uses the 5th, 10th, 90th and 95th percentiles

    '''
    if 'Year' not in df.columns:
        df['Year'] = df.index.year
    if 'Year' not in df_climate.columns:
        df_climate['Year'] = df_climate.index.year


    ### Visualizador de datos de un periodo concreto
    df['to_select_dates'] = 100*df.index.month + df.index.day

    first_selected_day = df.index[(df.index.day == inidate.day) & 
                                                 (df.index.month == inidate.month)]

    last_selected_day = df.index[(df.index.day == enddate.day) & 
                                                 (df.index.month == enddate.month)]
    selected_dates = []
    for i in range(len(last_selected_day)):
        selected_dates.append(pd.date_range(first_selected_day[i],last_selected_day[i]))
        
    selected_dates = np.unique(selected_dates)

    # Los datos de los días seleccionados
    df = df[df.index.isin(selected_dates)]
    df_climate = df_climate[df_climate.index.isin(selected_dates)]

    df = df.groupby('Year').apply(stat, numeric_only=True)
    df_climate = df_climate.groupby('Year').apply(stat, numeric_only=True)
    fig, ax = plt.subplots(figsize=(15,7))
    df[var].rolling(window=window, center=False).mean().plot(kind='line', ax=ax, color='black', lw=1.2, zorder=10, label='%i-year MA' %window)

    if kind=='line':
        df[var].plot(kind=kind, ax=ax, color='pink', markeredgecolor='black', markersize=80,zorder=1, label='_nolegend_')
        ax.scatter(x=df.index, y =df[var], color='red', edgecolor='black', s=80, clip_on = False, zorder=1000)
        ax.scatter(x=df.index[-1], y=df.loc[df.index[-1],var], color='orange', edgecolor='black', s=80, clip_on = False, zorder=1000, label='%i: %.1fºC' %(df.index.max(),df.loc[df.index.max(),var]))
        ax.plot(df.index, np.tile(df[var].median(),len(df)),color="#47fe60",label='Median: %.1f%s' %(df[var].median(), units), zorder=2)
#        ax.plot(df_climate.index, np.tile(df_climate[var+'_%s' %stat].median(),len(df_climate)),color="#47fe60",label='Median: %.1f%s' %(df_climate[var+'_%s' %stat].median(), units), zorder=2)
        if use_std == False:
            ax.fill_between(df.index,np.tile(df.quantile(0.90)[var], len(df)),np.tile(df.quantile(0.10)[var], len(df)),color='grey',alpha=0.45,label="10%-90%", zorder=0)
            ax.fill_between(df.index,np.tile(df.quantile(0.95)[var], len(df)),np.tile(df.quantile(0.05)[var], len(df)),color='grey',alpha=0.25,label="5%-95%", zorder=0)
        else:
            ax.fill_between(df.index,np.tile(df.quantile(0.90)[var], len(df)),np.tile(df.quantile(0.10)[var], len(df)),color='grey',alpha=0.45,label="+-1std", zorder=0)

    else:
        mask1 = df[var] >= 0
        mask2 = df[var] < 0

        if len(mask1[mask1 == True]) > 0:
            ax.bar(df.index[mask1], df[var][mask1], color="#ff4444", edgecolor='black',linewidth=0.8)
        if len(mask2[mask2 == True]) > 0:
            ax.bar(df.index[mask2], df[var][mask2], color="#0E6FFF", edgecolor='black',linewidth=0.8)

    ax.plot(df.index, np.tile(df[var].min(),len(df)),color='blue',linestyle='dashed',label='Record min.: %.1f%s' %(df[var].min(), units) + ' [%.0f' %df[var].idxmin()+']')
    ax.plot(df.index, np.tile(df[var].max(),len(df)),color='red',linestyle='dashed',label='Record max.: %.1f%s' %(df[var].max(), units) + ' [%.0f' %df[var].idxmax()+']')
    ax.grid(color='black',alpha=0.5)
    ax.set_xticks(np.arange(df.index.min(),df.index.max()+1,5))
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('',fontsize=12)
    ax.set_ylabel('%s (%s)' %(var,units),fontsize=15)
    ax.set_title('%s %s. Reference period: %s to %s' 
                  %(stat,var,dt.datetime.strftime(inidate,format='%d-%b'),
                    dt.datetime.strftime(enddate,format='%d-%b')),fontsize=18)
    plt.legend(bbox_to_anchor=(1.0, -0.10),ncol=4,fontsize=12).set_visible(True)
    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.25, 0.135),
                           bbox_transform=ax.transAxes, prop={'size': 13}, frameon=True)
    text.patch.set_alpha(0.5)

    ax.add_artist(text)
    fig.autofmt_xdate()
    plt.text(0.825, 0.935, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.905, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)

    ax.set_xlim([df.index.min()-0.5,df.index.max()+0.5])
    fig.savefig(plotdir+'/%saverageofperiod.png' %(var),dpi=300)


#def plot_movingaverages(df,df_climate):
    


def plot_data_and_accum_anoms(df,df_climate,year_to_plot,vars_list,units_list,colormap,database,climate_normal_period, station_name, plotdir, climate_stat='median', secondplot_type='accum', w=7, show_seasons=True):
    '''
    This function allows to plot climatological data of one variable against the climatological mean or median, and their cumulative or moving mean.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    year_to_plot: int
        Year of the data to be plotted
    vars_list: str
        Variable(s) to be plotted
    units_list: str
        Units of the variable(s) to be plotted
    inidate: datetime.datetime
        First date to be plotted
    enddate: datetime.datetime
        Last date to be plotted
    colormap: matplotlib colormap
        Colormap to be used in plotting
    database: str
        Name of the database from which the plotted data comes
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    plotdir: str
        Absolute path where the plot is going to be saved
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    secondplot_type: str
        Type of the second plot (accum or moving)
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''


    if len(vars_list) < 1:
        raise ValueError('len(vars_list) must be greater or equal than 1. Your vars_list has a length of: %i' %len(vars_list))
    
    if secondplot_type not in ['accum','moving']:
        raise ValueError('"anom_type" must be either "accum" or "moving".')

    df = df[df.index.year == year_to_plot]
    df_climate = df_climate[df_climate.index.year == year_to_plot]

    if df.index.month[-1] != 12 or df.index.month[-1] != 31:
        df = df.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))
    
    if df_climate.index.month[-1] != 12 or df_climate.index.month[-1] != 31:
        df_climate = df_climate.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))


    locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for i in range(len(vars_list)):
        fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(15,7),sharex=True)
        ax = axs.flatten()
        median, = ax[0].plot(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index, df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)],color='k',label='Climate %s' %climate_stat)
        diff_var = df.loc[df.index.year == year_to_plot, vars_list[i]] - df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)]
        mask1 = diff_var < 0
        mask2 = diff_var >= 0
        ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index[mask1], bottom=df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
        ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index[mask2], bottom=df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)
            

        ax[0].grid(color='black',alpha=0.5)
        ax[0].set_ylabel('%s (%s)' %(vars_list[i], units_list[i]), fontsize=17)

        # Accumulated anoms or MA anoms
        anoms_df = df.loc[df.index.year == year_to_plot, vars_list[i]] - df_climate.loc[df_climate.index.year == year_to_plot, '%s_%s' %(vars_list[i], climate_stat)]

        if secondplot_type == 'accum':
            accum_anom = anoms_df.cumsum()
            ax[1].plot(accum_anom,color='black', label='Accum. anomaly')
            ax[1].grid(color='black',alpha=0.5)
            ax[1].axhline(y=0, linestyle='--')
            ax[1].set_ylabel('%s (%s) accum. anomaly' %(vars_list[i], units_list[i]), fontsize=17)

            fig.suptitle(x=0.5,y=0.94, t='%s and accumulated anomaly during period %s to %s' %(vars_list[i], dt.datetime(year_to_plot,1,1).strftime('%d-%b-%Y'), dt.datetime(year_to_plot,12,31).strftime('%d-%b-%Y')),fontsize=16)

        else:
            ma_anom = anoms_df.rolling(window=w).mean()
            ax[1].plot(ma_anom,color='black', label='%i-period MA of anomaly' %w)
            ax[1].grid(color='black',alpha=0.5)
            ax[1].axhline(y=0, linestyle='--')

            ax[1].set_ylabel('%s (%s) %i-period MA anom.' %(vars_list[i], units_list[i], w), fontsize=17)

            fig.suptitle(x=0.5,y=0.94, t='%s and %i-period MA of anomaly during period %s to %s' %(vars_list[i], w, dt.datetime(year_to_plot,1,1).strftime('%d-%b-%Y'), dt.datetime(year_to_plot,12,31).strftime('%d-%b-%Y')),fontsize=16)

        
        # Legends
        ax[0].legend(loc='upper left',ncol=1).set_visible(True)
        ax[1].legend(loc='upper left', ncol=1).set_visible(True)
        ax[0].tick_params(labelsize=14)
        ax[1].tick_params(labelsize=14)
        text = AnchoredText('Alejandro Rodríguez Sánchez',
                            loc=1,bbox_to_anchor=(0.24, 0.185),
                            bbox_transform=ax[0].transAxes, prop={'size': 12}, frameon=True)
        text.patch.set_alpha(0.5)

        ax[0].add_artist(text)   
        ax[0].xaxis.set_major_locator(locator)
        ax[0].xaxis.set_major_formatter(formatter)
        ax[0].set_xlim([dt.datetime(year_to_plot,1,1) - dt.timedelta(days=1), dt.datetime(year_to_plot,12,31) + dt.timedelta(days=1)])

        # Show seasons
        if show_seasons == True:
            season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
            for i in range(2):
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),1,1)), mdates.date2num(dt.datetime(int(year_to_plot),3,1)), color="#4696db", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),3,1)), mdates.date2num(dt.datetime(int(year_to_plot),6,1)), color="#32a852", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),6,1)), mdates.date2num(dt.datetime(int(year_to_plot),9,1)), color="#da5757", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),9,1)), mdates.date2num(dt.datetime(int(year_to_plot),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),12,1)), mdates.date2num(dt.datetime(int(year_to_plot)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)


        plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
        #plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
        plt.subplots_adjust(hspace=0.1)
        fig.savefig(plotdir+'/%s_and_%sanom_climate%s%i%i.png' %(vars_list[i], secondplot_type, climate_stat, climate_normal_period[0], climate_normal_period[1]),dpi=300)




def plot_data_and_yearly_cycle(df,df_climate,year_to_plot,vars_list,units_list,colormap,database, climate_normal_period, station_name, plotdir, climate_stat='median', fillcolor_gradient=False, show_seasons=True):
    '''
    This function allows to plot climatological data of one variable against the climatological mean or median, and their yearly cycle.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    year_to_plot: int
        Year of the data to be plotted
    vars_list: str
        Variable(s) to be plotted
    units_list: str
        Units of the variable(s) to be plotted
    inidate: datetime.datetime
        First date to be plotted
    enddate: datetime.datetime
        Last date to be plotted
    colormap: matplotlib colormap
        Colormap to be used in plotting
    database: str
        Name of the database from which the plotted data comes
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    plotdir: str
        Absolute path where the plot is going to be saved
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    fillcolor_gradient: boolean
        Parameter that controls the way the colormap is employed. If true, the colormap is continue
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''


    if len(vars_list) < 1:
        raise ValueError('len(vars_list) must be greater or equal than 1. Your vars_list has a length of: %i' %len(vars_list))
    
    df = df[df.index.year == year_to_plot]
    df_climate = df_climate[df_climate.index.year == year_to_plot]

    if df.index.month[-1] != 12 or df.index.month[-1] != 31:
        df = df.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))
    
    if df_climate.index.month[-1] != 12 or df_climate.index.month[-1] != 31:
        df_climate = df_climate.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))

    locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for i in range(len(vars_list)):
        fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(15,7),sharex=True)
        ax = axs.flatten()
        if units_list[i] not in ['mm', 'in', 'cm']:
            median, = ax[0].plot(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index, df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)],color='k',label='Climate %s' %climate_stat)
            diff_var = df.loc[df.index.year == year_to_plot, vars_list[i]] - df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)]
            #print(diff_var)
            mask1 = diff_var < 0
            mask2 = diff_var >= 0

            ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index[mask1], bottom=df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)][mask1], height = diff_var[mask1], color = colormap([-1000]), alpha=0.7)
            ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)].index[mask2], bottom=df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)][mask2], height = diff_var[mask2], color = colormap([1000]), alpha=0.7)
                

            ax[0].grid(color='black',alpha=0.5)
            ax[0].set_ylabel('%s (%s)' %(vars_list[i], units_list[i]), fontsize=17)

            # Accumulated mean
            accum_mean = pd.DataFrame(df.loc[df.index.year == year_to_plot, vars_list[i]].cumsum())
            accum_mean[vars_list[i]+'_nocount'] = df.loc[df.index.year == year_to_plot, vars_list[i]]
            accum_mean['count'] = 0
            n = 1
            for h in range(len(accum_mean)):
                if np.isnan(accum_mean.iloc[h,1]) == False:
                    accum_mean.iloc[h,2] = n
                    n += 1
                else:
                    accum_mean.iloc[h,2] = n

            accum_mean['cummean'] = accum_mean[vars_list[i]]/ accum_mean['count']

            accum_mean_climate = pd.DataFrame(df_climate.loc[df_climate.index.year == year_to_plot, vars_list[i]+'_%s' %climate_stat].cumsum())
            accum_mean_climate[vars_list[i]+'_nocount'] = df_climate.loc[df_climate.index.year == year_to_plot, vars_list[i]+'_%s' %climate_stat]
            accum_mean_climate['count'] = 0
            nc = 1
            for h in range(len(accum_mean_climate)):
                if np.isnan(accum_mean_climate.iloc[h,1]) == False:
                    accum_mean_climate.iloc[h,2] = nc
                    nc += 1
                else:
                    accum_mean_climate.iloc[h,2] = nc

            accum_mean_climate['cummean climate'] = accum_mean_climate[vars_list[i]+'_%s' %climate_stat]/ accum_mean_climate['count']


            accum_mean_all = pd.concat([accum_mean['cummean'], accum_mean_climate['cummean climate']], axis=1)
            accum_mean_all.columns = [vars_list[i], vars_list[i]+' climate']
            accum_mean_all['anomaly'] = accum_mean_all[vars_list[i]] - accum_mean_all[vars_list[i]+' climate']


            ax[1].plot(accum_mean_all[vars_list[i]],color='black', label='Accum. mean %i' %year_to_plot)
            ax[1].plot(accum_mean_all[vars_list[i]+' climate'],color='black', ls='--', label='Accum. mean climate')

            if fillcolor_gradient == False:
                ax[1].fill_between(accum_mean_all.index,accum_mean_all.loc[:, vars_list[i]],accum_mean_all.loc[:,'%s climate' %vars_list[i]],where=accum_mean_all.loc[:, vars_list[i]] >= accum_mean_all.loc[:,'%s climate' %vars_list[i]],facecolor='red', interpolate=True)
                ax[1].fill_between(accum_mean_all.index,accum_mean_all.loc[:, vars_list[i]],accum_mean_all.loc[:,'%s climate' %vars_list[i]],where=accum_mean_all.loc[:, vars_list[i]] <= accum_mean_all.loc[:,'%s climate' %vars_list[i]],facecolor='blue', interpolate=True)
            else:
                cmap, norm = fill_between_colormap(accum_mean_all.loc[:, vars_list[i]].index, accum_mean_all.loc[:, vars_list[i]],accum_mean_all.loc[:,'%s climate' %vars_list[i]], cmap=accum_mean_all,alpha=0.9)
                # Add color bar
                cax = ax[1].inset_axes([1.025, 0, 0.025, 0.95])
                cbar=plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax) #, shrink=.98,ticks=levels, pad=0.02)
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_title(units_list[i],fontsize=14)


            ax[1].grid(color='black',alpha=0.5)
            ax[1].set_ylabel('%s (%s) accum. mean' %(vars_list[i], units_list[i]), fontsize=17)

            # Indicate last accumulated anomaly:
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax[1].text(x=0.7, y=0.05, s='Last accumulated anomaly: %.1f %s' %(accum_mean_all.dropna(subset=['anomaly'])['anomaly'][-1], units_list[i]), transform=ax[1].transAxes, fontsize=14, bbox=props)

            fig.suptitle(x=0.5,y=0.94, t='Daily %s and accumulated mean during period %s to %s' %(vars_list[i], dt.datetime(year_to_plot,1,1).strftime('%d-%b-%Y'), dt.datetime(year_to_plot,12,31).strftime('%d-%b-%Y')),fontsize=16)

        else:
            diff_var1 = df.loc[df.index.year == year_to_plot, vars_list[i]] - df_climate.loc[df_climate.index.year == year_to_plot,'%s_%s' %(vars_list[i], climate_stat)]
            mask1 = diff_var1 > 0
            mask2 = diff_var1 < 0
            mask3 = diff_var1 = 0
    
            ax[0].bar(df.loc[df.index.year == year_to_plot, vars_list[i]].index[mask1], df.loc[df.index.year == year_to_plot, vars_list[i]][mask1], color='#32a852', edgecolor='black',linewidth=0.8)
            ax[0].bar(df.loc[df.index.year == year_to_plot, vars_list[i]].index[mask2], df.loc[df.index.year == year_to_plot, vars_list[i]][mask2], color='#996100', edgecolor='black',linewidth=0.8)
            ax[0].bar(df.loc[df.index.year == year_to_plot, vars_list[i]].index[mask3], df.loc[df.index.year == year_to_plot, vars_list[i]][mask3], color='black', edgecolor='black',linewidth=0.8)

            # Accumulated value
            cum_df = pd.concat([df.loc[df.index.year == year_to_plot, vars_list[i]].cumsum(), df_climate.loc[df_climate.index.year == year_to_plot, vars_list[i]+'_%s' %climate_stat].cumsum()], axis=1)
            cum_df.columns = [vars_list[i], vars_list[i]+' climate']
            cum_df['anomaly'] = cum_df[vars_list[i]] - cum_df[vars_list[i]+' climate']


            ax[1].plot(cum_df[vars_list[i]],color='black', label='Accum. rainfall %i' %year_to_plot)
            ax[1].plot(cum_df[vars_list[i]+' climate'],color='black', label='Accum. rainfall climate')

            if fillcolor_gradient == False:
                ax[1].fill_between(cum_df.index,accum_mean_all.loc[:, vars_list[i]],cum_df.loc[:,'%s climate' %vars_list[i]],where=cum_df.loc[:, vars_list[i]] >= cum_df.loc[:,'%s climate' %vars_list[i]],facecolor='red', interpolate=True)
                ax[1].fill_between(cum_df.index,accum_mean_all.loc[:, vars_list[i]],cum_df.loc[:,'%s climate' %vars_list[i]],where=cum_df.loc[:, vars_list[i]] <= cum_df.loc[:,'%s climate' %vars_list[i]],facecolor='blue', interpolate=True)
            else:
                g=fill_between_colormap(cum_df.loc[:, vars_list[i]].index, cum_df.loc[:, vars_list[i]],cum_df.loc[:,'%s climate' %vars_list[i]], cmap=cum_df,alpha=0.9)
                # axins = ax[1].inset_axes([0.8, 0.01, 0.17, 0.02])
               # cbar = matplotlib.colorbar.ColorbarBase(axins, cmap=cum_df, orientation='horizontal',location='lower right')


            ax[1].grid(color='black',alpha=0.5)
            ax[1].set_ylabel('%s (%s) accum.' %(vars_list[i], units_list[i]), fontsize=17)

            # Indicate last accumulated anomaly:
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax[1].text(x=0.5, y=0.04, s='Last accumulated anomaly: %.1f' %cum_df.dropna(subset=['anomaly'])['anomaly'][-1], transform=ax[1].transAxes, fontsize=14, bbox=props)

            fig.suptitle(x=0.5,y=0.94, t='Timeseries of %s and accumulated mean during period %s to %s' %(vars_list[i], dt.datetime(year_to_plot,1,1).strftime('%d-%b-%Y'), dt.datetime(year_to_plot,12,31).strftime('%d-%b-%Y')),fontsize=16)

        
        # Legends        
        ax[0].legend(loc='upper left',ncol=1).set_visible(True)
        ax[1].legend(loc='upper left', ncol=1).set_visible(True)
        ax[0].tick_params(labelsize=14)
        ax[1].tick_params(labelsize=14)
        text = AnchoredText('Alejandro Rodríguez Sánchez',
                            loc=1,bbox_to_anchor=(0.24, 0.185),
                            bbox_transform=ax[0].transAxes, prop={'size': 12}, frameon=True)
        text.patch.set_alpha(0.5)

        ax[0].add_artist(text)   
        ax[0].xaxis.set_major_locator(locator)
        ax[0].xaxis.set_major_formatter(formatter)
        ax[0].set_xlim([dt.datetime(year_to_plot,1,1) - dt.timedelta(days=1), dt.datetime(year_to_plot,12,31) + dt.timedelta(days=1)])

        # Show seasons
        if show_seasons == True:
            season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
            for i in range(2):
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),1,1)), mdates.date2num(dt.datetime(int(year_to_plot),3,1)), color="#4696db", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),3,1)), mdates.date2num(dt.datetime(int(year_to_plot),6,1)), color="#32a852", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),6,1)), mdates.date2num(dt.datetime(int(year_to_plot),9,1)), color="#da5757", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),9,1)), mdates.date2num(dt.datetime(int(year_to_plot),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
                ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),12,1)), mdates.date2num(dt.datetime(int(year_to_plot)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)


        plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
        #plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
        plt.subplots_adjust(hspace=0.1)
        fig.savefig(plotdir+'/%s_and_yearlycycleaccum_climate%s%i%i.png' %(vars_list[i], climate_stat, climate_normal_period[0], climate_normal_period[1]),dpi=300)


def plot_timeseries(df, df_climate, var, units, climate_normal_period, database, station_name, filename, plot_MA=False, climate_stat='median', window=10):
    '''
    This function allows to plot climatological data of one variable against the climatological mean or median, and their cumulative or moving mean.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Variable to be plotted
    units: str
        Units of the variable to be plotted
    filename: str
        Absolute path where the plot is going to be saved
    colors: list
        List of colors to plot (one for each variable in vars_list)
    plot_MA: boolean
        If True, plots the moving average of that variable with period=window
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)
    window: int
        Length of the moving average period
    '''

    df = df.copy()

    if plot_MA == True:
        df[var] = df[var].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(df.index, df[var],'k-',fillstyle='full',markersize='5',clip_on=False)
    ax.plot(df.index, np.tile(df[var].min(),len(df)),'b--',label='Record min.: %.1fºC [%s]' %(df[var].min(),df[var].idxmin().date()))
    ax.plot(df.index, np.tile(df[var].max(),len(df)),'r--',label='Record max.: %.1fºC [%s]' %(df[var].max(),df[var].idxmax().date()))
    if plot_MA == False:
        ax.plot(df.index, df[var].rolling(window=window).mean(),color='cyan',label='%i-MA' %window)
        ax.plot(df_climate.index, df_climate[var+'_%s' %climate_stat],color='g',label='Climate normal')
        ax.fill_between(df_climate.index,df_climate['%s_p090' %var],df_climate['%s_p010' %var],color='grey',alpha=0.5,label="10%-90%")
        ax.fill_between(df_climate.index,df_climate['%s_p095' %var],df_climate['%s_p005' %var],color='grey',alpha=0.25,label="5%-95%")
    else:
        ax.fill_between(df.index,df[var].mean()+df[var].std(),df[var].mean()-df[var].std(),color='grey',alpha=0.45,label="+-1std")

    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s [%s]' %(var,units),fontsize=16)
    ax.tick_params(axis='both',labelsize=16)
    if plot_MA == False:
        ax.set_title('Timeseries of %s [%i-%i]' %(var,df.index.year.min(),df.index.year.max()),fontsize=20)
    else:
        ax.set_title('Timeseries of %s MA [%i-%i]' %(var,df.index.year.min(),df.index.year.max()),fontsize=20)

    ax.legend(bbox_to_anchor=(1, -0.10),ncol=3,fontsize=12).set_visible(True)
    ax.set_xlim([df.index[0]-dt.timedelta(days=15),df.index[-1]+dt.timedelta(days=15)])   
    fig.autofmt_xdate()
    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.955, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.825, 0.925, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    fig.savefig(filename,dpi=300)

    return None


def timeseries_extremevalues(df, var, units, climate_normal_period, database, station_name, filename, time_scale='Year'):
    '''
    This function allows to plot the timeseries of the: maximum, minimum, p010 and p090 values of several variables.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Variable to be plotted
    units: str
        Units of the variable to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    time_scale: str
        The time scale for which extract the extreme values. It can be: 'Year', 'Month' or 'season'.
    '''

    df = df.copy()

    if time_scale.lower() not in ['year','month', 'season']:
        raise ValueError('"time_scale" must be "Year" or "Month" or "season"')

    # Group to season
    month_to_season_lu = np.array([
        None,
        'DJF', 'DJF',
        'MAM', 'MAM', 'MAM',
        'JJA', 'JJA', 'JJA',
        'SON', 'SON', 'SON',
        'DJF'
    ])
    grp_ary = month_to_season_lu[df.index.month]
    df['season'] = grp_ary 

    # Step 1: Compute desired percentiles
    if time_scale.lower() == 'year':
        df_extremes = pd.DataFrame({'min. value': df.groupby('Year')[var].min(), 
                                                    'p10 value': df.groupby('Year')[var].quantile(0.1),
                                                    'p90 value': df.groupby('Year')[var].quantile(0.9),
                                                    'max. value': df.groupby('Year')[var].max()})
        ##  Broken y-axis figure
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,10))
        
        # Trend lines
        slope_max, intercept_max, r_value_max, pv_max, se_max = stats.linregress(df_extremes.index, df_extremes['max. value'])
        ax1.plot(df_extremes.index, df_extremes['max. value'], '-o', color='r', markersize=10,  label='Annual max.')
        ax1.plot(df_extremes.index, intercept_max + slope_max*df_extremes.index, 'r', label='Trend: %.1f %s/decade' %(10*slope_max, units))
        #sns.regplot(x=df_extremes.index, y="max. value", data=df_extremes, ax=ax1, marker=None, color='red',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
        
        slope_p90, intercept_p90, r_value_p90, pv_p90, se_p90 = stats.linregress(df_extremes.index, df_extremes['p90 value'])
        ax1.plot(df_extremes.index, df_extremes['p90 value'], '-o', color='salmon', markersize=10, label='Annual p90')
        ax1.plot(df_extremes.index, intercept_p90 + slope_p90*df_extremes.index, 'salmon', label='Trend: %.1f %s/decade' %(10*slope_p90, units))
        #sns.regplot(x=df_extremes.index, y="p90 value", data=df_extremes, ax=ax1, marker=None, color='salmon',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
        
        slope_p10, intercept_p10, r_value_p10, pv_p10, se_p10 = stats.linregress(df_extremes.index, df_extremes['p10 value'])
        ax2.plot(df_extremes.index, df_extremes['p10 value'], '-o', color='deepskyblue', markersize=10, label='Annual p10')
        ax2.plot(df_extremes.index, intercept_p10 + slope_p10*df_extremes.index, 'deepskyblue', label='Trend: %.1f %s/decade' %(10*slope_p10, units))
        #sns.regplot(x=df_extremes.index, y="p10 value", data=df_extremes, ax=ax2, marker=None, color='deepskyblue',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
        
        slope_min, intercept_min, r_value_min, pv_min, se_min = stats.linregress(df_extremes.index, df_extremes['min. value'])
        ax2.plot(df_extremes.index, df_extremes['min. value'], '-o', color='blue', markersize=10, label='Annual min.')
        ax2.plot(df_extremes.index, intercept_min + slope_min*df_extremes.index, 'blue', label='Trend: %.1f %s/decade' %(10*slope_min, units))
        #sns.regplot(x=df_extremes.index, y="min. value", data=df_extremes, ax=ax2, marker=None, color='blue',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
        
        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(df_extremes['p90 value'].min()*0.95, df_extremes.max().max()*1.05)  # outliers only
        ax2.set_ylim(min(df_extremes.min().min()*0.95, df_extremes.min().min()*1.05), df_extremes['p10 value'].max()*1.05)  # outliers only
        
        # hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False, labelsize=13)  # don't put tick labels at the top
        ax2.tick_params(labelsize=13)
        ax2.xaxis.tick_bottom()
        ax1.legend(ncol=2,fontsize=13).set_visible(True)
        ax2.legend(ncol=2,fontsize=13).set_visible(True)

        # Slanted lines
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
            
        ## legend
        #ax1.legend(bbox_to_anchor=(0.4, -0.22),ncol=2,fontsize=14).set_visible(True)
        #ax2.legend().set_visible(False)

        fig.text(x=0.02, y=0.25, s='%s [%s]' %(var,units), rotation=90, fontsize=22)

        ax1.tick_params(axis='both', labelsize=18)
        ax2.tick_params(axis='both', labelsize=18)

        ax1.grid(color='k')
        ax2.grid(color='k')

        plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        fig.suptitle('Annual extreme values for %s' %(var), fontsize=24)
        fig.savefig(filename,dpi=300)

    elif time_scale.lower() == 'month':
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_extremes = pd.DataFrame({'min. value': df.groupby(['Month','Year'])[var].min(), 
                                                'p10 value': df.groupby(['Month','Year'])[var].quantile(0.1),
                                                'p90 value': df.groupby(['Month','Year'])[var].quantile(0.9),
                                                'max. value': df.groupby(['Month','Year'])[var].max()})

        df_extremes = df_extremes.reset_index()

        ##  Broken y-axis figure
        fig = plt.figure(figsize=(16,10)) #(8, 3, sharex=True, figsize=(15,10))
        gs = matplotlib.gridspec.GridSpec(4, 3, hspace=0.3,wspace=0.2, figure=fig)
        for n in range(12):
            df_plot = df_extremes[df_extremes.Month == n+1].set_index('Year').drop(columns=['Month'])

            gss = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[n],hspace=0.03)
            ax0 = fig.add_subplot(gss[0])
            ax1 = fig.add_subplot(gss[1], sharex=ax0)
            
            # Trend lines
            slope_max, intercept_max, r_value_max, pv_max, se_max = stats.linregress(df_plot.index, df_plot['max. value'])
            ax0.plot(df_plot.index, df_plot['max. value'], '-o', color='r', markersize=9,  label='Annual max.')
            ax0.plot(df_plot.index, intercept_max + slope_max*df_plot.index, 'r', label='%.1f %s/10y' %(10*slope_max, units))
            #sns.regplot(x=df_extremes.index, y="max. value", data=df_extremes, ax=ax1, marker=None, color='red',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_p90, intercept_p90, r_value_p90, pv_p90, se_p90 = stats.linregress(df_plot.index, df_plot['p90 value'])
            ax0.plot(df_plot.index, df_plot['p90 value'], '-o', color='salmon', markersize=9, label='Annual p90')
            ax0.plot(df_plot.index, intercept_p90 + slope_p90*df_plot.index, 'salmon', label='%.1f %s/10y' %(10*slope_p90, units))
            #sns.regplot(x=df_extremes.index, y="p90 value", data=df_extremes, ax=ax1, marker=None, color='salmon',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_p10, intercept_p10, r_value_p10, pv_p10, se_p10 = stats.linregress(df_plot.index, df_plot['p10 value'])
            ax1.plot(df_plot.index, df_plot['p10 value'], '-o', color='deepskyblue', markersize=9, label='Annual p10')
            ax1.plot(df_plot.index, intercept_p10 + slope_p10*df_plot.index, 'deepskyblue', label='%.1f %s/10y' %(10*slope_p10, units))
            #sns.regplot(x=df_extremes.index, y="p10 value", data=df_extremes, ax=ax2, marker=None, color='deepskyblue',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_min, intercept_min, r_value_min, pv_min, se_min = stats.linregress(df_plot.index, df_plot['min. value'])
            ax1.plot(df_plot.index, df_plot['min. value'], '-o', color='blue', markersize=9, label='Annual min.')
            ax1.plot(df_plot.index, intercept_min + slope_min*df_plot.index, 'blue', label='%.1f %s/10y' %(10*slope_min, units))

            # zoom-in / limit the view to different portions of the data
            # Set y-axes limits
            ax0.set_ylim(df_plot['p90 value'].min()*0.95, df_plot.max().max()*1.05)  # outliers only
            ax1.set_ylim(min(df_plot.min().min()*0.95, df_plot.min().min()*1.05), df_plot['p10 value'].max()*1.05)  # outliers only
            ax0.spines.bottom.set_visible(False)
            ax1.spines.top.set_visible(False)
            ax0.xaxis.tick_top()
            ax1.xaxis.tick_bottom()
            ax0.tick_params(labeltop=False, labelsize=15)  # don't put tick labels at the top
            ax1.tick_params(labeltop=False, labelsize=15)  # don't put tick labels at the top

            if ax1.get_ylim()[1] < ax0.get_ylim()[0]:
                ax1.set_ylim([min(df_plot.min().min()*0.95, df_plot.min().min()*1.05),ax0.get_ylim()[0]])
            else:

                # Slanted lines
                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax0.plot([0, 1], [0, 0], transform=ax0.transAxes, **kwargs)
                ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
                    
            # legend
            if n == 10:
                #ax0.legend(bbox_to_anchor=(0.3, -0.52),ncol=2,fontsize=13).set_visible(True)
                # Put a legend below current axis
                ax1.legend(loc='upper center', bbox_to_anchor=(0.4, -0.25),fancybox=True, shadow=True, ncol=5, fontsize=13)
                ax0.legend().set_visible(False)
            else:
                ax0.legend().set_visible(False)
                ax1.legend().set_visible(False)

            ax0.grid(color='black', alpha=0.3)
            ax0.set_xlabel('')
            ax1.grid(color='black', alpha=0.3)
            ax1.set_xlabel('')

            ax0.set_ylabel('', fontsize=0)
            ax1.set_ylabel('', fontsize=0)

            # Axis title
            ax0.set_title(month_names[n], fontsize=16)

        # Text
        fig.text(x=0.02, y=0.25, s='%s [%s]' %(var,units), rotation=90, fontsize=22)


        plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.07, right=0.98, hspace=0.1, wspace=0.1)

        fig.suptitle('Monthly extreme values for %s' %(var), fontsize=24)
        fig.savefig(filename,dpi=300)


    elif time_scale.lower() == 'season':
        df_extremes = pd.DataFrame({'min. value': df.groupby(['season','Year'])[var].min(), 
                                                'p10 value': df.groupby(['season','Year'])[var].quantile(0.1),
                                                'p90 value': df.groupby(['season','Year'])[var].quantile(0.9),
                                                'max. value': df.groupby(['season','Year'])[var].max()})

        df_extremes = df_extremes.reset_index()

        ##  Broken y-axis figure
        #fig, axs = plt.subplots(8,3, sharex=True, figsize=(15,10))
        season_names = ['DJF', 'MAM', 'JJA', 'SON']
        season_labels = ['December-January-February', 'March-April-May', 'June-July-August', 'September-October-November']
        fig = plt.figure(figsize=(16,10)) #(8, 3, sharex=True, figsize=(15,10))
        gs = matplotlib.gridspec.GridSpec(2, 2, hspace=0.3,wspace=0.2, figure=fig)
        # ax = axs.flatten()
        for n in range(4):
            df_plot = df_extremes[df_extremes.season == season_names[n]].set_index('Year').drop(columns=['season'])

            gss = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[n],hspace=0.03)
            ax0 = fig.add_subplot(gss[0])
            ax1 = fig.add_subplot(gss[1], sharex=ax0)
            
            # Trend lines
            slope_max, intercept_max, r_value_max, pv_max, se_max = stats.linregress(df_plot.index, df_plot['max. value'])
            a,=ax0.plot(df_plot.index, df_plot['max. value'], '-o', color='r', markersize=9,  label='Annual max.')
            a_t,=ax0.plot(df_plot.index, intercept_max + slope_max*df_plot.index, 'r', label='%.1f %s/10y' %(10*slope_max, units))
            #sns.regplot(x=df_extremes.index, y="max. value", data=df_extremes, ax=ax1, marker=None, color='red',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_p90, intercept_p90, r_value_p90, pv_p90, se_p90 = stats.linregress(df_plot.index, df_plot['p90 value'])
            b,=ax0.plot(df_plot.index, df_plot['p90 value'], '-o', color='salmon', markersize=9, label='Annual p90')
            b_t,=ax0.plot(df_plot.index, intercept_p90 + slope_p90*df_plot.index, 'salmon', label='%.1f %s/10y' %(10*slope_p90, units))
            #sns.regplot(x=df_extremes.index, y="p90 value", data=df_extremes, ax=ax1, marker=None, color='salmon',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_p10, intercept_p10, r_value_p10, pv_p10, se_p10 = stats.linregress(df_plot.index, df_plot['p10 value'])
            c,=ax1.plot(df_plot.index, df_plot['p10 value'], '-o', color='deepskyblue', markersize=9, label='Annual p10')
            c_t,=ax1.plot(df_plot.index, intercept_p10 + slope_p10*df_plot.index, 'deepskyblue', label='%.1f %s/10y' %(10*slope_p10, units))
            #sns.regplot(x=df_extremes.index, y="p10 value", data=df_extremes, ax=ax2, marker=None, color='deepskyblue',robust=True, scatter_kws={'s': 80, 'edgecolor': 'black'})
            
            slope_min, intercept_min, r_value_min, pv_min, se_min = stats.linregress(df_plot.index, df_plot['min. value'])
            d,=ax1.plot(df_plot.index, df_plot['min. value'], '-o', color='blue', markersize=9, label='Annual min.')
            d_t,=ax1.plot(df_plot.index, intercept_min + slope_min*df_plot.index, 'blue', label='%.1f %s/10y' %(10*slope_min, units))


            # zoom-in / limit the view to different portions of the data
            # Set y-axes limits
            ax0.set_ylim(df_plot['p90 value'].min()*0.95, df_plot.max().max()*1.05)  # outliers only
            ax1.set_ylim(min(df_plot.min().min()*0.95, df_plot.min().min()*1.05), df_plot['p10 value'].max()*1.05)  # outliers only
            ax0.spines.bottom.set_visible(False)
            ax1.spines.top.set_visible(False)
            ax0.xaxis.tick_top()
            ax1.xaxis.tick_bottom()
            ax0.tick_params(labeltop=False, labelsize=15)  # don't put tick labels at the top
            ax1.tick_params(labeltop=False, labelsize=15)  # don't put tick labels at the top

            if ax1.get_ylim()[1] < ax0.get_ylim()[0]:
                ax1.set_ylim([min(df_plot.min().min()*0.95, df_plot.min().min()*1.05),ax0.get_ylim()[0]])
            else:

                # Slanted lines
                d = .5  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                ax0.plot([0, 1], [0, 0], transform=ax0.transAxes, **kwargs)
                ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
                    
            # Create another legend for the records.
            #second_legend = ax1.legend(handles=[a_t, b_t], bbox_to_anchor=(0.1, 1.05), ncol=2)
            ax1.legend(handles=[a_t, b_t, c_t, d_t], fontsize=13, ncol=2).set_visible(True)#, bbox_to_anchor=(0.99, 1.06), ncol=2, fontsize=13).set_visible(True)
            #ax1.legend(handles=[c_t, d_t], loc='upper right', ncol=2)
            # Add the legend manually to the Axes.
            #fig.add_artist(second_legend)
            #fig.add_artist(third_legend)


            ax0.grid(color='black', alpha=0.3)
            ax0.set_xlabel('')
            ax1.grid(color='black', alpha=0.3)
            ax1.set_xlabel('')

            ax0.set_ylabel('', fontsize=0)
            ax1.set_ylabel('', fontsize=0)

            # Axis title
            ax0.set_title(season_labels[n], fontsize=18)


        plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.07, right=0.98, hspace=0.1, wspace=0.1, bottom=0.05)

        fig.suptitle('Seasonal extreme values for %s' %(var), fontsize=24)
        fig.savefig(filename,dpi=300)


def plot_yearly_cycles(df, variable, units, year_to_plot, climate_normal_period, database, station_name, colors, filename, yearly_cycle=False, criterion='latest'): #, units, climate_normal_period, database, station_name, plotdir, criterion='latest'):
    '''
    This function allows to plot the yearly cycles of one variable for every year included in data

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    variable: str
        Variable to be plotted
    units: str
        Units of the variable to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    colors: list
        List of colors to be used in plotting
    filename: str
        Absolute path where the plot is going to be saved
    yearly_cycle: boolean
        If false, it plots the yearly data as it happened, insted of the yearly cycle
    '''


    if criterion not in ['highest', 'lowest', 'latest']:
        raise ValueError('"criterion" must be "highest" or "lowest" or "latest"')

    # Remove years with not enough data
    for y in df.index.year.unique():
        a = df[df.index.year == y]
        if a[variable].isna().sum() >= 365*0.15 and y != df.index.year.max():
            df = df[df.index.year != y]


    if yearly_cycle == False:
        if criterion == 'highest':
            highlighted_years = df[df.index.year != year_to_plot].groupby('Year').mean(numeric_only=True).sort_values(by=variable, ascending=False)[:3].index
        elif criterion == 'lowest':
            highlighted_years = df[df.index.year != year_to_plot].groupby('Year').mean(numeric_only=True).sort_values(by=variable, ascending=True)[:3].index
        elif criterion == 'latest':
            highlighted_years = df[df.index.year != year_to_plot].groupby('Year').mean(numeric_only=True)[-3:].index
    else:
        if criterion == 'highest':
            highlighted_years = df[(df.index.year != year_to_plot) & (df.index.month == 12) & (df.index.day == 31)].sort_values(by=variable, ascending=False)[:3].index.year
        elif criterion == 'lowest':
            highlighted_years = df[(df.index.year != year_to_plot) & (df.index.month == 12) & (df.index.day == 31)].sort_values(by=variable, ascending=True)[:3].index.year
        elif criterion == 'latest':
            highlighted_years = df[(df.index.year != year_to_plot) & (df.index.month == 12) & (df.index.day == 31)][-3:].index.year
        
    

    # Comparo con 3 años más calidos
    fig, ax = plt.subplots(figsize=(15,7))
    #ax.set_facecolor('#fafcd4')

    # Ploteo añor normales de gris (elimino días bisiestos)
    for ye in df[~df.index.year.isin(highlighted_years)].index.year.unique(): # and not in [year_to_plot]:
        ax.plot(np.arange(0,len(df.loc[(df.index.year == ye) & ~((df.index.day == 29) & (df.index.month == 2))].index),1), df.loc[(df.index.year == ye) & ~((df.index.day == 29) & (df.index.month == 2)), variable],color='grey', alpha=0.2, lw=0.9, zorder=0)
        #ax.plot(np.arange(0,len(df1_complete_join.loc[df1_complete_join.index.year == ye].index),1), df1_complete_join.loc[df1_complete_join.index.year == ye, sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]],color='grey', alpha=0.2, lw=0.9, zorder=0)
    # Ploteo años destacados con colores
    for ye in range(len(highlighted_years)): # and not in [year_to_plot]:
        #ax.plot(df1_complete_join.loc[df1_complete_join.index.year == annual_vals[sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]].sort_values().index[-5:][ye]].index, df1_complete_join.loc[df1_complete_join.index.year == annual_vals[sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]].sort_values().index[-5:][ye], sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]],color=colores[ye])
#                ax.plot(np.arange(0,len(df1_complete_join.loc[df1_complete_join.index.year == annual_vals[sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]].sort_values().index[-3:][ye]].index),1), df1_complete_join.loc[df1_complete_join.index.year == annual_vals[sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]].sort_values().index[-3:][ye], sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]],color=colores[ye], label=df1_complete_join[df1_complete_join.index.year.isin(annual_vals[sorted(list(set(df1_complete_join.columns) & set(variables)), key=lambda x: variables.index(x))[i]].sort_values().index[-3:])].index.year.unique()[ye], lw=1.1, zorder=1)
        ax.plot(np.arange(0,len(df.loc[(df.index.year == highlighted_years[ye]) & ~((df.index.month ==2) & (df.index.day == 29))].index),1), df.loc[(df.index.year == highlighted_years[ye]) & ~((df.index.month == 2) & (df.index.day == 29)), variable],color=colors[ye], label=highlighted_years[ye], lw=1.5, zorder=1)

    if year_to_plot in df.index.year.unique():
        ax.plot(np.arange(0,len(df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)), variable].index),1), df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)), variable],color='magenta',lw=2.2,label=year_to_plot,zorder=2)
        ax.plot(np.arange(0,len(df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)), variable].index),1), df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)),'%s_median' %variable],color='k',label='Mediana [%i-%i]' %(climate_normal_period[0],climate_normal_period[1]),zorder=3)
        ax.plot(np.arange(0,len(df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)), variable].index),1), df[(df.index.year != year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29))].groupby(['Month','Day'])[variable].max().values,color='r',ls='--',lw=1,label='Maximum [%i-%i]' %(climate_normal_period[0],climate_normal_period[1]),zorder=3)
        ax.plot(np.arange(0,len(df.loc[(df.index.year == year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29)), variable].index),1), df[(df.index.year != year_to_plot) & ~((df.index.month == 2) & (df.index.day == 29))].groupby(['Month','Day'])[variable].min().values,color='b',ls='--',lw=1,label='Minimum [%i-%i]' %(climate_normal_period[0],climate_normal_period[1]),zorder=3)
    else:
        warnings.warn('year_to_plot has too many missing data to be plotted')
    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s (%s)' %(variable,units), fontsize=15)
    if criterion == 'highest':
        if units in ['ºC', 'ºF', 'K']:
            ax.legend(loc='lower right',ncol=4, fontsize=14, title='hottest years', title_fontsize=15).set_visible(True)
        elif units in ['mm', 'in']:
            ax.legend(loc='upper right',ncol=4, fontsize=14, title='wettest years', title_fontsize=15).set_visible(True)
        elif units in ['m/s', 'km/h', 'kts']:
            ax.legend(loc='upper right',ncol=4, fontsize=14, title='windiest years', title_fontsize=15).set_visible(True)
    elif criterion == 'lowest':
        if units in ['ºC', 'ºF', 'K']:
            ax.legend(loc='lower right',ncol=4, fontsize=14, title='coldest years', title_fontsize=15).set_visible(True)
        elif units in ['mm', 'in']:
            ax.legend(loc='upper right',ncol=4, fontsize=14, title='dryiest years', title_fontsize=15).set_visible(True)
        elif units in ['m/s', 'km/h', 'kts']:
            ax.legend(loc='upper right',ncol=4, fontsize=14, title='calmest years', title_fontsize=15).set_visible(True)
    elif criterion == 'latest':
        ax.legend(loc='lower right',ncol=4, fontsize=14, title='latest years', title_fontsize=15).set_visible(True)

    ax.tick_params(labelsize=14)

    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.185),
                        bbox_transform=ax.transAxes, prop={'size': 12}, frameon=True)
    text.patch.set_alpha(0.5)
    
    ax.set_xticks([0,31,59,90,120,151,180,211,242,272,303,333])
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.07, right=0.98, hspace=0.1, wspace=0.1)
    fig.suptitle('Yearly cycle of %s' %(variable), y=0.97, fontsize=24)

    fig.autofmt_xdate()
    plt.show()
    fig.savefig(filename,dpi=300)


def get_yearly_cycle(df, df_climate, vars_list):
    '''
    This function allows to extract the yearly cycles of selected variables

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological data
    vars_list: str
        Variable(s) to be extracted
    '''

    accum_mean_df = pd.DataFrame()
    for var in vars_list:
        accum_mean_tmp_df = pd.DataFrame()
        for y in df.index.year.unique():
            # Accumulated mean
            accum_mean = pd.DataFrame(df.loc[df.index.year == y, var].cumsum())
            accum_mean[var+'_nocount'] = df.loc[df.index.year == y, var]
            accum_mean['count'] = 0
            n = 1
            for h in range(len(accum_mean)):
                if np.isnan(accum_mean.iloc[h,1]) == False:
                    accum_mean.iloc[h,2] = n
                    n += 1
                else:
                    accum_mean.iloc[h,2] = n

            accum_mean['cummean'] = accum_mean[var]/ accum_mean['count']

            accum_mean_climate = pd.DataFrame(df_climate.loc[df_climate.index.year == y, var+'_median'].cumsum())
            accum_mean_climate[var+'_nocount'] = df_climate.loc[df_climate.index.year == y, var+'_median']
            accum_mean_climate['count'] = 0
            for h in range(len(accum_mean_climate)):
                accum_mean_climate.iloc[h,2] = h+1

            accum_mean_climate['cummean climate'] = accum_mean_climate[var+'_median']/ accum_mean_climate['count']



            accum_mean_all = pd.concat([accum_mean['cummean'], accum_mean_climate['cummean climate']], axis=1)

            accum_mean_all.columns = [var, var+'_median']

            accum_mean_tmp_df = pd.concat([accum_mean_tmp_df, accum_mean_all], axis=0)

        accum_mean_df = pd.concat([accum_mean_df, accum_mean_tmp_df], axis=1)


    accum_mean_df['Year'] = accum_mean_df.index.year
    accum_mean_df['Month'] = accum_mean_df.index.month
    accum_mean_df['Day'] = accum_mean_df.index.day

    return accum_mean_df


def annual_meteogram(df, df_climate, year_to_plot, climate_normal_period, database, station_name, filename,plot_anoms=False, show_seasons=True):
    '''
    This function allows to plot and compare the annual meteogram of a certain year (year_to_plot) with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological normal data
    year_to_plot: int
        Year to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    plot_anoms: boolean
        If true, temperature and wind values will be plotted as departures from the climatological selected statistic    
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season    
    '''
    
    if 'Temp' not in df.columns:
        raise ValueError('The temperature data should be included in df with column name "Temp"')
    if 'Rainfall' not in df.columns:
        raise ValueError('The rainfall data should be included in df with column name "Rainfall"')
    if 'WindSpeed' not in df.columns:
        raise ValueError('The wind speed data should be included in df with column name "WindSpeed"')

    yearmin = df.index.year.min()
    yearmax = df.index.year.max() 

    df = df[df.index.year == year_to_plot]
    df_climate = df_climate[df_climate.index.year == year_to_plot]

    df['Accum. Rainfall'] = df['Rainfall'].cumsum()
    df_climate['Accum. Rainfall'] = df_climate['Rainfall_mean'].cumsum()
    
    if df.index.month[-1] != 12 or df.index.month[-1] != 31:
        df = df.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))
    
    if df_climate.index.month[-1] != 12 or df_climate.index.month[-1] != 31:
        df_climate = df_climate.reindex(pd.date_range('%i-01-01' %year_to_plot, '%i-12-31' %year_to_plot, freq='1D'))


    fig, axs = plt.subplots(3, 1, figsize=(15,17), sharex=True)
    ax = axs.flatten()
    fmt_minor = mdates.DayLocator(interval=1)
    locator = mdates.AutoDateLocator(minticks=10, maxticks=16)
    formatter = mdates.ConciseDateFormatter(locator, zero_formats=['%d', '%d', '%d', '', '', ''], offset_formats=['','','','','',''])


    Tmedian, = ax[0].plot(df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'].index, df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'],color='k',label='Climate normal')
    if plot_anoms == False:
        ax[0].plot(df.loc[df.index.year == year_to_plot, 'Temp'].index, df.loc[df.index.year == year_to_plot,'Temp'],color='r',label='%i' %year_to_plot)
    else:
        diff_var = df.loc[df.index.year == year_to_plot, 'Temp'] - df_climate.loc[df_climate.index.year == year_to_plot,'Temp']
        mask1 = diff_var < 0
        mask2 = diff_var >= 0

        if len(mask1[mask1 == True]) > 0:
            ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'].index[mask1], bottom=df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'][mask1], height = diff_var[mask1], color ='#34b1eb', alpha=0.7)
        if len(mask2[mask2 == True]) > 0:
            ax[0].bar(df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'].index[mask2], bottom=df_climate.loc[df_climate.index.year == year_to_plot, 'Temp'][mask2], height = diff_var[mask2], color = '#eb4034', alpha=0.7)

    
    accumrainmedian, = ax[1].plot(df_climate.loc[df_climate.index.year == year_to_plot, 'Accum. Rainfall'].index, df_climate.loc[df_climate.index.year == year_to_plot, 'Accum. Rainfall'],color='k',label='_nolegend_')
    ax[1].plot(df.loc[df.index.year == year_to_plot, 'Accum. Rainfall'].index, df.loc[df.index.year == year_to_plot,'Accum. Rainfall'],color='r',label='_nolegend_')
    ax1 = ax[1].twinx()
    ax1.bar(x=df_climate.loc[df_climate.index.year == year_to_plot, 'Rainfall_mean'].index, height=df_climate.loc[df_climate.index.year == year_to_plot,'Rainfall_mean'],color='k',label='Climate normal', alpha=0.7)
    ax1.bar(x=df.loc[df.index.year == year_to_plot, 'Rainfall'].index, height=df.loc[df.index.year == year_to_plot,'Rainfall'],color='r',label='%i' %year_to_plot, alpha=0.7)

    windmedian, = ax[2].plot(df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed_median'].index, df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed_median' ],color='k',label='Climate normal')
    if plot_anoms == False:
        ax[2].plot(df.loc[df.index.year == year_to_plot, 'WindSpeed'].index, df.loc[df.index.year == year_to_plot,'WindSpeed'],color='r',label='%i' %year_to_plot)
    else:
        diff_var = df.loc[df.index.year == year_to_plot, 'WindSpeed'] - df_climate.loc[df_climate.index.year == year_to_plot,'WindSpeed_median']
        mask1 = diff_var < 0
        mask2 = diff_var >= 0

        if len(mask1[mask1 == True]) > 0:
            ax[2].bar(df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed'].index[mask1], bottom=df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed_median'][mask1], height = diff_var[mask1], color ='#34b1eb', alpha=0.7)
        if len(mask2[mask2 == True]) > 0:
            ax[2].bar(df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed'].index[mask2], bottom=df_climate.loc[df_climate.index.year == year_to_plot, 'WindSpeed_median'][mask2], height = diff_var[mask2], color = '#eb4034', alpha=0.7)




    for i in range(len(ax)):
        ax[i].tick_params(labelsize=16)
        ax[i].grid(color='black')
        ax[i].legend(fontsize=14)
        ax[i].set_xlim(df.index.min()-dt.timedelta(days=2), df.index.max()+dt.timedelta(days=2))
        if i< len(ax)-1:
            ax[i].set_xticklabels('', fontsize=0)
        else:
            ax[i].xaxis.set_major_locator(locator)
            ax[i].xaxis.set_major_formatter(formatter)

        ax[1].legend().set_visible(False)

        # Show seasons
        if show_seasons == True:
            season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
            ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),1,1)), mdates.date2num(dt.datetime(int(year_to_plot),3,1)), color="#4696db", alpha=0.2, zorder=-10)
            ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),3,1)), mdates.date2num(dt.datetime(int(year_to_plot),6,1)), color="#32a852", alpha=0.2, zorder=-10)
            ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),6,1)), mdates.date2num(dt.datetime(int(year_to_plot),9,1)), color="#da5757", alpha=0.2, zorder=-10)
            ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),9,1)), mdates.date2num(dt.datetime(int(year_to_plot),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
            ax[i].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),12,1)), mdates.date2num(dt.datetime(int(year_to_plot)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)


    ax1.legend(fontsize=14.5)
    ax1.tick_params(labelsize=17)

    ax[0].set_title('Daily temperature', fontsize=20)
    ax[1].set_title('Daily and accumulated rainfall', fontsize=20)
    ax[2].set_title('Daily mean wind speed', fontsize=20)

    text = AnchoredText('Alejandro Rodríguez Sánchez',
                        loc=1,bbox_to_anchor=(0.24, 0.185),
                        bbox_transform=ax[2].transAxes, prop={'size': 12}, frameon=True)
    text.patch.set_alpha(0.5)
    plt.text(0.031, 0.95, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=15, transform=plt.gcf().transFigure)
    plt.text(0.031, 0.93, 'Period with data: %i-%i' %(yearmin, yearmax), fontsize=15, transform=plt.gcf().transFigure)
    plt.text(0.79, 0.95, 'Database: %s' %database, fontsize=15, transform=plt.gcf().transFigure)
    plt.text(0.79, 0.93, 'Location: %s' %station_name, fontsize=15, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.1, wspace=0.1,bottom=0.05, top=0.9)
    #fig.autofmt_xdate()
    fig.savefig(filename,dpi=300)


def when_season_starts(df, df_climate, variable, units, year_to_plot,  climate_normal_period, database, station_name, filename, climate_stat='median'):
    '''
    This function allows to plot and compare the start of a season (based on a 15-day MA of variable) with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological normal data
    var: str
        Name of the variable to be plotted
    units: str
        Units of the variable to be plotted
    year_to_plot: int or list
        Year(s) to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    climate_stat: str
        Metric to compute the climatological normal values (mean or median)

    '''
    
    df = df.copy()

    minyear = df.index.year.min()
    maxyear = df.index.year.max()

    df_climate = df[(df.index.year>=climate_normal_period[0]) & (df.index.year<=climate_normal_period[1])][variable].rolling(window=15).mean()
    df_climate = df_climate.groupby([df_climate.index.month, df_climate.index.day]).mean(numeric_only=True)
    df_climate = df_climate[~((df_climate.index.get_level_values(1) == 29) & (df_climate.index.get_level_values(0)  == 2))]

    if isinstance(year_to_plot, list):
        #df = df[df.index > dt.datetime(year_to_plot.min(), 1, 1) - dt.timedelta(days=15)].groupby(['Day','Month']).mean(numeric_only=True).rolling(window=15).mean()[variable]
        df1 = df[(df.index > dt.datetime(year_to_plot.min(), 1, 1) - dt.timedelta(days=15))]
        df = df1.rolling(window=15).mean(numeric_only=True).groupby([df1.index.month, df1.index.day]).mean(numeric_only=True)[variable]
        df = df[~((df.index.get_level_values(1) == 29) & (df.index.get_level_values(0)  == 2))]

        #df_climate = df_climate[df_climate.index.year==max(year_to_plot)][variable+'_'+climate_stat].rolling(window=15).mean()

    else:
        df = df[(df.index > dt.datetime(year_to_plot, 1, 1) - dt.timedelta(days=15)) & (df.index.year < year_to_plot+1)][variable].rolling(window=15).mean()
        df = df.groupby([df.index.month,df.index.day]).mean(numeric_only=True)
        df = df[~((df.index.get_level_values(1) == 29) & (df.index.get_level_values(0)  == 2))]

        #df_climate = df_climate[df_climate.index.year==year_to_plot][variable+'_'+climate_stat].rolling(window=15).mean()

    # Set season limit values
    #spring_start_value = float(df_climate[(df_climate.index.month == 3) & (df_climate.index.day == 1)])#.values[-1]
    #summer_start_value = float(df_climate[(df_climate.index.month == 6) & (df_climate.index.day == 1)])#.values[-1]
    #autumn_start_value = float(df_climate[(df_climate.index.month == 9) & (df_climate.index.day == 1)])#.values[-1]
    #winter_start_value = float(df_climate[(df_climate.index.month == 12) & (df_climate.index.day == 1)])#.values[-1]

    spring_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 3) & (df_climate.index.get_level_values(1) == 1)])
    summer_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 6) & (df_climate.index.get_level_values(1) == 1)])
    autumn_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 9) & (df_climate.index.get_level_values(1) == 1)])
    winter_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 12) & (df_climate.index.get_level_values(1) == 1)])


    start_values = [spring_start_value, summer_start_value, autumn_start_value, winter_start_value]

    start_dates = {}

    orig_dates = [59,151,242,333]

    df1 = df.copy() # For manipulation regarding NaN values
    df1 = df1.ffill() # Fill NaN values with last non-NaN value

    for i in range(4):
        if i < 2:
            cond = df1 > start_values[i] # Identify values exceeding threshold
            cond = cond.reset_index(drop=True)
            #cond = df[df != np.nan] > start_values[i] # Identify values exceeding threshold

            if i > 0:
                b = cond[cond.index > max(start_dates.values())].cumsum() # Convert boolean to int
            else:
                #b = cond[cond.index <dt.datetime(year_to_plot,6,1)].cumsum()
                b = cond.cumsum()
            c = b.sub(b.mask(cond).ffill().fillna(0)).astype(int) # Count consecutive values

            if c.max() > 0:
                start_dates[i] = c.idxmax() - c.max() - 1 #dt.timedelta(days=int(c.max()-1)) # - dt.timedelta(days=int(c.max()-1))
            else:
                start_dates[i] = orig_dates[i] #np.nan

        else:
            #cond = df1[::-1][(df1[::-1] != np.nan)] >= start_values[i] # Identify values exceeding threshold
            cond = df1.reset_index(drop=True)[::-1] >= start_values[i] # Identify values exceeding threshold

            if len(cond[cond == True]) == 0:
                start_dates[i] = orig_dates[i] #np.nan
            else:
                start_dates[i] = cond.idxmax() + 1 # cond.idxmax() + dt.timedelta(days=1)

    #locator = mdates.MonthLocator() #minticks=3, maxticks=12)
    #formatter = mdates.ConciseDateFormatter(locator)
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(15,7),sharex=True, sharey=True)
    ax = axs.flatten()
    # Plot climatological normal
    ax[0].plot(df_climate.values, linewidth=1.2, color='#000000', label='Climate normal')

    # Plot year to plot
    ax[1].plot(df.values, linewidth=1.2, color='#000000', label='%i' %year_to_plot)

    # Axes properties
    ax[0].set_title('Climatological normal', fontsize=18)
    if isinstance(year_to_plot, list):
        ax[1].set_title('Average of %i-%i period' %(min(year_to_plot), max(year_to_plot)), fontsize=18)
    else:
        ax[1].set_title('%i' %year_to_plot, fontsize=18)


    for i in range(len(ax)):
        ax[i].tick_params(labelsize=16)
        ax[i].grid(color='black')
        #ax[i].legend(fontsize=14)
        #ax[i].set_xlim(dt.datetime(year_to_plot,1,1)-dt.timedelta(days=3), df.index.max()+dt.timedelta(days=3))
        ax[i].margins(x=0)
        ax[i].set_ylabel('%s [%s]' %(variable, units), fontsize=17)
        if i< len(ax)-1:
            ax[i].set_xticklabels('', fontsize=0)
        else:
            #ax[i].xaxis.set_major_locator(locator)
            #ax[i].xaxis.set_major_formatter(formatter)
            ax[i].set_xticks([0,31,59,90,120,151,180,211,242,272,303,333])
            ax[i].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=14)



    season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46', "#4696db"]
    # Climate normal seasons
    #ax[0].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),1,1)), mdates.date2num(dt.datetime(int(year_to_plot),3,1)), color="#4696db", alpha=0.2, zorder=-10)
    #ax[0].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),3,1)), mdates.date2num(dt.datetime(int(year_to_plot),6,1)), color="#32a852", alpha=0.2, zorder=-10)
    #ax[0].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),6,1)), mdates.date2num(dt.datetime(int(year_to_plot),9,1)), color="#da5757", alpha=0.2, zorder=-10)
    #ax[0].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),9,1)), mdates.date2num(dt.datetime(int(year_to_plot),12,1)), color="#d6db46", alpha=0.2, zorder=-10)
    #ax[0].axvspan(mdates.date2num(dt.datetime(int(year_to_plot),12,1)), mdates.date2num(dt.datetime(int(year_to_plot)+1,1,1)), color="#4696db", alpha=0.2, zorder=-10)
    orig_dates_plot = [0,59,151,242,333]
    for i in range(len(orig_dates_plot)-1):
        ax[0].axvspan(orig_dates_plot[i], orig_dates_plot[i+1], color=season_colors[i], alpha=0.2, zorder=-10)
    ax[0].axvspan(333,365, color=season_colors[0], alpha=0.2, zorder=-10)

    # year_to_plot seasons, based on climate normal temperature limits
    # Variables for arrow and text
    #orig_dates = [dt.datetime(int(year_to_plot),3,1), dt.datetime(int(year_to_plot),6,1), dt.datetime(int(year_to_plot),9,1), dt.datetime(int(year_to_plot),12,1)]
    shifts = {}
    #for i in range(4):
       # if isinstance(start_dates[i] , float):
       #     shifts[i] = np.nan
       # else:
       #     shifts[i] = start_dates[i] - orig_dates[i]
    # Variables for arrow and text
    shifts = {}
    for i in range(4):
        shifts[i] = start_dates[i] - orig_dates[i]


    #for i in range(len(start_dates)):
    #    if isinstance(start_dates[i], float):
    #        start_dates[i] = orig_dates[i]
#    start_dates_plot = [dt.datetime(int(year_to_plot),1,1)] + list(start_dates.values()) + [dt.datetime(int(year_to_plot)+1,1,1)]
    start_dates_plot = [0] + list(start_dates.values()) + [366]
    for i in range(len(start_dates_plot)-1):
        #ax[1].axvspan(mdates.date2num(pd.to_datetime(start_dates_plot[i])), mdates.date2num(pd.to_datetime(start_dates_plot[i+1])), color=season_colors[i], alpha=0.2, zorder=-10)
        ax[1].axvspan(start_dates_plot[i], start_dates_plot[i+1], color=season_colors[i], alpha=0.2, zorder=-10)
   
    # Add reference lanes to climate normal
    #ax[1].axvline(mdates.date2num(dt.datetime(int(year_to_plot),3,1)), linestyle='--', color='k', alpha=0.3)
    #ax[1].axvline(mdates.date2num(dt.datetime(int(year_to_plot),6,1)), linestyle='--', color='k', alpha=0.3)
    #ax[1].axvline(mdates.date2num(dt.datetime(int(year_to_plot),9,1)), linestyle='--', color='k', alpha=0.3)
    #ax[1].axvline(mdates.date2num(dt.datetime(int(year_to_plot),12,1)), linestyle='--', color='k', alpha=0.3)
    ax[1].axvline(59, linestyle='--', color='k', alpha=0.8)
    ax[1].axvline(151, linestyle='--', color='k', alpha=0.8)
    ax[1].axvline(242, linestyle='--', color='k', alpha=0.8)
    ax[1].axvline(333, linestyle='--', color='k', alpha=0.8)


    for i in range(4):
        if isinstance(shifts[i] , float):
            continue
        if shifts[i] > 0:
            t = ax[1].text(orig_dates[i], max(ax[1].get_ylim())*0.94, '%i d' %np.abs(shifts[i]),
                    ha="center", va="center", rotation=0, size=13,
                    bbox=dict(boxstyle="rarrow,pad=0.3",
                        fc="lightblue", ec="steelblue", lw=2))
            bb = t.get_bbox_patch()
            #bb.set_boxstyle("rarrow", pad=0.6)

        else:
            ax[1].text(orig_dates[i], max(ax[1].get_ylim())*0.94, '%i d' %np.abs(shifts[i]),#.days),
                    ha="center", va="center", rotation=0, size=13,
                    bbox=dict(boxstyle="larrow,pad=0.3",
                        fc="lightblue", ec="steelblue", lw=2))


    plt.text(0.03, 0.955, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.925, 'Period with data: %i-%i' %(minyear,maxyear), fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.955, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.8, 0.925, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.07, right=0.98, hspace=0.15, wspace=0.1)
    fig.savefig(filename,dpi=300)


def when_season_starts_evolution(df, df_climate, variable, year_to_plot,  climate_normal_period, database, station_name, filename, plot_type='heatmap', grouping_freq=10, extract_data=False):
    '''
    This function allows to plot the evolution along years of the start of each season (based on a 15-day MA of variable) with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the data
    var: str
        Name of the variable to be plotted
    year_to_plot: int or list
        Year(s) to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    plot_type: str
        String that controls how to plot the data
    grouping_freq: int
        The length (in years) of each group to be analysed
    extract_data: boolean
        If True, it extracts the evolution of the number of days of each season
    '''
    df = df.copy()
    yearmin = df.index.year.min()
    yearmax = df.index.year.max()

    season_names = ['winter', 'spring', 'summer', 'autumn']

    df_climate = df[(df.index.year>=climate_normal_period[0]) & (df.index.year<=climate_normal_period[1])][variable].rolling(window=15).mean()
    df_climate = df_climate.groupby([df_climate.index.month, df_climate.index.day]).mean(numeric_only=True)
    #df_climate = df_climate[~((df_climate.index.get_level_values(1) == 29) & (df_climate.index.get_level_values(0)  == 2))].reset_index(drop=True)

    # Group by X years
    decade_ini = np.arange(df.index.year.min(),df.index.year.max()+1,grouping_freq)
    dec_names = []
    for dec in range(len(decade_ini)): #, len(decade_fin))):
        dec_names.append('%s-%s' %(decade_ini[dec],decade_ini[dec]+grouping_freq-1))

    if isinstance(year_to_plot, list):
        grouped_df_list = {}
        for i in range(len(decade_ini)):
            df1 = df[(df.index >= dt.datetime(decade_ini[i], 1, 1) - dt.timedelta(days=15)) & (df.index < dt.datetime(decade_ini[i]+grouping_freq, 1, 1))]
            grouped_df_list[i] = df1.rolling(window=15).mean(numeric_only=True).groupby([df1.index.month, df1.index.day]).mean(numeric_only=True)[variable]
            grouped_df_list[i] = grouped_df_list[i][~((grouped_df_list[i].index.get_level_values(1) == 29) & (grouped_df_list[i].index.get_level_values(0)  == 2))].reset_index(drop=True)

    else:
        df = df[df.index.year==year_to_plot][variable].rolling(window=15).mean()


    # Set season limit values
    spring_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 3) & (df_climate.index.get_level_values(1) == 1)])
    summer_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 6) & (df_climate.index.get_level_values(1) == 1)])
    autumn_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 9) & (df_climate.index.get_level_values(1) == 1)])
    winter_start_value = float(df_climate[(df_climate.index.get_level_values(0) == 12) & (df_climate.index.get_level_values(1) == 1)])

    start_values = [spring_start_value, summer_start_value, autumn_start_value, winter_start_value]

    start_dates_plot_list = []
    seasons_df = pd.DataFrame()
    seasons_df_nans = pd.DataFrame()
    orig_dates = [59,151,242,333]
    season_colors = ["#4696db", '#32a852', "#da5757", '#d6db46']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('season_colors', season_colors, N=4)

    for j in range(len(grouped_df_list)):
        start_dates = {}
        df1 = grouped_df_list[j].copy() # For manipulation regarding NaN values
        df1 = df1.ffill() # Fill NaN values with last non-NaN value

        for i in range(4):
            if i < 2:
                cond = df1[df1 != np.nan] >= start_values[i] # Identify values exceeding threshold
                if i > 0:
                    b = cond[cond.index > max(start_dates.values())].cumsum() # Convert boolean to int
                else:
                    b = cond.cumsum()
                c = b.sub(b.mask(cond).ffill().fillna(0)).astype(int) # Count consecutive values

                if c.max() > 0:
                    start_dates[i] = c.idxmax() - c.max() - 1 #dt.timedelta(days=int(c.max()-1))
                else:
                    start_dates[i] = orig_dates[i] #np.nan
            else:
                cond = df1[::-1][(df1[::-1] != np.nan)] >= start_values[i] # Identify values exceeding threshold

                if len(cond[cond == True]) == 0:
                    start_dates[i] = orig_dates[i] #np.nan
                else:
                    start_dates[i] = cond.idxmax() + 1
            

        # Variables for arrow and text
        shifts = {}
        for i in range(4):
            shifts[i] = start_dates[i] - orig_dates[i]

        # Assign values to each season
        df = pd.DataFrame(grouped_df_list[j])
        df['season'] = 1
        df['season'][df.index >= list(start_dates.values())[0]] = 2
        df['season'][df.index >= list(start_dates.values())[1]] = 3
        df['season'][df.index >= list(start_dates.values())[2]] = 4
        df['season'][df.index >= list(start_dates.values())[3]] = 1
        seasons_df = pd.concat([seasons_df, pd.DataFrame(df['season'])], axis=1)

        df['nan'] = 0
        df['nan'][df[variable].isnull()] = 1
        seasons_df_nans = pd.concat([seasons_df_nans, pd.DataFrame(df['nan'])], axis=1)

    seasons_df.columns = dec_names

    # For ndays plotting
    seasons_valuecounts = seasons_df.apply(pd.Series.value_counts, axis=0).fillna(0) #seasons_df.T.value_counts()
    #seasons_valuecounts.columns = ['winter','spring', 'summer', 'autumn']

    # For hatching
    seasons_df_nans = seasons_df_nans.T #.columns = dec_names
    seasons_df_nans.index = np.arange(0,len(seasons_df_nans),1)
    seasons_df_nans['bool'] = seasons_df_nans.gt(0).sum(axis=1).ge(1)
    seasons_df_nans = seasons_df_nans[seasons_df_nans['bool'] ==True]

    # Plotting
    if plot_type == 'heatmap':
        fig=plt.figure(figsize=(15,7))
        ax=fig.add_subplot(111)
        r=ax.matshow(seasons_df.T, interpolation=None,cmap=cmap, vmax=4, vmin=1,aspect='auto', alpha=0.7)
        # Add hatching where no data is available
        for i in seasons_df_nans.index:
            row = seasons_df_nans.loc[i,:]
            row = row.drop(index='bool')
            nans = row[row == 1]
            list_of_nans = [d for _, d in nans.groupby(nans.index - np.arange(len(nans)))]
            for j in range(len(list_of_nans)):
                ax.add_patch(matplotlib.patches.Rectangle(((list_of_nans[j].index[0]-0.5), (i-0.5)),list_of_nans[j].index[-1] - list_of_nans[j].index[0], 1, 
                hatch='//', fill=False, snap=False, linewidth=0))
        ax.axvline(59, linestyle='--', color='k', alpha=0.8)
        ax.axvline(151, linestyle='--', color='k', alpha=0.8)
        ax.axvline(242, linestyle='--', color='k', alpha=0.8)
        ax.axvline(333, linestyle='--', color='k', alpha=0.8)


        ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)
        ax.set_xticks([0,31,59,90,120,151,180,211,242,272,303,333])
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=14)
        # For the minor ticks, use no labels; default NullFormatter.
        ax.set_xticks([14,45,73,104,134,165,194,225,256,286,317,347], minor=True)
        #ax.set_xticklabels(['15','15','15','15','15','15','15','15','15','15','15','15'], fontsize=10, minor=True)
        #ax.set_yticks(np.arange(0,len(seasons_df.columns),1))
        ticks = [int(x) for x in ax.get_yticks()]
        ticks = [i for i in ticks if i < len(dec_names)]
        ax.set_yticklabels(list(np.array(dec_names)[ticks]),fontsize=15)
#        ax.set_yticklabels(dec_names[int(x) for x in ax.get_yticks()])
        #print(ax.get_yticks())
        #print(ax.get_yticks())

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1.2, axis='y',zorder=12)
        ax.grid(which='major', color='white', linestyle='-', linewidth=1.2, axis='y',zorder=12)

        # Annotate season duration in first and last rows.
        #asd = seasons_df.T
        #for i in range(len(vegetables)):
        #    for j in range(len(farmers)):
        #        text = ax.text(j, i, harvest[i, j],
        #                       ha="center", va="center", color="w")

        ## Matshow legend
        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [ r.cmap(r.norm(value)) for value in [1,2,3,4]]
        # create a patch (proxy artist) for every color 
        patches = [ matplotlib.patches.Patch(color=colors[i], label=season_names[i]) for i in range(len([1,2,3,4])) ]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches, bbox_to_anchor=(.77, -.05),ncols=4, fontsize=16) #, loc=2, borderaxespad=0. )

        fig.suptitle('How have seasons evolved in time?', y=0.97, fontsize=24)
        plt.text(0.03, 0.96, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.93, 'Period with data: %i-%i' %(yearmin,yearmax), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.96, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.93, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)
        fig.savefig(filename,dpi=300)

    if plot_type == 'line':
        fig, ax=plt.subplots(figsize=(15,7))
        for i in range(4):
            r=ax.plot(seasons_valuecounts.columns, seasons_valuecounts.iloc[i,:], color = season_colors[i], zorder=10, alpha=1, label=['winter','spring', 'summer', 'autumn'][i])

        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, right=True, labelright=True, labelsize=16)
        if len(seasons_valuecounts.columns) > 10 and len(seasons_valuecounts.columns) < 20:
            ax.set_xticks(np.arange(0,len(seasons_valuecounts.columns),2))
            ax.set_xticklabels(seasons_valuecounts.columns[::2])
        elif len(seasons_valuecounts.columns) >= 20 and len(seasons_valuecounts.columns) < 40:
            ax.set_xticks(np.arange(0,len(seasons_valuecounts.columns),5))
            ax.set_xticklabels(seasons_valuecounts.columns[::5])
        elif len(seasons_valuecounts.columns) >= 40:
            ax.set_xticks(np.arange(0,len(seasons_valuecounts.columns),10))
            ax.set_xticklabels(seasons_valuecounts.columns[::10])
        # Gridlines based on minor ticks
        ax.grid(color='grey', linestyle='-') #, linewidth=1.2, axis='y',zorder=12)
        ax.margins(x=0)
        #ax.yaxis.set_major_locator(plt.MaxNLocator(4, min_n_ticks=3))

        ax.legend(ncol=2, fontsize=15)

        fig.suptitle('Seasons duration in days', y=0.97, fontsize=24)
        plt.text(0.03, 0.96, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.93, 'Period with data: %i-%i' %(yearmin,yearmax), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.96, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.93, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure)    
        fig.savefig(filename,dpi=300)

    if plot_type == 'both':
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        fig=plt.figure(figsize=(15,7))
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        r=ax.matshow(seasons_df.T, interpolation=None,cmap=cmap, vmax=4, vmin=1,aspect='auto', alpha=0.7)
        # Add hatching where no data is available
        for i in seasons_df_nans.index:
            row = seasons_df_nans.loc[i,:]
            row = row.drop(index='bool')
            nans = row[row == 1]
            list_of_nans = [d for _, d in nans.groupby(nans.index - np.arange(len(nans)))]
            for j in range(len(list_of_nans)):
                ax.add_patch(matplotlib.patches.Rectangle(((list_of_nans[j].index[0]-0.5), (i-0.5)),list_of_nans[j].index[-1] - list_of_nans[j].index[0], 1, 
                hatch='//', fill=False, snap=False, linewidth=0))
        ax.axvline(59, linestyle='--', color='k', alpha=0.8)
        ax.axvline(151, linestyle='--', color='k', alpha=0.8)
        ax.axvline(242, linestyle='--', color='k', alpha=0.8)
        ax.axvline(333, linestyle='--', color='k', alpha=0.8)


        ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, right=True, labelright=True)
        ax.set_xticks([0,31,59,90,120,151,180,211,242,272,303,333])
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=14)
        # For the minor ticks, use no labels; default NullFormatter.
        ax.set_xticks([14,45,73,104,134,165,194,225,256,286,317,347], minor=True)
        #ax.set_yticks(np.arange(0,len(seasons_df.columns),1))
        ticks = [int(x) for x in ax.get_yticks()]
        ticks = [i for i in ticks if i < len(dec_names)]
        ax.set_yticklabels(list(np.array(dec_names)[ticks]), fontsize=14.5)

        # Gridlines based on minor ticks
        #ax.grid(which='minor', color='k', linestyle='-', linewidth=1.2, axis='y',zorder=12)
        #ax.grid(which='major', color='white', linestyle='-', linewidth=1.2, axis='y',zorder=12)

        ## Matshow legend
        # Add legend for hatching
        patch_hatched = matplotlib.patches.Patch(facecolor='None', hatch='///', edgecolor="k", label='No Data')
        legend_2=ax.legend(handles=[patch_hatched], loc='upper left', bbox_to_anchor=(.6, 0.995), fontsize=15, framealpha=0.6)

        # as soon as a second legend is made, the first disappears and needs to be added back again
        ax.add_artist(legend_2) #python now knows that "figure" must take the "legend_1" along with "legend_2"

        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [ r.cmap(r.norm(value)) for value in [1,2,3,4]]
        # create a patch (proxy artist) for every color 
        patches = [ matplotlib.patches.Patch(color=colors[i], label=season_names[i]) for i in range(len([1,2,3,4])) ]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(.58, 0.995),ncols=4, fontsize=15, framealpha=0.6) #, loc=2, borderaxespad=0. )


        # Annotate season duration in first and last rows.
#        asd = seasons_df.T
        #for i in range(len(vegetables)):
        #    for j in range(len(farmers)):
        #        text = ax.text(j, i, harvest[i, j],
        #                       ha="center", va="center", color="w")

        fig.suptitle('How have seasons evolved in time?', y=0.97, fontsize=24)
        plt.text(0.03, 0.96, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.03, 0.93, 'Period with data: %i-%i' %(yearmin,yearmax), fontsize=14, transform=plt.gcf().transFigure)
        #plt.text(0.8, 0.96, 'Database: %s' %database, fontsize=14, transform=plt.gcf().transFigure)
        plt.text(0.8, 0.93, 'Location: %s' %station_name, fontsize=14, transform=plt.gcf().transFigure, wrap=True)

        for i in range(4):
            r=ax2.plot(seasons_valuecounts.columns, seasons_valuecounts.iloc[i,:], color = season_colors[i], zorder=10, alpha=1, label=['winter','spring', 'summer', 'autumn'][i])

        ax2.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, right=True, labelright=True, labelsize=14)
        #ax.set_xticks()
        if len(seasons_valuecounts.columns) > 10 and len(seasons_valuecounts.columns) < 20:
            ax2.set_xticks(np.arange(0,len(seasons_valuecounts.columns),2))
            ax2.set_xticklabels(seasons_valuecounts.columns[::2])
        elif len(seasons_valuecounts.columns) >= 20 and len(seasons_valuecounts.columns) < 40:
            ax2.set_xticks(np.arange(0,len(seasons_valuecounts.columns),5))
            ax2.set_xticklabels(seasons_valuecounts.columns[::5])
        elif len(seasons_valuecounts.columns) >= 40:
            ax2.set_xticks(np.arange(0,len(seasons_valuecounts.columns),10))
            ax2.set_xticklabels(seasons_valuecounts.columns[::10])


        # Gridlines based on minor ticks
        ax2.grid(color='grey', linestyle='-') #, linewidth=1.2, axis='y',zorder=12)
        ax2.margins(x=0)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4, min_n_ticks=3))
    
        plt.subplots_adjust(bottom=0.05,left=0.08) #, right=0.98, hspace=0.15, wspace=0.1)    
        fig.savefig(filename,dpi=300)

        if extract_data == True:
            return seasons_valuecounts

def plot_accumulated_anomalies(df, var, units, year_to_plot, climate_normal_period, database, station_name, filename, window=12,freq='1D'):
    '''
    This function allows to plot the evolution along years of the anomalies with respect with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Name of the variable to be plotted
    units: str
        Units of the variable to be plotted
    year_to_plot: int
        Year to be highlighted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    window: int
        Length of the window for plotting moving average of the anomalies
    gfreq: int
        The frequency of each group to be analysed
    '''
    
    df = df.copy()
    df_climate = df[(df.index.year >= climate_normal_period[0]) & (df.index.year <= climate_normal_period[1])][var]

    # 1. Compute anomalies
    df = df[~((df.index.day == 29) & (df.index.month == 2))]
    df['Day'] = df.index.day
    df['Month'] = df.index.month

    df = df[['Month','Day',var]]
    #df_anoms = df_anoms.groupby([df_anoms.index.month,df_anoms.index.day]).mean(numeric_only=True)
    #df_anoms = df_anoms[~((df_anoms.index.get_level_values(1) == 29) & (df_anoms.index.get_level_values(0)  == 2))][var]

    df_climate_anoms = df_climate[~((df_climate.index.day == 29) & (df_climate.index.month == 2))] 
    df_climate_anoms = df_climate_anoms.groupby([df_climate_anoms.index.month,df_climate_anoms.index.day]).mean(numeric_only=True)
    df_climate_anoms = df_climate_anoms[~((df_climate_anoms.index.get_level_values(1) == 29) & (df_climate_anoms.index.get_level_values(0)  == 2))].reset_index()
    df_climate_anoms.columns = ['Month','Day',var+'_climate']
    
    df_anoms = df[[var, 'Month', 'Day']].merge(df_climate_anoms, on=['Month', 'Day'])
    df_anoms[var+'_anom'] = df_anoms[var] - df_anoms[var+'_climate']
    df_anoms.index = df.index


    # 2. Grouping
    df_anoms = df_anoms.groupby(pd.Grouper(freq=freq)).mean(numeric_only=True)
    
    # 2.1 After grouping, group by year and compute cumulative anomalies
    df_dict = {}
    final_anom = {} # Save final anomalies for plotting reasons
    for y in df_anoms.index.year.unique():
        df_year = df_anoms[df_anoms.index.year == y].reset_index()
        df_year['anom_cumsum'] = df_year[var+'_anom'].cumsum()
        df_year[var+'_cumanom'] = df_year['anom_cumsum'] / (df_year.index + 1)

        df_dict[y] = df_year
        final_anom[y] = df_year.loc[len(df_year)-1,var+'_cumanom']

    # Get lowest and highest final anomalies
    lowest_anom = min(final_anom, key=final_anom.get) # Get year with lowest anomaly
    highest_anom = max(final_anom, key=final_anom.get) # Get year with highest anomaly

    # Barras de anomalías mensuales
    fig, ax = plt.subplots(figsize=(15,7))

    # 2.1 Plot values one year at a time
    for y in df_anoms.index.year.unique():
        df_year = df_dict[y]
#        df_year['anom_cumsum'] = df_year[var+'_anom'].cumsum()
#        df_year[var+'_cumanom'] = df_year['anom_cumsum'] / (df_year.index + 1)

        if y == lowest_anom:
            ax.plot(df_year.index, df_year[var+'_cumanom'], color = 'blue', alpha=1,linewidth=1.2, zorder=10, label='Lowest anom.: %s' %str(lowest_anom) )
        elif y == highest_anom:
            ax.plot(df_year.index, df_year[var+'_cumanom'], color = 'red', alpha=1,linewidth=1.2, zorder=10, label = 'Highest anom.: %s' %str(highest_anom))
        elif y == year_to_plot:
            ax.plot(df_year.index, df_year[var+'_cumanom'], color = 'k', alpha=1,linewidth=1.3, zorder=11, label = year_to_plot)
        else:
            ax.plot(df_year.index, df_year[var+'_cumanom'], color = 'grey', alpha=0.5,linewidth=0.9)

    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s anomaly [%s]' %(var, units),fontsize=15)
    #ax.set_title('%s cumulative anomaly' %var, fontsize=18, wrap=True)
    ax.set_xlim([0,max(df_year.index)])
    if 'W' in freq:
        ax.set_xlim([0,np.ceil(52/int(freq[:-1]))])
    if 'D' in freq:
        locator = mdates.AutoDateLocator(maxticks=20) #minticks=3, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    elif freq == '1M':
        ax.set_xticks(np.arange(0,12,1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',' Dec'])

    elif 'W' in freq:
        ax.set_xticks(np.arange(0,52,2*int(freq[:-1])))

    ax.tick_params(axis='both', labelsize=15)

    #ax.xaxis.set_major_locator(mdates.YearLocator(2))
    # Get only the year to show in the x-axis:
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.text(0.03, 0.96, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.93, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.96, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.93, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    fig.suptitle('%s anomaly [%s]' %(var, units), fontsize=24)
    ax.legend(fontsize=14).set_visible(True)
    ax.margins(0.01, 0.05)
    plt.tight_layout()
    plt.savefig(filename,dpi=300)

def plot_anomalies(df, var, units, climate_normal_period, database, station_name, filename, window=12,freq='1D'):
    '''
    This function allows to plot the evolution along years of the anomalies with respect with the climatological normal

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Name of the variable to be plotted
    units: str
        Units of the variable to be plotted
    climate_normal_period: list
        First and last year of the reference period to be used as climatological normal
    station_name: str
        Name of the location of the data
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    window: int
        Length of the window for plotting moving average of the anomalies
    gfreq: int
        The frequency of each group to be analysed
    '''
    
    df = df.copy()
    df_climate = df[(df.index.year >= climate_normal_period[0]) & (df.index.year <= climate_normal_period[1])][var]

    # 1. Compute anomalies
    df = df[~((df.index.day == 29) & (df.index.month == 2))]
    df['Day'] = df.index.day
    df['Month'] = df.index.month

    df = df[['Month','Day',var]]
    #df_anoms = df_anoms.groupby([df_anoms.index.month,df_anoms.index.day]).mean(numeric_only=True)
    #df_anoms = df_anoms[~((df_anoms.index.get_level_values(1) == 29) & (df_anoms.index.get_level_values(0)  == 2))][var]

    df_climate_anoms = df_climate[~((df_climate.index.day == 29) & (df_climate.index.month == 2))] 
    df_climate_anoms = df_climate_anoms.groupby([df_climate_anoms.index.month,df_climate_anoms.index.day]).mean(numeric_only=True)
    df_climate_anoms = df_climate_anoms[~((df_climate_anoms.index.get_level_values(1) == 29) & (df_climate_anoms.index.get_level_values(0)  == 2))].reset_index()
    df_climate_anoms.columns = ['Month','Day',var+'_climate']
    
    df_anoms = df[[var, 'Month', 'Day']].merge(df_climate_anoms, on=['Month', 'Day'])
    df_anoms[var+'_anom'] = df_anoms[var] - df_anoms[var+'_climate']
    df_anoms.index = df.index


    # 2. Grouping
    df_anoms = df_anoms.groupby(pd.Grouper(freq=freq)).mean(numeric_only=True)
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(15,7))

    mask1 = df_anoms[var+'_anom'] > 0
    mask2 = df_anoms[var+'_anom'] <= 0

    if len(mask1[mask1 == True]) > 0:
        ax.bar(df_anoms.index[mask1], df_anoms[var+'_anom'][mask1],color='tab:red',width=21,edgecolor='None',linewidth=0.85)
    if len(mask2[mask2 == True]) > 0:
        ax.bar(df_anoms.index[mask2], df_anoms[var+'_anom'][mask2],color='tab:blue',width=21,edgecolor='None',linewidth=0.85)
    ax.grid(color='black',alpha=0.5)
    ax.set_ylabel('%s anomaly [%s]' %(var, units),fontsize=15)
#    ax.set_xlim([0,max(df_year.index)])
    
    ax.plot(df_anoms[var+'_anom'].rolling(window=window).mean(),color='black',label='%i-period MA' %window)

    locator = mdates.AutoDateLocator(maxticks=20) #minticks=3, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis='both', labelsize=15)

    #ax.xaxis.set_major_locator(mdates.YearLocator(2))
    # Get only the year to show in the x-axis:
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.text(0.03, 0.96, 'Climate normal period: %i-%i' %(climate_normal_period[0],climate_normal_period[1]), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.93, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.96, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.93, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    fig.suptitle('%s anomaly [%s]' %(var, units), fontsize=24)
    ax.legend(fontsize=14).set_visible(True)
    ax.margins(0.01, 0.05)
    plt.tight_layout()
    plt.savefig(filename,dpi=300)

def compare_with_globaldataset(df, var, units, database, station_name, filename, window=None, global_dataset='HadCRUT'):
    '''
    This function allows to compare the annual trend of your timeseries with that of one global dataset

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        Name of the variable to be plotted
    units: str
        Units of the variable to be plotted
    database: str
        Name of the database from which the plotted data comes
    station_name: str
        Name of the location of the data
    filename: str
        Absolute path where the plot is going to be saved
    window: int
        Length of the window for moving average computation. If None, no moving average is computed and the original data anomalies are plotted.
    global_dataset: str
        The name of the dataset to compare with. At the moment, only "HadCRUT" is accepted.
    '''

    df = df.copy()
    path = Path(os.path.dirname(__file__)) # Current directory

    normal_values = df[(df.index.year >= 1961) & (df.index.year <= 1990)]
    if len(normal_values) == 0:
        raise ValueError('Your data has not data within the period 1961-1990. Therefore, there cannot be perform a comparison with the global dataset. Exiting...')

    normal_values = normal_values.groupby(normal_values.index).mean(numeric_only=True)[var].mean(numeric_only=True)
    

    df_plot = df.groupby(df.index.year).mean(numeric_only=True) - float(normal_values)
    label = station_name
    title = '%s anomaly [%s]' %(var, units)

    if global_dataset == 'HadCRUT':
        df1 = pd.read_csv(str(path.parent.absolute())+'\data\HadCRUT.5.1.0.0\HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv', sep=',', decimal='.', header=0)


    if window != None:
        df_plot = df_plot.rolling(window=window).mean()
        df1 = df1.rolling(window=window).mean()
        title = '%s anomaly [%s] %i-year moving average' %(var, units, window)


    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(df1.iloc[:,0], df1.iloc[:,1], color='k', lw=2, label=global_dataset)
    ax.fill_between(df1.iloc[:,0], df1.iloc[:,2], df1.iloc[:,3], color='grey', lw=2, label='%s 95%% confidence interval' %global_dataset, alpha=0.7)
    ax.plot(df_plot[var], color='blue', lw=1.5, label=label)
    ax.grid(color='k', linestyle='--')
    ax.axhline(y=0, color='k')
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylabel('Anomaly to 1961-1900 [ºC]', fontsize=16)

    plt.text(0.03, 0.96, 'Climate normal period: 1961-1990', fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.03, 0.93, 'Period with data: %i-%i' %(df.index.year.min(),df.index.year.max()), fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.96, 'Database: %s' %database, fontsize=12, transform=plt.gcf().transFigure)
    plt.text(0.77, 0.93, 'Location: %s' %station_name, fontsize=12, transform=plt.gcf().transFigure)
    fig.suptitle(title, fontsize=24)
    ax.legend(fontsize=14).set_visible(True)
    ax.margins(0.01, 0.05)
    plt.tight_layout()
    plt.savefig(filename,dpi=300)
