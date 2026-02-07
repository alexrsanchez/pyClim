##############################################################
#### Example script of the pyClim package ####################
#### Author: Alejandro Rodríguez Sánchez #####################
#### Contact: ars.rodriguezs@gmail.com #######################
##############################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import datetime as dt

from pandas import Grouper,DataFrame
import os
import glob

from scipy import stats, interpolate, signal

# Import pyclim
from pyclim_test import * 

#light_jet = cmap_map(lambda x: x*0.85, mpl.cm.jet)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None, N_colors=256):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=N_colors)
    return cmp


# Load the example data
path = '%s/data/' %os.getcwd()
csvfiles = glob.glob(os.path.join(path, 'example_data.txt'))
print(csvfiles)

# Create metadata
metadata = pd.DataFrame(['example','example',361.00,-361.00,461]).T
metadata.columns=['Estacion','Codigo','latitud_OK','longitud_OK','Altitud']
        
# Some useful variables
year_to_plot = 2025
climate_normal_period = [1991,2020]
variables = ['Tmin','Tmean','Tmax', 'Rainfall', 'WindSpeed']
database = 'Example'

# Mapping variables
units_list = {} #['ºC','ºC','ºC', 'm/s'] #,'mm']
units_list['Tmin'] = 'ºC'
units_list['Tmean'] = 'ºC'
units_list['Tmax'] = 'ºC'
units_list['Rainfall'] = 'mm'
units_list['WindSpeed'] = 'm/s'

wd_map = {'N':0, 'NNE':22.5, 'NE':45, 'ENE':67.5, 'E':90, 'ESE':112.5, 'SE':125, 'SSE':147.5, 'S':180, 'SSW':202.5, 'SW':225, 'WSW':247.5, 'W':270, 'WNW':292.5, 'NW':315, 'NNW':337.5}

colors_var = ['blue','black','red']
variables_anom = ['Tmin_anom','Tmax_anom','Tmean_anom'] #,'accumpcp_anom']
variables_names = ['temperatura mínima','temperatura máxima','temperatura media'] #,'precipitación acumulada']
colormap = matplotlib.cm.get_cmap('RdBu_r')

colors_anom_bars = ['#34b1eb','#eb4034']
levels_anom_bars = [0,1]
cmap_anom_bars = get_continuous_cmap(colors_anom_bars, levels_anom_bars, 2) # Solo funciona con colores HEX

    
#%%
for listanombres in ['example']: #,len(dfs)):

    if listanombres not in metadata.Codigo.unique():
        print('Código sin datos. Continuando con siguiente estación...')
        continue
    metadatos_sta = metadata[metadata.Codigo == listanombres]
    print(metadatos_sta)
    codigo_sta = metadatos_sta.Codigo.values[0]
    #nombres_mod[listanombres] = nombres_mod[listanombres].replace('/','-')
    station_name = str(metadatos_sta.Estacion.values[0])

    station_name = station_name.replace('/','-')
    print(station_name)
    plt.close('all')
    plotdir = os.path.join(os.getcwd(),'plots/%s/%s' %(database,listanombres))
    plotdir = plotdir.replace("\\","/")
    if os.path.isdir(plotdir) == False:
        try:
            os.mkdir(plotdir)
        except OSError:
            print ("Creation of the directory %s failed" % plotdir)
        else:
            print ("Successfully created the directory %s " % plotdir)


    # Read data
    input_file2 = [x for x in csvfiles if codigo_sta in x][0]    

    df1 = pd.read_csv(input_file2,sep = ";", decimal = ",", header=0, encoding='latin-1')

    df1 = df1.rename(columns={'fecha':'Fecha', 'Fecha_OK':'Fecha', 'indicativo': 'IdEstacion', 'provincia': 'IdProvincia', 'TempMedia':'Tmean', 'Precipitacion':'Rainfall', 'TempMin':'Tmin', 
                                'TempMax':'Tmax', 'VelViento':'WindSpeed', 'VelVientoMax':'WindMaxSpeed', 'DirViento': 'WindDir', 'DirVientoVelMax': 'WindGustDir', 'pres':'Pressure',
                                'HumedadMaxima':'RHmax', 'HumedadMinima':'RHmin'}, errors='ignore')


    # Correct values
    if str(df1['WindDir'].dtype) in ['object']:
        df1['WindDir'] = df1['WindDir'].replace(wd_map)
        
    if 'Rainfall' in df1.columns:
        df1['Rainfall'][df1.Rainfall=="Ip"]='0,05'
        df1['Rainfall'][df1.Rainfall=="Acum"]='-999'
        if any(df1.applymap(type).Rainfall==str) == True:
            df1['Rainfall'] = df1['Rainfall'].str.replace(',','.').astype(float)
    else:
        df1['Rainfall'] = np.nan

    if 'PePMon' in df1.columns:
        df1['PePMon'][df1.PePMon=="Ip"]='0,05'
        if any(df1.applymap(type).PePMon==str) == True:
            df1['PePMon'] = df1['PePMon'].str.replace(',','.').astype(float)

    # df1[df1==-999]=np.nan
    # for cols in df1.columns:
    df1[df1.columns].mask(df1[df1.columns].eq(0).all(axis=1))

    if 'Tmean' in df1.columns:
        if any(df1.applymap(type).Tmean==str) == True:
            df1['Tmean'].str.strip() #Deletes blank spaces
            df1['Tmean'] = df1['Tmean'].str.replace('"',"")
            df1['Tmean'] = df1['Tmean'].str.replace('NA',"-999")
            df1['Tmean'] = df1['Tmean'].str.replace(',',".")
            df1['Tmean'] = df1['Tmean'].astype(float)
        df1['Tmean'] = df1['Tmean'].replace(-999,np.nan)
    if 'Tmax' in df1.columns:
        if any(df1.applymap(type).Tmax==str) == True:
            df1['Tmax'].str.strip() #Deletes blank spaces
            df1['Tmax'] = df1['Tmax'].str.replace('"',"")
            df1['Tmax'] = df1['Tmax'].str.replace('NA',"-999")
            df1['Tmax'] = df1['Tmax'].str.replace(',',".")
            df1['Tmax'] = df1['Tmax'].astype(float)
        df1['Tmax'] = df1['Tmax'].replace(-999,np.nan)
    if 'Tmin' in df1.columns:
        if any(df1.applymap(type).Tmin==str) == True:
            df1['Tmin'].str.strip() #Deletes blank spaces
            df1['Tmin'] = df1['Tmin'].str.replace('"',"")
            df1['Tmin'] = df1['Tmin'].str.replace('NA',"-999")
            df1['Tmin'] = df1['Tmin'].str.replace(',',".")
            df1['Tmin'] = df1['Tmin'].astype(float)
        df1['Tmin'] = df1['Tmin'].replace(-999,np.nan)
    if 'WindSpeed' in df1.columns:
        if any(df1.applymap(type).WindSpeed==str) == True:
            df1['WindSpeed'].str.strip() #Deletes blank spaces
            df1['WindSpeed'] = df1['WindSpeed'].str.replace('"',"")
            df1['WindSpeed'] = df1['WindSpeed'].str.replace('NA',"-999")
            df1['WindSpeed'] = df1['WindSpeed'].str.replace(',',".")
            df1['WindSpeed'] = df1['WindSpeed'].astype(float)
        df1['WindSpeed'] = df1['WindSpeed'].replace(-999,np.nan)
    if 'WindDir' in df1.columns:
        if any(df1.applymap(type).WindDir==str) == True:
            df1['WindDir'].str.strip() #Deletes blank spaces
            df1['WindDir'] = df1['WindDir'].str.replace('"',"")
            df1['WindDir'] = df1['WindDir'].str.replace('NA',"-999")
            df1['WindDir'] = df1['WindDir'].str.replace(',',".")
            df1['WindDir'] = df1['WindDir'].astype(float)
        df1['WindDir'] = df1['WindDir'].replace(-999,np.nan)
    if 'WindMaxSpeed' in df1.columns:
        if any(df1.applymap(type).WindMaxSpeed==str) == True:
            df1['WindMaxSpeed'].str.strip() #Deletes blank spaces
            df1['WindMaxSpeed'] = df1['WindMaxSpeed'].str.replace('"',"")
            df1['WindMaxSpeed'] = df1['WindMaxSpeed'].str.replace('NA',"-999")
            df1['WindMaxSpeed'] = df1['WindMaxSpeed'].str.replace(',',".")
            df1['WindMaxSpeed'] = df1['WindMaxSpeed'].astype(float)
        df1['WindMaxSpeed'] = df1['WindMaxSpeed'].replace(-999,np.nan)
    if 'Dirrachas' in df1.columns:
        if any(df1.applymap(type).Dirrachas==str) == True:
            df1['Dirrachas'].str.strip() #Deletes blank spaces
            df1['Dirrachas'] = df1['Dirrachas'].str.replace('"',"")
            df1['Dirrachas'] = df1['Dirrachas'].str.replace('NA',"-999")
            df1['Dirrachas'] = df1['Dirrachas'].astype(float) 
        df1['Dirrachas'] = df1['Dirrachas'].replace(-999,np.nan)
    if 'Rainfall' in df1.columns:
        if any(df1.applymap(type).Rainfall==str) == True:
            df1['Rainfall'].str.strip() #Deletes blank spaces
            df1['Rainfall'] = df1['Rainfall'].str.replace('"',"")
            df1['Rainfall'] = df1['Rainfall'].str.replace('NA',"-999")
            df1['Rainfall'] = df1['Rainfall'].astype(float)
        df1['Rainfall'] = df1['Rainfall'].replace(-999,np.nan)
    if 'PePMon' in df1.columns:
        if any(df1.applymap(type).PePMon==str) == True:
            df1['PePMon'].str.strip() #Deletes blank spaces
            df1['PePMon'] = df1['PePMon'].str.replace('"',"")
            df1['PePMon'] = df1['PePMon'].str.replace('NA',"-999")
            df1['PePMon'] = df1['PePMon'].astype(float)
        df1['PePMon'] = df1['PePMon'].replace(-999,np.nan)
        

    # If most values are NaN values, skip datafile and continue with next
    if min(df1['Tmax'].isnull().sum(), df1['Tmean'].isnull().sum(), df1['Tmin'].isnull().sum()) >= 0.15*len(df1):
        print('skipping iteration...')
        continue  

    
    # amplitudes = df1.Tmax - df1.Tmin
    if set(['Tmin', 'Tmax']).issubset(df1.columns):
        df1['Amplitudes'] = df1.Tmax - df1.Tmin
    else:
        df1['Amplitudes'] = np.nan
        
        
    
    # Añado fechas a Campillos
    df1['Fecha'] = pd.date_range(start=dt.datetime(2010,9,1), periods=len(df1),freq='1D')
    df1 = df1.set_index('Fecha')


    date1 = df1.index[0] #.iloc[0,'Fecha_OK']
    date1_dt = date1
    date2 = df1.index[-1]
    date2_dt = date2

    ## PROVISIONAL
    if df1.index.year.max() < year_to_plot:
        continue
    # datehoy = date2
    date1 = df1.index[0]
    date2 = df1.index[-1] #['Fecha'][len(df1)-1]
    daytoday = int(date2_dt.strftime('%j'))

    df1['Day'] = df1.index.day
    df1['Month'] = df1.index.month
    df1['Year'] = df1.index.year
    yeartoday = int(df1.Year[-1])
    yearinicio = int(df1.Year[0])
    bisiestos=[2020,2024,2028,2032,2036,2040,2044,2048,2052,2056,2060]
    if yeartoday not in bisiestos:
        ndaysyear=365
    else:
        ndaysyear=366    
            
    df1['DayofYear'] = df1.index.dayofyear
    

    datetoday = date2

    df1['wateryear'] = df1.index.year.where(df1.index.month < 9, df1.index.year + 1)
    df1['Accumpcphidro'] = df1.groupby(df1.wateryear)['Rainfall'].cumsum()

    
    df1['Accumpcp'] = df1.groupby(df1.index.year)['Rainfall'].cumsum()
    df1['rain_episode'] = df1['Rainfall'].groupby((df1['Rainfall'] < 1).cumsum()).cumcount()
    df1['dry_episode'] = df1['Rainfall'].groupby((df1['Rainfall'] >= 1).cumsum()).cumcount()

    
    climate_vars = ['Tmax','Tmean','Tmin','Rainfall','Accumpcp','Accumpcphidro','WindSpeed','WindMaxSpeed']

    df1_complete = df1.reindex(pd.date_range('%i-01-01' %yearinicio, '%i-12-31' %yeartoday, freq='1D'))
    df1_complete['Day'] = df1_complete.index.day
    df1_complete['Month'] = df1_complete.index.month
    df1_complete['Year'] = df1_complete.index.year
    df1_complete['DayofYear'] = df1_complete.index.dayofyear

    # Pass quality control
    df1_complete = quality_control(df1_complete, ['Tmean', 'WindSpeed'], t_units='C', wind_units='km/h')
    #diff = df1_complete1[df1_complete.index.year == 2025]['Tmean'] - df1_complete[df1_complete.index.year == 2025]['Tmean']

    climate_df = compute_climate(df1_complete.loc[df1_complete.index.isin(df1.index),df1_complete.columns.isin(df1.columns)], climate_vars, [1991,2020], separate_df=False)
    climate_df_sep = compute_climate(df1_complete.loc[:,df1_complete.columns.isin(df1.columns)], climate_vars, [1991,2020])

    ## Create dataframe from first day of initial year to last day of current year
    #climate_df_sep = climate_df_sep.reindex(pd.date_range('%i-01-01' %yearinicio, '%i-12-31' %yeartoday, freq='1D'))

    
    ## Compute anomalies
    #for var in ['Tmax','Tmean','Tmin','Rainfall','Accumpcp','WindSpeed','WindMaxSpeed']:
    #    climate_df['%s_anom' %var] = climate_df[var] - climate_df['%s_median' %var]
    #    df1_complete['%s_anom' %var] = df1_complete[var] - climate_df['%s_median' %var]
            
    
    ### Anomalies
    plot_anomalies(df1_complete, 'Tmean', 'ºC', climate_normal_period, database, station_name, plotdir+'/daily_anomalies_Tmean.png', window=12, freq='1D')
    plot_anomalies(df1_complete, 'Tmean', 'ºC', climate_normal_period, database, station_name, plotdir+'/monthly_anomalies_Tmean.png', window=12, freq='1M')
    
    #### Accumulated anomalies
    plot_accumulated_anomalies(df1_complete, 'Tmean', 'ºC', 2025, climate_normal_period, database, station_name, plotdir+'/Tmean_accum_anoms_daily.png',freq='1D')
    plot_accumulated_anomalies(df1_complete, 'Tmean', 'ºC', 2025, climate_normal_period, database, station_name, plotdir+'/Tmean_accum_anoms_weekly.png',freq='1W')
    plot_accumulated_anomalies(df1_complete, 'Tmean', 'ºC', 2025, climate_normal_period, database, station_name, plotdir+'/Tmean_accum_anoms_monthly.png',freq='1M')    
            
    ndays = 365
    ndaysago = datetoday - dt.timedelta(days=ndays)

    # Plot data from a certain period versus the climatological normal
    records_df_allvars = pd.DataFrame()
    for i in range(len(sorted(list(set(df1_complete.columns) & set(variables)), key=lambda x: variables.index(x)))):
        variable = sorted(list(set(df1_complete.columns) & set(variables)), key=lambda x: variables.index(x))[i]
        units = units_list[variable]
        enddate = datetoday
        #plot data
        plot_data_vs_climate(df1_complete,climate_df_sep,variable,units,ndaysago,enddate,cmap_anom_bars,
                             database,climate_normal_period, station_name, plotdir+'/%speriodtimeseries_climatemedian19912020.png' %variable, kind='bar', fillcolor_gradient=False)

            
    multiyearrecords_df_allvars = pd.DataFrame()    # For saving multiple variable records' DataFrames

    varis = ['Tmax','Tmean','Rainfall']
    units_varis = []
    for i in range(len(varis)):
        units_varis.append(units_list[varis[i]])

        multiyearrecords_df = compute_daily_records(df1_complete, varis[i], df1_complete.index.year.unique()) # Compute records for variable
        multiyearrecords_df_allvars = pd.concat([multiyearrecords_df_allvars, multiyearrecords_df], axis=1)

    
    # Plot annual records
    plot_records_count(multiyearrecords_df_allvars, 'Tmean', database, station_name, plotdir+'/annual_records_Tmean.png', freq='day') # Plot number of days exceeding daily records

    # Multiyear records
    for i in range(len(varis)):
        plot_data_vs_climate_withrecords(df1_complete,climate_df_sep,multiyearrecords_df_allvars,varis[i],units_varis[i],ndaysago,enddate,cmap_anom_bars,
                                                database,climate_normal_period, station_name, plotdir+'/%stimeseries_climatemedian19912020_withrecords.png' %varis[i], kind='bar', fillcolor_gradient=False)

    plot_data_vs_climate_withrecords_multivar(df1_complete,climate_df_sep,multiyearrecords_df_allvars,varis,units_varis,ndaysago,enddate,cmap_anom_bars,database,climate_normal_period, station_name, plotdir+'/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords.png' %varis[i],
                                               kind='bar', fillcolor_gradient=False)
    plot_data_vs_climate_withrecords_multivar(df1_complete,climate_df_sep,multiyearrecords_df_allvars,varis,units_varis,ndaysago,enddate,cmap_anom_bars,database,climate_normal_period, station_name, plotdir+'/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords_std.png' %varis[i], 
                                              kind='bar', fillcolor_gradient=False, use_std=True)
    plot_data_vs_climate_withrecords_multivar(df1_complete,climate_df_sep,multiyearrecords_df_allvars,varis,units_varis,ndaysago,enddate,cmap_anom_bars,database,climate_normal_period, station_name, plotdir+'/multivar%s_Rainfall_timeseries_climatemedian19912020_withrecords_std_line.png' %varis[i], 
                                              kind='line', fillcolor_gradient=False, use_std=True)

    # Plot variable and anomalies
    varis = ['Tmin','Tmax','Tmean']
    units_varis = []
    for i in range(len(varis)):
        units_varis.append(units_list[varis[i]])

    # Plot variable and accumulated anomaly
    plot_data_and_accum_anoms(df1_complete,climate_df_sep,year_to_plot,varis,units_varis,cmap_anom_bars,database,climate_normal_period, station_name, plotdir, secondplot_type='accum', w=7)
    plot_data_and_accum_anoms(df1_complete,climate_df_sep,year_to_plot,varis,units_varis,cmap_anom_bars,database,climate_normal_period, station_name, plotdir, secondplot_type='moving', w=7)
    # Plot variable and accumulated mean or value
    plot_data_and_yearly_cycle(df1_complete,climate_df_sep,year_to_plot,varis,units_varis,cmap_anom_bars,database,climate_normal_period, station_name, plotdir, fillcolor_gradient=True)



    # Plot annual meteogram
    df1_complete['Temp'] = df1_complete['Tmean']
    climate_df_sep['Temp'] = climate_df_sep['Tmean_median']
    climate_df_sep['WindSpeed'] = climate_df_sep['WindSpeed_median']
    annual_meteogram(df1_complete,climate_df_sep, year_to_plot, climate_normal_period, database, station_name, plotdir+'/%i_meteogram.png' %year_to_plot)
#    df1_complete['Temp'] = get_yearly_cycle(df1_complete, climate_df_sep, ['Tmean'])['Tmean']
#    climate_df_sep['Temp_median'] = get_yearly_cycle(df1_complete, climate_df_sep, ['Tmean'])['Tmean_median']
    annual_meteogram(df1_complete,climate_df_sep, year_to_plot, climate_normal_period, database, station_name, plotdir+'/%i_meteogram_bars.png' %year_to_plot, plot_anoms=True)
    
    # Timeseries
    plot_timeseries(df1_complete, climate_df_sep, varis[-1], units_varis[-1], climate_normal_period, database, station_name, plotdir+'/%s_timeseries.png' %varis[-1], plot_MA=False, climate_stat='median', window=30)
    plot_timeseries(df1_complete, climate_df_sep, varis[-1], units_varis[-1], climate_normal_period, database, station_name, plotdir+'/%s_timeseries_MA.png' %varis[-1], plot_MA=True, climate_stat='median', window=365)

    ### Plot mean or median value of a certain period for every year with data
    for var in list(set(df1_complete.columns) & set(variables)):
        units = units_list[var]
        plot_periodaverages(df1_complete,climate_df_sep, varis[-1], units_varis[-1], dt.datetime(2025,6,1), dt.datetime(2025,9,1), station_name, database, plotdir, stat='median', window=10)
        

    ### Yearly cycles

    tmax_accum_anom = get_yearly_cycle(df1_complete, climate_df, varis) # Get yearly cycle of accumulated anomalies

    colores_calidos = ['#800008',"#B80101",'#ff949b']
    colores_frios = ["#2205A8","#1542F5","#7397fb"]
    #colores_calidos = ['#ff949b',"#B80101",'#800008']
    plot_yearly_cycles(tmax_accum_anom, 'Tmax', 'ºC', year_to_plot, climate_normal_period, database, station_name, colores_frios, plotdir+'/Tmax_yearlycycle_colorlowest.png', yearly_cycle=True, criterion='lowest')
    plot_yearly_cycles(tmax_accum_anom, 'Tmax', 'ºC', year_to_plot, climate_normal_period, database, station_name, colores_calidos, plotdir+'/Tmax_yearlycycle_colorhighest.png', yearly_cycle=True, criterion='highest')


    ### Seasonal plots      
    for var in sorted(list(set(df1_complete.columns) & set(variables)), key=lambda x: variables.index(x)):
        units = units_list[var]
        # Extreme values
        timeseries_extremevalues(df1_complete, var, units, climate_normal_period, database, station_name, plotdir+'/Annualextremevalues_%s_lines.png' %var, time_scale='Year')
        timeseries_extremevalues(df1_complete, var, units, climate_normal_period, database, station_name, plotdir+'/seasonalextremevalues_%s_lines.png' %var, time_scale='season')
        
        # Plot variables evolution in time
        plot_variable_trends(df1_complete, var, units, plotdir+'/%s_withmean.png' %var, database, station_name, averaging_period=5, grouping='year', grouping_stat='mean', rain_limit=1)
        plot_variable_trends(df1_complete, var, units, plotdir+'/%s_withmean_season.png' %var, database, station_name, averaging_period=5, grouping='season', grouping_stat='mean', rain_limit=1)
        plot_variable_trends(df1_complete, var, units, plotdir+'/%s_withmean_month.png' %var, database, station_name, averaging_period=5, grouping='month', grouping_stat='mean', rain_limit=1)


        # Exceedances of threshold values
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_annualexceedances_withmeans_2b.png' %var, threshold=20, time_scale='year', upwards=True, plot_means=True, averaging_period=10)
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_annualexceedances_b.png' %var, threshold=20, time_scale='year', upwards=True)
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_monthlyexceedances.png' %var, threshold=20, time_scale='month', upwards=True)
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_monthlyexceedances_withmeans.png' %var, threshold=20, time_scale='month', upwards=True, plot_means=True, averaging_period=10)
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_seasonalexceedances.png' %var, threshold=20, time_scale='season', upwards=True)
        compute_and_plot_exceedances(df1_complete, var, database, station_name, plotdir+'/%s_seasonalexceedances_withmeans.png' %var, threshold=20, time_scale='season', upwards=True, plot_means=True, averaging_period=10)
    
    ### WHEN DO SEASONS START?
    when_season_starts(df1_complete, climate_df_sep, 'Tmean', 'ºC', 2024,  climate_normal_period, database, station_name, plotdir+'/when_season_starts_2024_b.png')
    when_season_starts(df1_complete, climate_df_sep, 'Tmean', 'ºC', 2005,  climate_normal_period, database, station_name, plotdir+'/when_season_starts_2005_b.png')
    when_season_starts(df1_complete, climate_df_sep, 'Tmean', 'ºC', 2019,  climate_normal_period, database, station_name, plotdir+'/when_season_starts_2019_b.png')
    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  climate_normal_period, database, station_name, plotdir+'/when_season_starts_evol_refclimatenormal.png', plot_type='heatmap', grouping_freq=1)
    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  climate_normal_period, database, station_name, plotdir+'/when_season_starts_evol_refclimatenormal_lines.png', plot_type='line', grouping_freq=1)
    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  climate_normal_period, database, station_name, plotdir+'/when_season_starts_evol_refclimatenormal_both.png', plot_type='both', grouping_freq=1)
    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  climate_normal_period, database, station_name, plotdir+'/when_season_starts_evol_refclimatenormal_both_decades.png', plot_type='both', grouping_freq=10)
#    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  [1921,1950], database, station_name, plotdir+'/when_season_starts_evol_ref1921-1950_both_decades.png', plot_type='both', grouping_freq=10)
#    when_season_starts_evolution(df1_complete, climate_df_sep, 'Tmean', list(np.arange(2015,2025,1)),  [1920,1924], database, station_name, plotdir+'/when_season_starts_evol_ref1921-1950.png', climate_stat='median', grouping_freq=5)
