How to use
==========

Introduction
------------
pyClim is a Python package composed of several Python functions that allows the user to check the quality of their climatological data; to compute climatological normal values; and to plot their data in a simple way -with no need of programming- with the purpose of allowing all kinds of users (from beginners to experienced ones) to extract valuable information from their data in a visual way.

Basic Usage
-----------

Performing a quality control on your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform a quality control on your data, and remove suspicious data if any, use the :func:`pyclim.quality_control` function. This function needs as a mandatory argument a pandas DataFrame containing the data you want to perform the quality control on, and as optional arguments:

 * the units of your data,
 * the maximum and minimum allowed values for temperature and wind speed (by default, maximum allowed temperature is 50ºC, minimum allowed temperature is -50ºC, and maximum allowed wind speed is 200 km/h).

These two arguments allow the user to refine the quality control procedure, in case that they have knowledge about the climatology of the data.

After removing the data above (below) the maximum (minimum) allowed values, the function removes values which exceeds 4 times the standard deviation of the climatological values.

In the example below, a quality control is performed on a set of data (df), for the variables "Tmean" and "WindSpeed" using the default limit values:

.. code:: python

    cured_df = quality_control(
        df, ["Tmean", "WindSpeed"], t_units="C", wind_units="km/h"
    )


Computing the climatological normal values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For several functions within pyClim, it is useful to have the climatological normal values in another separate DataFrame. The :func:`pyclim.compute_climate` function allows the user to do that, given that they provide a list with the initial and end years of the reference period to be used for the computation of climatological normal values.
The code of the example below extracts the 1991-2020 climatological normal values for the variables "Tmean", "Tmax" and "Tmin" of the "df" DataFrame:

.. code:: python

    climate_vars = ["Tmean", "Tmax", "Tmin"]
    climate_df = compute_climate(df, climate_vars, [1991, 2020])
