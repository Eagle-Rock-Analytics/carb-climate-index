#!/usr/bin/env python
# coding: utf-8

### Cal-CRAI metric: # of chill hours 
# This script calculates extreme heat loss metric: 
# `change in average number of seasonal chill hours` 
# from Cal-Adapt: Analytics Engine data. This script may be expanded or modified for inclusion in cae-notebooks in the future. 
#
# **Order of operations**:
# 1. Read data in
# 2. Calculate base function (FFWI, SPEI, warm nights, etc.)
# 3. Calculate chronic
# 4. Calculate delta signal
# 5. Reprojection to census tracts
# 6. Min-max standardization
# 7. Export data
# 
# **Runtime**: This notebook will take approximately 1 hour to run due to data size, warming levels, and reprojection steps. 
# 
# **References**:
# Chill hours:
# - 45 F and Under Model 
#     - 1 hour <= 45 F = 1.0 chill unit
# - 32F - 45F Model
#     - 1 hour between 32F and 45F =1.0 chill unit
# - Utah Model
#     - 1 hour below 34 degrees F = 0.0 chill unit 
#     - 1 hour 35-36 degrees F = 0.5 chill units
#     - 1 hour 37-48 degrees F = 1.0 chill units
#     - 1 hour 49-54 degrees F = 0.5 chill units
#     - 1 hour 55-60 degrees F = 0.0 chill units
#     - 1 hour 61-65 degrees F = -0.5 chill units
#     - 1 hour >65 degrees F = -1.0 chill units
# 
# From: Chilling Accumulation: its Importance and Estimation; David H. Byrne and Terry Bacon, Dept. Of Horticultural Sciences, Texas A&M University, College Station, TX
# ----------------------------------------------------------------------------------------------------------------------

## Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
import pandas as pd
import numpy as np
import geopandas as gpd

import pyproj
import rioxarray as rio
import xarray as xr

## projection information
import cartopy.crs as ccrs
crs = ccrs.LambertConformal(
    central_longitude=-70, 
    central_latitude=38, 
    false_easting=0.0, 
    false_northing=0.0,  
    standard_parallels=[30, 60], 
    globe=None, 
    # cutoff=-30
)

# ----------------------------------------------------------------------------------------------------------------------
# Helpful function set-up
sim_name_dict = {  
    'WRF_CNRM-ESM2-1_r1i1p1f2_Historical + SSP 3-7.0 -- Business as Usual' :
    'CNRM-ESM2-1',
    'WRF_EC-Earth3-Veg_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
    'EC-Earth3-Veg',
    'WRF_CESM2_r11i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
    'CESM2',
    'WRF_FGOALS-g3_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
    'FGOALS-g3'
}

def reproject_to_tracts(ds_delta, ca_boundaries):
    df = ds_delta.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x,df.y))
    gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(ca_boundaries.crs)
    
    ca_boundaries = ca_boundaries.set_index(['GEOID'])
    
    clipped_gdf = gpd.sjoin_nearest(ca_boundaries, gdf, how='left')
    clipped_gdf = clipped_gdf.drop(['index_right'], axis=1)
    ### some coastal tracts do not contain any land grid cells ###
    ### due to the WRF's underlying surface type for a given grid cell. ###
    
    # aggregate the gridded data to the tract level
    clipped_gdf_diss = clipped_gdf.reset_index().dissolve(
        by='GEOID', aggfunc='mean')
    clipped_gdf_diss = clipped_gdf_diss.rename(
        columns={f"{ds_delta.name}_right":
                 ds_delta.name}
    )
    
    # separate tracts with data from tracts without data
    clipped_gdf_nan = clipped_gdf_diss[np.isnan(
        clipped_gdf_diss[ds_delta.name]
    )]
    clipped_gdf_nan = clipped_gdf_nan[["geometry",ds_delta.name]]
    clipped_gdf_valid = clipped_gdf_diss[~np.isnan(
        clipped_gdf_diss[ds_delta.name]
    )]
    clipped_gdf_valid = clipped_gdf_valid[["geometry",ds_delta.name]]

    # compute the centroid of each tract
    clipped_gdf_nan["centroid"] = clipped_gdf_nan.centroid
    clipped_gdf_nan = clipped_gdf_nan.set_geometry("centroid")
    clipped_gdf_valid["centroid"] = clipped_gdf_valid.centroid
    clipped_gdf_valid = clipped_gdf_valid.set_geometry("centroid")
    
    # fill in missing tracts with values from the closest tract
    # in terms of distance between the tract centroids
    clipped_gdf_filled = clipped_gdf_nan.sjoin_nearest(clipped_gdf_valid, how='left')
    clipped_gdf_filled = clipped_gdf_filled[["geometry_left",f"{ds_delta.name}_right"]]
    clipped_gdf_filled = clipped_gdf_filled.rename(columns={
        "geometry_left":"geometry", f"{ds_delta.name}_right":ds_delta.name
    })
    clipped_gdf_valid = clipped_gdf_valid.drop(columns="centroid")
 
    # concatenate filled-in tracts with the original tract which had data
    gdf_all_tracts = pd.concat([clipped_gdf_valid,clipped_gdf_filled])

    return gdf_all_tracts

def min_max_standardize(df, col):
    '''
    Calculates min and max values for specified columns, then calculates
    min-max standardized values.

    Parameters
    ----------
    df: DataFrame
        Input dataframe   
    cols_to_run_on: list
        List of columns to calculate min, max, and standardize
    '''
    max_value = df[col].max()
    min_value = df[col].min()

    # Get min-max values, standardize, and add columns to df
    prefix = col # Extracting the prefix from the column name
    df[f'{prefix}_min'] = min_value
    df[f'{prefix}_max'] = max_value
    df[f'{prefix}_min_max_standardized'] = ((df[col] - min_value) / (max_value - min_value))

    # checker to make sure new min_max column values arent < 0 > 1
    df[f'{prefix}_min_max_standardized'].loc[df[f'{prefix}_min_max_standardized'] < 0] = 0
    df[f'{prefix}_min_max_standardized'].loc[df[f'{prefix}_min_max_standardized'] > 1] = 1

    # Drop the original columns -- we want to keep as a check
    # df = df.drop(columns=[col])
     
    return df

# ----------------------------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# 30 year centered around 2.0C warming level (SSP3-7.0)
# Historical baseline 1981-2010 (Historical Climate)

## Step 1a) Chronic data (2.0degC WL)
wl = warming_levels()
wl.wl_params.timescale = "hourly"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable = "Air Temperature at 2m"
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "degF"
wl.wl_params.resolution = "3 km" # 9km for testing on AE hub
wl.wl_params.anom = "No"
wl.calculate()
ds = wl.sliced_data["2.0"] # grab 2.0 degC data
ds = ds.sel(all_sims = list(sim_name_dict.keys()))
ds = add_dummy_time_to_wl(ds) # add time dimension back in, as this is removed by WL and is required for xclim functionality


## Step 1b) Historical baseline data (1981-2010)
selections = ck.Select()
selections.area_average = 'No'
selections.timescale = 'hourly'
selections.variable = 'Air Temperature at 2m'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.simulation = list(sim_name_dict.values())
selections.time_slice = (1981, 2010)
selections.resolution = '3 km' ## 9km for testing on AE hub
selections.units = 'degC'
hist_ds = selections.retrieve()

# ----------------------------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# Difference between chronic (at 2.0degC warming level) and historical baseline (1981-2010)
# calculate metric -- subset for Nov 1 - Feb 28/29 (excluding March 1)
winter_months = [11, 12, 1, 2]
ds_winter = ds.isel(time=ds.time.dt.month.isin(winter_months))
ds_winter_hist = hist_ds.isel(time=hist_ds.time.dt.month.isin(winter_months))

# using 45 and under model
chill_threshold = 45
ds_winter_proj = (ds_winter <= chill_threshold).groupby('time.year').sum('time')
ds_winter_proj = ds_winter_proj.mean(dim='all_sims').mean(dim='year').squeeze()

ds_winter_hist = (ds_winter_hist <= chill_threshold).groupby('time.year').sum('time')
ds_winter_hist = ds_winter_hist.mean(dim='simulation').mean(dim='year').squeeze()

# calculate delta
ds_delta = ds_winter_proj - ds_winter_hist
ds_delta.name = "change_chill_hours" # assign name

# ----------------------------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# reproject
chill_df = reproject_to_tracts(ds_delta, ca_boundaries)

# ----------------------------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
data_std = min_max_standardize(chill_df, cols_to_run_on=['change_chill_hours'])

# ----------------------------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
data_std = data_std.drop(columns=['geometry'])

# export
data_std.to_csv('climate_heat_chill_hours_metric.csv')