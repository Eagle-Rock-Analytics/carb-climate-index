#!/usr/bin/env python
# coding: utf-8

## Cal-CRAI metric: Fosberg Fire Weather Index 
# This script calculates the wildfire metric:
# `change in median annual # of days with FFWI greater than 50` 
# from the Cal-Adapt: Analytics Engine. This script may be expanded upon for inclusion in cae-notebooks in the future. 
# 
# **Order of operations:**
# 1. Read data in
# 2. Calculate base function (FFWI, SPEI, warm nights, etc.)
# 3. Calculate chronic
# 4. Calculate delta signal
# 5. Reprojection to census tracts
# 6. Min-max standardization
# 7. Export data
# 
# **Runtime**: This notebook takes approximately ~1 hours to run due to data size, warming levels, and reprojection steps.
# 
## Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
from climakitae.tools.indices import fosberg_fire_index
import pandas as pd
import numpy as np
import geopandas as gpd

import pyproj
import rioxarray as rio
import xarray as xr

# projection information
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

# -------------------------------------------------------------------------------------------------
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
    # this step takes about 12 minutes with 3km data (~1 min with 9km data)
    df = ds_delta.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x,df.y))
    gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(ca_boundaries.crs)

    # clipped_gdf = clipped_gdf.set_index(['USCB_GEOID'])
    ca_boundaries = ca_boundaries.set_index(['USCB_GEOID'])
    clipped_gdf = gpd.sjoin_nearest(ca_boundaries, gdf, how='left')
    clipped_gdf = clipped_gdf[["geometry",ds_delta.name]]

    diss_gdf = clipped_gdf.reset_index().dissolve(by='USCB_GEOID', aggfunc='mean')
    return diss_gdf

# -------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# * 30 year centered around 2.0C warming level (SSP3-7.0)
# * Historical baseline 1981-2010 (Historical Climate) 

## Step 1a) Chronic data (2.0°C WL)
wl = warming_levels()

## FFWI options
wl.wl_params.variable_type = 'Derived Index'
wl.wl_params.variable = 'Fosberg fire weather index'
wl.wl_params.timescale = "hourly"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.resolution = '3 km' # must test in 45 km on Hub
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "degF"
wl.wl_params.anom = "No"
wl.calculate()
ds_f = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_f = add_dummy_time_to_wl(ds_f) # add time dimension back in, as this is removed by WL and is required for xclim functionality

## In case FFWI needs to be manually calculated -- testing via FFWI retrieve first
## air temperature
# wl.wl_params.timescale = "hourly"
# wl.wl_params.downscaling_method = "Dynamical"
# wl.wl_params.resolution = '45 km'
# wl.wl_params.variable = 'Air Temperature at 2m'
# wl.wl_params.area_subset = "states"
# wl.wl_params.cached_area = ["CA"]
# wl.wl_params.warming_levels = ["2.0"]
# wl.wl_params.units = "degF"
# wl.wl_params.anom = "No"
# wl.wl_params.load_data = False
# wl.calculate()
# ds_T = wl.sliced_data["2.0"] # grab 2.0 degC data
# ds_T = add_dummy_time_to_wl(ds_T) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ck.export(ds_T, 'ds_T', 'NetCDF') # exports to local file tree on left

## relative humidity 
# wl.wl_params.variable = 'Relative humidity'
# wl.wl_params.units = "[0 to 100]"
# wl.calculate()
# ds_RH = wl.sliced_data["2.0"] # grab 2.0 degC data
# ds_RH = add_dummy_time_to_wl(ds_RH) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ds_RH
# ck.export(ds_RH, 'ds_RH', 'NetCDF')

## wind speed 
# wl.wl_params.variable = 'Wind speed at 10m'
# wl.wl_params.units = "mph"
# wl.calculate()
# ds_WS = wl.sliced_data["2.0"] # grab 2.0 degC data
# ds_WS = add_dummy_time_to_wl(ds_WS) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ck.export(ds_WS, 'ds_WS', 'NetCDF')

## Calculate FFWI with manual inputs
# ds_ffwi = fosberg_fire_index(t2_F=ds_T, rh_percent=ds_RH, windspeed_mph=ds_WS)

## Step 1b) Historical baseline data (1981-2010)
selections = ck.Select()
selections.area_average = 'No'
selections.timescale = 'hourly'
selections.variable_type = 'Derived Index'
selections.variable = 'Fosberg fire weather index'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.simulation = list(sim_name_dict.values())
selections.time_slice = (1981, 2010)
selections.resolution = '3 km' ## 45km for testing on AE hub
selections.units = 'degF'
hist_ds = selections.retrieve()

# -------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# calculate metric
ffwi_threshold = 50
ds_ffwi_f = (ds_f >= ffwi_threshold).groupby('time.year').sum('time', min_count=1)
ds_ffwi_h = (hist_ds >= ffwi_threshold).groupby('time.year').sum('time', min_count=1)

# Difference between chronic (at 2.0°C warming level) and historical baseline (1981-2010)
# handle different dimensions first

ds_ffwi_chronic_avg = ds_ffwi_chronic.mean(dim='year')
ds_ffwi_chronic_avg.name = "change_ffwi_days" # assign name so it can convert to pd.DataFrame


# -------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# read in CA census tiger file -- not working from s3 link, uploading manually to keep testing
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
# census_shp_dir = "tl_2021_06_tract.shp"
ca_boundaries = gpd.read_file(census_shp_dir)

# # need to rename columns so we don't have any duplicates in the final geodatabase
column_names = ca_boundaries.columns
new_column_names = ["USCB_"+column for column in column_names if column != "geometry"]
ca_boundaries = ca_boundaries.rename(columns=dict(zip(column_names, new_column_names)))
ca_boundaries = ca_boundaries.to_crs(crs=3857) 

df = ds_ffwi_chronic_avg.to_dataframe().reset_index()

# this step takes about 12 minutes with 3km data (~1 min with 9km data)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x,df.y))
gdf = gdf.set_crs(crs)
gdf = gdf.to_crs(ca_boundaries.crs)

# clipped_gdf = clipped_gdf.set_index(['USCB_GEOID'])
ca_boundaries = ca_boundaries.set_index(['USCB_GEOID'])
clipped_gdf = gpd.sjoin_nearest(ca_boundaries, gdf, how='left')
clipped_gdf = clipped_gdf[["geometry","change_ffwi_days"]]

diss_gdf = clipped_gdf.reset_index().dissolve(by='USCB_GEOID', aggfunc='mean')

# -------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
def min_max_standardize(df, cols_to_run_on):
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
    for col in cols_to_run_on:
        max_value = df[col].max()
        min_value = df[col].min()

        # Get min-max values, standardize, and add columns to df
        prefix = col # Extracting the prefix from the column name
        df[f'{prefix}_min'] = min_value
        df[f'{prefix}_max'] = max_value
        df[f'{prefix}_min_max_standardized'] = ((df[col] - min_value) / (max_value - min_value))
        
        # note to add checker to make sure new min_max column values arent < 0 >
        
        # Drop the original columns
        df.drop(columns=[col], inplace=True)
     
    return df

data_std = min_max_standardize(diss_gdf, cols_to_run_on=['change_ffwi_days'])

# -------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
data_std = data_std.drop(columns=['geometry'])

# export
data_std.to_csv('climate_wildfire_ffwi_metric.csv')