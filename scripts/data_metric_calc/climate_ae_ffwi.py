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
from climakitae.core.data_interface import DataParameters
import pandas as pd
import numpy as np
import geopandas as gpd

import os
import sys
import s3fs
import boto3
sys.path.append(os.path.expanduser('../../'))
from scripts.utils.file_helpers import upload_csv_aws

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
sims_wl = [
    'WRF_MPI-ESM1-2-HR_r3i1p1f1_Historical + SSP 3-7.0 -- Business as Usual',
    'WRF_MIROC6_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual',
    'WRF_EC-Earth3_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual',
    'WRF_TaiESM1_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual',
]
sims_hist = [
    'WRF_MPI-ESM1-2-HR_r3i1p1f1',
    'WRF_MIROC6_r1i1p1f1', 
    'WRF_EC-Earth3_r1i1p1f1',
    'WRF_TaiESM1_r1i1p1f1', 
]

sim_name_dict = dict(zip(sims_wl,sims_hist)) 

def reproject_to_tracts(ds_delta, ca_boundaries):
    df = ds_delta.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x,df.y))
    gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(ca_boundaries.crs)
    
    ca_boundaries = ca_boundaries.set_index(['GEOID'])
    
    clipped_gdf = gpd.sjoin_nearest(ca_boundaries, gdf, how='left')
    clipped_gdf = clipped_gdf.drop(['index_right'], axis=1)
    clipped_gdf = clipped_gdf.reset_index()[
        ["GEOID",f"{ds_delta.name}","geometry"]]
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

# -------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# * 30 year centered around 2.0C warming level (SSP3-7.0)
# * Historical baseline 1981-2010 (Historical Climate) 

area_subset = "CA counties"
cached_area = "Nevada County"
res = "9 km"
## Step 1a) Chronic data (2.0°C WL)
wl = warming_levels()

## FFWI options
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.timescale = "hourly"
wl.wl_params.variable_type = 'Derived Index'
wl.wl_params.variable = 'Fosberg fire weather index'
wl.wl_params.resolution = res # must test in 45 km on Hub
wl.wl_params.area_subset = area_subset
wl.wl_params.cached_area = [cached_area]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.anom = "No"
wl.calculate()
ds_f = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_f = ds_f.sel(all_sims = list(sim_name_dict.keys()))
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
# ds_T = ds_T.sel(all_sims = list(sim_name_dict.keys()))
# ds_T = add_dummy_time_to_wl(ds_T) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ck.export(ds_T, 'ds_T', 'NetCDF') # exports to local file tree on left

## relative humidity 
# wl.wl_params.variable = 'Relative humidity'
# wl.wl_params.units = "[0 to 100]"
# wl.calculate()
# ds_RH = wl.sliced_data["2.0"] # grab 2.0 degC data
# ds_RH = ds_RH.sel(all_sims = list(sim_name_dict.keys()))
# ds_RH = add_dummy_time_to_wl(ds_RH) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ds_RH
# ck.export(ds_RH, 'ds_RH', 'NetCDF')

## wind speed 
# wl.wl_params.variable = 'Wind speed at 10m'
# wl.wl_params.units = "mph"
# wl.calculate()
# ds_WS = wl.sliced_data["2.0"] # grab 2.0 degC data
# ds_WS = ds_WS.sel(all_sims = list(sim_name_dict.keys()))
# ds_WS = add_dummy_time_to_wl(ds_WS) # add time dimension back in, as this is removed by WL and is required for xclim functionality
# ck.export(ds_WS, 'ds_WS', 'NetCDF')

## Calculate FFWI with manual inputs
# ds_ffwi = fosberg_fire_index(t2_F=ds_T, rh_percent=ds_RH, windspeed_mph=ds_WS)

## Step 1b) Historical baseline data (1981-2010)
selections = DataParameters()
selections.area_average = 'No'
selections.timescale = 'hourly'
selections.variable_type = 'Derived Index'
selections.variable = 'Fosberg fire weather index'
selections.area_subset = area_subset
selections.cached_area = cached_area
selections.scenario_historical = ['Historical Climate']
selections.time_slice = (1981, 2010)
selections.resolution = res ## 45km for testing on AE hub
hist_ds = selections.retrieve()
hist_ds = hist_ds.sel(simulation = sims_hist)

# -------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# calculate metric
ffwi_threshold = 50
ds_ffwi_f = (ds_f >= ffwi_threshold).groupby('time.year').sum('time', min_count=1)
ds_ffwi_h = (hist_ds >= ffwi_threshold).groupby('time.year').sum('time', min_count=1)

# Difference between chronic (at 2.0°C warming level) and historical baseline (1981-2010)
ds_delta = ds_ffwi_f - ds_ffwi_h
ds_delta.name = "change_ffwi_days" # assign name so it can convert to pd.DataFrame

# -------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# reproject
# load in census tract shapefile
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/" 
ca_boundaries = gpd.read_file(census_shp_dir)
# convert to area-preserving CRS
ca_boundaries = ca_boundaries.to_crs(crs=3310)
ffwi_df = reproject_to_tracts(ds_delta, ca_boundaries)

# -------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
data_std = min_max_standardize(ffwi_df, cols_to_run_on=['change_ffwi_days'])

# -------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
data_std = data_std.drop(columns=['geometry'])

# export
bucket_name = 'ca-climate-index'
directory = '3_fair_data/index_data'

metric_fname = 'climate_wildfire_ffwi_metric.csv'
data_std.to_csv(metric_fname)
upload_csv_aws([metric_fname], bucket_name, directory)
