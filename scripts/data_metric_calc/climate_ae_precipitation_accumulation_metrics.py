#!/usr/bin/env python
# coding: utf-8

### Absolute change in 99th percentile 1-day accumulated precipitation
# This script calculates the in-land flooding exposure metric:
# `change in 99th percentile 1-day accumulated precipitation`
# from Cal-Adapt: Analytics Engine data. This script may be adapted for inclusion in cae-notebooks in the future. 
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
# **Runtime**: This script must be run via pcluster due to the size and complexity of the data. 
# ----------------------------------------------------------------------------------------------------------------------
##  Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
import pandas as pd
import numpy as np
import geopandas as gpd

import pyproj
import rioxarray as rio
import xarray as xr
from bokeh.models import HoverTool
import os
import sys

import s3fs
import boto3
sys.path.append(os.path.expanduser('../../'))
from scripts.utils.file_helpers import upload_csv_aws

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
## Helpful function set-up

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

# ----------------------------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# 30 year centered around 2.0C warming level (SSP3-7.0)
# Historical baseline 1981-2010 (Historical Climate)

## Step 1a) Chronic data (2.0degC WL)
# retrieve 2 deg C precipitation total data
wl = warming_levels()
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable = "Precipitation (total)"
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "mm"
wl.wl_params.resolution = "3 km"
wl.wl_params.anom = "No"
wl.calculate()
ds = wl.sliced_data["2.0"] # grab 2.0 degC data
ds = ds.sel(all_sims = list(sim_name_dict.keys()))
total_precip = add_dummy_time_to_wl(ds)

# retrieve 2 deg C snowfall (snow and ice) data
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable = "Snowfall"
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "mm"
wl.wl_params.resolution = "3 km"
wl.calculate()
ds = wl.sliced_data["2.0"] # grab 2.0 degC data
ds = ds.sel(all_sims = list(sim_name_dict.keys()))
total_snowfall = add_dummy_time_to_wl(ds)

## Step 1b: Retrieve historical baseline data (1981-2010)
# precip
selections = ck.Select()
selections.area_average = 'No'
selections.timescale = 'daily'
selections.variable = 'Precipitation (total)'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.time_slice = (1981, 2010)
selections.resolution = '3 km'
selections.units = 'mm'
hist_precip_ds = selections.retrieve()
hist_precip_ds = hist_precip_ds.sel(simulation=sims_hist)

# Snowfall (snow and ice)
selections.area_average = 'No'
selections.timescale = 'daily'
selections.variable = 'Snowfall'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.time_slice = (1981, 2010)
selections.resolution = '3 km'
selections.units = 'mm'
hist_snow_ds = selections.retrieve()
hist_snow_ds = hist_snow_ds.sel(simulation=sims_hist)

# ----------------------------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# remove snow from precip
rain_wl = total_precip - total_snowfall
rain_wl = rain_wl.clip(min=0.1)
rain_hist = hist_precip_ds - hist_snow_ds
rain_hist = rain_hist.clip(min=0.1)

# remove leap days from historical data
rain_hist = rain_hist.sel(time=~((rain_hist.time.dt.month == 2) & (rain_hist.time.dt.day == 29)))

# pool the data first
hist_pool = rain_hist.stack(index=['simulation', 'time']).squeeze()
wl_pool = rain_wl.stack(index=['all_sims', 'time']).squeeze()

hist_wrf_pool_perc = hist_pool.chunk(
    dict(index=-1)).quantile([.99],
    dim='index').compute().squeeze()

wl_wrf_pool_perc = wl_pool.chunk(
    dict(index=-1)).quantile([.99],
    dim='index').compute().squeeze()

# calculate delta signal from percentiles
delta_wrf_pool_perc = (wl_wrf_pool_perc - hist_wrf_pool_perc)
# absolute change in 99th percentile, data pooled

# rename metric to be friendly for our remaining process
delta_wrf_pool_perc.name = "precip_99percentile"

# ----------------------------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# load in census tract shapefile
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/" 
ca_boundaries = gpd.read_file(census_shp_dir)
# convert to area-preserving CRS
ca_boundaries = ca_boundaries.to_crs(crs=3310)
rain_df = reproject_to_tracts(delta_wrf_pool_perc, ca_boundaries)

# ----------------------------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
rain_std = min_max_standardize(rain_df, col=rain_delta_ds.name)

# ----------------------------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# data will be exported via pcluster run

# clean up dataframes prior to export
rain_std = rain_std.drop(columns=['geometry'])

# export
bucket_name = 'ca-climate-index'
directory = '3_fair_data/index_data'

precip_name = 'climate_flood_exposure_precipitation_metric.csv'
rain_std.to_csv(precip_name)
upload_csv_aws([precip_name], bucket_name, directory)