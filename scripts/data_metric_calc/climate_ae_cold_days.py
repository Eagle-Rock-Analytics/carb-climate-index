#!/usr/bin/env python
# coding: utf-8

### Cal-CRAI metric: # of chill hours 
# This script calculates extreme heat loss metric: 
# `change in average number of cold days` 
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
# ----------------------------------------------------------------------------------------------------------------------

## Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
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

def reproject_to_tracts(ds_delta, ca_boundaries, county):
    ca_boundaries = ca_boundaries.to_crs(crs=3310) # area-preserving CA Albers
    df = ds_delta.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.x,df.y))
    gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(ca_boundaries.crs)
    
    ca_boundaries = ca_boundaries.set_index(['GEOID'])
    
    clipped_gdf = gpd.sjoin_nearest(ca_boundaries, gdf, how='left')
    clipped_gdf = clipped_gdf.drop(['index_right'], axis=1)
    clipped_gdf = clipped_gdf[clipped_gdf["NAME"]==county]
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

def count_delta_extreme_cold_events(ds_hist, ds_wl):    
  
    # define the months over which we are going to 
    # determine the 2nd percentile temperature threshold
    # to define a cold spell
    months_to_measure = [m for m in np.arange(1,13,1)]
    
    sim_coord_dict = dict(zip(sims_wl,sims_hist))
    
    ds_hist = ds_hist.squeeze()
    ds_wl = ds_wl.squeeze()
    ds_template = ds_hist.isel(time=0, simulation=0).squeeze()
    # first set consistent coordinates
    ds_hist = ds_hist.sortby("simulation")
    ds_wl = ds_wl.rename({"all_sims" : "simulation"})
    ds_wl = ds_wl.sortby("simulation")
    ds_wl = ds_wl.assign_coords({'simulation': list(sim_coord_dict.values())})
    ds_wl = ds_wl.transpose("simulation","time","y","x")

    # compute 2nd percentile historical temperature
    thresh_ds = ds_hist.sel(
        time=ds_hist.time.dt.month.isin(months_to_measure)).chunk(
            dict(time=-1)).quantile(0.02, dim="time")
    
    # count total days < 2nd percentile in historical data and take annual average
    hist_count = xr.where(ds_hist < thresh_ds, x=1, y=0).groupby(
        "time.year").sum().mean(dim="year").mean(dim="simulation")
    
    # count total days < 2nd percentile in warming levels data and take annual average
    chronic_count = xr.where(ds_wl < thresh_ds, x=1, y=0).groupby(
        "time.year").sum().mean(dim="year").mean(dim="simulation")
    
    # get the delta signal
    delta_count = chronic_count - hist_count
    
    # nan out non-CA grid points
    delta_count = xr.where(np.isnan(ds_template), x=np.nan, y=delta_count)
    return delta_count

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

# Step 0: Load in shapefiles
# load in census tract shapefile
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/" 
ca_boundaries = gpd.read_file(census_shp_dir)
ca_counties_dir = "s3://ca-climate-index/0_map_data/ca_counties/"
ca_counties = gpd.read_file(ca_counties_dir)
ca_counties = ca_counties.to_crs(ca_boundaries.crs)
# clean up and convert to area-preserving CRS
ca_boundaries = ca_boundaries[["COUNTYFP","GEOID","geometry"]]
ca_boundaries = pd.merge(ca_boundaries,ca_counties[["COUNTYFP","NAME"]],on="COUNTYFP")
ca_boundaries = ca_boundaries.to_crs(crs=crs) 

# ----------------------------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# 30 year centered around 2.0C warming level (SSP3-7.0)
## Step 1a) Chronic data (2.0degC WL)
wl = warming_levels()
wl.wl_params.timescale = "daily"
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
selections.timescale = 'daily'
selections.variable = 'Air Temperature at 2m'
selections.area_subset = "states"
selections.cached_area = ["CA"]  
selections.scenario_historical = ['Historical Climate']
selections.time_slice = (1981, 2010)
selections.resolution = '3 km' ## 9km for testing on AE hub
selections.units = 'degF'
hist_ds = selections.retrieve()
hist_ds = hist_ds.sel(simulation=sims_hist)
    
# ----------------------------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# Difference between chronic (at 2.0degC warming level) and historical baseline (1981-2010)

# calculate delta
cold_delta_ds = count_delta_extreme_cold_events(
    ds_hist, ds
)
cold_delta_ds = ck.load(cold_delta_ds)
cold_delta_ds.name = "mean_change_cold_days" # assign name

# ----------------------------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# reproject   
df = reproject_to_tracts(cold_delta_ds, ca_boundaries, county)

# ----------------------------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
data_std = min_max_standardize(df, col='mean_change_cold_days')

# ----------------------------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
data_std = data_std.drop(columns=['geometry'])

# export
bucket_name = 'ca-climate-index'
directory = '3_fair_data/index_data'

metric_fname = 'climate_heat_cold_days_metric.csv'
data_std.to_csv(metric_fname)
upload_csv_aws([metric_fname], bucket_name, directory)