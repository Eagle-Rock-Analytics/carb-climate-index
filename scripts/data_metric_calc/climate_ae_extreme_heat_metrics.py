#!/usr/bin/env python
# coding: utf-8

### Mean change in annual extreme heat day and warm nights
# This script calculates the extreme heat exposure metrics:
# `mean change in annual extreme heat day` and `mean change in annual warm nights` 
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

def count_delta_extreme_heat_events(ds_hist,ds_wl):    
    sim_coord_dict = {
        'WRF_CESM2_r11i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
        'WRF_CESM2_r11i1p1f1',
        'WRF_CNRM-ESM2-1_r1i1p1f2_Historical + SSP 3-7.0 -- Business as Usual' :
        'WRF_CNRM-ESM2-1_r1i1p1f2',
        'WRF_EC-Earth3-Veg_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
        'WRF_EC-Earth3-Veg_r1i1p1f1',
        'WRF_FGOALS-g3_r1i1p1f1_Historical + SSP 3-7.0 -- Business as Usual' :
        'WRF_FGOALS-g3_r1i1p1f1'
    }                      
    
    ds_hist = ds_hist.squeeze()
    ds_wl = ds_wl.squeeze()
    ds_template = ds_hist.isel(time=0, simulation=0).squeeze()
    
    # first set consistent coordinates
    ds_hist = ds_hist.sortby("simulation")
    ds_wl = ds_wl.rename({"all_sims" : "simulation"})
    ds_wl = ds_wl.sortby("simulation")
    ds_wl = ds_wl.assign_coords({'simulation': list(sim_coord_dict.values())})
    ds_wl = ds_wl.transpose("simulation","time","y","x")

    # compute 98th percentile historical
    thresh_ds = ds_hist.chunk(dict(time=-1)).quantile(0.98, dim="time")
    # count total days > 98th percentile in historical data and take annual average
    hist_count = xr.where(ds_hist > thresh_ds, x=1, y=0).groupby(
        "time.year").sum().mean(dim="year").mean(dim="simulation")
    # count total days > 98th percentile in warming levels data and take annual average
    chronic_count = xr.where(ds_wl > thresh_ds, x=1, y=0).groupby(
        "time.year").sum().mean(dim="year").mean(dim="simulation")
    # get the delta signal
    delta_count = chronic_count - hist_count
    # nan out non-CA grid points
    delta_count = xr.where(np.isnan(ds_template), x=np.nan, y=delta_count)
    return delta_count

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

# ----------------------------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# 30 year centered around 2.0C warming level (SSP3-7.0)
# Historical baseline 1981-2010 (Historical Climate)

## Step 1a) Chronic data (2.0degC WL)
# max air temperature
wl = warming_levels()
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable = "Maximum air temperature at 2m"
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "degC"
wl.wl_params.resolution = "3 km" ## 9km for testing on AE hub
wl.wl_params.anom = "No"
wl.calculate()
ds = wl.sliced_data["2.0"] # grab 2.0 degC data
ds = ds.sel(all_sims = list(sim_name_dict.keys()))
wl_max_ds = add_dummy_time_to_wl(ds) # add time dimension back in, as this is removed by WL and is required for xclim functionality

# min air temperature
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable = "Minimum air temperature at 2m"
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.units = "degC"
wl.wl_params.resolution = "3 km" ## 9km for testing on AE hub
wl.wl_params.anom = "No"
wl.calculate()
ds = wl.sliced_data["2.0"] # grab 2.0 degC data
ds = ds.sel(all_sims = list(sim_name_dict.keys()))
wl_min_ds = add_dummy_time_to_wl(ds) # add time dimension back in, as this is removed by WL and is required for xclim functionality

## Step 1b: Historical baseline data (1981-2010)
# max air temperature
selections = ck.Select()
selections.area_average = 'No'
selections.timescale = 'daily'
selections.variable = 'Maximum air temperature at 2m'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.simulation = list(sim_name_dict.values())
selections.time_slice = (1981, 2010)
selections.resolution = '3 km' ## 9km for testing on AE hub
selections.units = 'degC'
hist_max_ds = selections.retrieve()

# min air temperature
selections.area_average = 'No'
selections.timescale = 'daily'
selections.variable = 'Minimum air temperature at 2m'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical = ['Historical Climate']
selections.simulation = list(sim_name_dict.values())
selections.time_slice = (1981, 2010)
selections.resolution = '3 km' ## 9km for testing on AE hub
selections.units = 'degC'
hist_min_ds = selections.retrieve()

# ----------------------------------------------------------------------------------------------------------------------
## Step 2: Calculate delta signal
# Difference between chronic (at 2.0degC warming level) and historical baseline (1981-2010)

# get change in # of hot days
hd_delta_ds = count_delta_extreme_heat_events(
    hist_max_ds, wl_max_ds
)
hd_delta_ds = ck.load(hd_delta_ds)
hd_delta_ds.name = "mean_change_annual_heat_days"

# get change in # of warm nights
wn_delta_ds = count_delta_extreme_heat_events(
    hist_min_ds, wl_min_ds
)
wn_delta_ds = ck.load(wn_delta_ds)
wn_delta_ds.name = "mean_change_annual_warm_nights"

# ----------------------------------------------------------------------------------------------------------------------
## Step 3: Reproject data to census tract projection
# read in CA census tiger file -- not working from s3 link, uploading manually to keep testing
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
#census_shp_dir = "tl_2021_06_tract.shp"
ca_boundaries = gpd.read_file(census_shp_dir)

# # need to rename columns so we don't have any duplicates in the final geodatabase
column_names = ca_boundaries.columns
new_column_names = ["USCB_"+column for column in column_names if column != "geometry"]
ca_boundaries = ca_boundaries.rename(columns=dict(zip(column_names, new_column_names)))
ca_boundaries = ca_boundaries.to_crs(crs=3857) 

# reproject
hd_df = reproject_to_tracts(hd_delta_ds, ca_boundaries)
wn_df = reproject_to_tracts(wn_delta_ds, ca_boundaries)

## Check results for hot days -- AE NB
# hd_df.plot(column = hd_delta_ds.name, legend=True)
# wn_df.plot(column = wn_delta_ds.name, legend=True)

# ----------------------------------------------------------------------------------------------------------------------
## Step 4: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
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

    # note to add checker to make sure new min_max column values arent < 0 >

    # Drop the original columns
    # df = df.drop(columns=[col])
     
    return df

wn_data_std = min_max_standardize(wn_df, col=wn_delta_ds.name)
hd_data_std = min_max_standardize(hd_df, col=hd_delta_ds.name)

## Check out standardized hot days and warm nights -- AE NB
# hd_data_std.plot(column = "mean_change_annual_heat_days_min_max_standardized", legend=True)
# wn_data_std.plot(column = "mean_change_annual_warm_nights_min_max_standardized", legend=True)

# ----------------------------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
hd_data_std = hd_data_std.drop(columns=['geometry'])
wn_data_std = wn_data_std.drop(columns=['geometry'])

# export
wn_data_std.to_csv('climate_extreme_heat_warm_night_metric.csv')
hd_data_std.to_csv('climate_extreme_heat_hot_day_metric.csv')
