#!/usr/bin/env python
# coding: utf-8

### Cal-CRAI metric: SPEI
# This script calculates the drought exposure metric:
# `% change in probability that a water year is classified as having Moderate, 
# Severe, or Extreme drought conditions via Standardized Precipitation Evapotranspiration Index (SPEI)` 
# from Cal-Adapt: Analytics Engine data. This notebook may be expanded upon for inclusion in cae-notebooks in the future. 
# SPEI will be added as an available data metric to climakitae as a part of this development. 
# 
# **Order of operations:**
# 
# 1. Read data in
# 2. Calculate base function (FFWI, SPEI, warm nights, etc.)
# 3. Calculate chronic
# 4. Calculate delta signal
# 5. Reprojection to census tracts
# 6. Min-max standardization
# 7. Export data
# 
# **Runtime**: This script must be run via pcluster due to the size and complexity of the data. 
# 
# **References**: 
# 1. S. M. Vicente-Serrano, S. Beguería, and J. I. López-Moreno, “A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index,” Journal of Climate, vol. 23, no. 7, pp. 1696–1718, Apr. 2010, doi: 10.1175/2009JCLI2909.1.
# 2. George H. Hargreaves and Zohrab A. Samani. Reference Crop Evapotranspiration from Temperature. Applied engineering in agriculture, 1(2):96–99, 1985. PubAg AGID: 5662005. doi:10.13031/2013.26773
# 3. https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.potential_evapotranspiration
# 4. https://xclim.readthedocs.io/en/stable/indices.html#xclim.indices.standardized_precipitation_evapotranspiration_index
# 
# Variables:
# 1. Daily Water Budget, which is the difference between:
#     - Daily precipitation and
#     - Daily potential evapotranspiration, derived from some combo of the following, depending on method:
#        - Daily Min Temperature
#        - Daily Max Temperature
#        - Daily Mean Temperature
#        - Relative Humidity
#        - Surface Downwelling Shortwave Radiation
#        - Surface Upwelling Shortwave Radiation
#        - Surface Downwelling Longwave Radiation
#        - Surface Upwelling Longwave Radiation
#        - 10m Wind Speed        
#    - We will be using the Hargreaves and Samani (1985) version, so we use daily min and max temperatures
# 
# 2. Calibration Daily Water Budget
#     - Can be computed from Daily Water Budget over a given "calibration" time period
# ----------------------------------------------------------------------------------------------------------------------

## Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from xclim.indices import (
    potential_evapotranspiration, 
    standardized_precipitation_evapotranspiration_index,
    standardized_precipitation_index
)

import os
import sys
import s3fs
import boto3
sys.path.append(os.path.expanduser('../../'))
from scripts.utils.file_helpers import upload_csv_aws

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
    # this step takes about 12 minutes with 3km data (~1 min with 9km data)
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

    # note to add checker to make sure new min_max column values arent < 0 > 1
    df[f'{prefix}_min_max_standardized'].loc[df[f'{prefix}_min_max_standardized'] < 0] = 0
    df[f'{prefix}_min_max_standardized'].loc[df[f'{prefix}_min_max_standardized'] > 1] = 1
    
    return df

# ----------------------------------------------------------------------------------------------------------------------
## Step 1: Retrieve data
# We need to calculate:
# * 30 year centered around 2.0C warming level (SSP3-7.0)
# * Historical baseline 1981-2010 (Historical Climate)

res = '3 km'
area_subset = "states"
cached_area = ["CA"]
## Step 1a) Chronic data (2.0°C WL)
wl = warming_levels()
# max air temperature
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.resolution = res
wl.wl_params.variable = 'Maximum air temperature at 2m'
wl.wl_params.area_subset = area_subset 
wl.wl_params.cached_area = cached_area
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.anom = "No"
wl.calculate()
ds_maxT = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_maxT = ds_maxT.sel(all_sims = list(sim_name_dict.keys()))
ds_maxT = add_dummy_time_to_wl(ds_maxT) # add time dimension back in, as this is removed by WL and is required for xclim functionality

# min air temperature
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.resolution = res
wl.wl_params.variable = 'Minimum air temperature at 2m'
wl.wl_params.area_subset = area_subset 
wl.wl_params.cached_area = cached_area
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.anom = "No"
wl.calculate()
ds_minT = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_minT = ds_minT.sel(all_sims = list(sim_name_dict.keys()))
ds_minT = add_dummy_time_to_wl(ds_minT) # add time dimension back in, as this is removed by WL and is required for xclim functionality

# precip
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.resolution = res
wl.wl_params.variable = 'Precipitation (total)'
wl.wl_params.area_subset = area_subset 
wl.wl_params.cached_area = cached_area
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.anom = "No"
wl.calculate()
ds_precip = wl.sliced_data["2.0"]
ds_precip = ds_precip.sel(all_sims = list(sim_name_dict.keys()))
ds_precip = add_dummy_time_to_wl(ds_precip)
ds_precip = ds_precip.clip(min=0.)

## Retrieve historical baseline data (1981-2010)
selections = ck.Select()
selections.timescale = 'daily'
selections.variable = 'Maximum air temperature at 2m'
selections.area_subset = area_subset 
selections.cached_area = cached_area
selections.scenario_historical=['Historical Climate']
selections.area_average = 'No'
selections.time_slice = (1981,2010) 
selections.resolution = res
max_t_hist = selections.retrieve()
max_t_hist = max_t_hist.sel(simulation=sims_hist)

# now min temperature
selections.timescale = 'daily'
selections.variable = 'Minimum air temperature at 2m'
selections.area_subset = area_subset 
selections.cached_area = cached_area
selections.scenario_historical=['Historical Climate']
selections.area_average = 'No'
selections.time_slice = (1981,2010) 
selections.resolution = res
min_t_hist = selections.retrieve()
min_t_hist = min_t_hist.sel(simulation=sims_hist)

# also need precip
selections.timescale = 'daily'
selections.variable = 'Precipitation (total)'
selections.area_subset = area_subset 
selections.cached_area = cached_area
selections.scenario_historical=['Historical Climate']
selections.area_average = 'No'
selections.time_slice = (1981,2010) 
selections.resolution = res
precip_hist = selections.retrieve()
precip_hist = precip_hist.clip(min=0.)
precip_hist = precip_hist.sel(simulation=sims_hist)

# ----------------------------------------------------------------------------------------------------------------------
## Step 2: Calculate metric
# GWL model-mean # drought years - historical model-mean # drought years
def calculate_wb(tasmin, tasmax, precip):
    # first calculate PET
    pet = potential_evapotranspiration(tasmin=tasmin, tasmax=tasmax, method='HG85')
    pet = pet * (60*60*24) # convert from per second to per day
    pet.attrs['units'] = 'mm'
    # calculate water budget
    wb = precip - pet
    wb.attrs['units'] = 'mm/day'
    wb = wb.chunk(dict(time=-1)).compute()
    return wb

def calculate_spei(wb):
    
    hist0 = '1981-01-01'
    hist1 = '2010-12-31'   
    da_list = []
    for sim in wb.simulation.values:
        da = wb.sel(simulation=sim)
        wb_min = da.min().values
        offset = str(abs(wb_min))+" mm/day"
        # finally calculate 3 month SPEI
        spei = standardized_precipitation_evapotranspiration_index(
            wb=da, 
            wb_cal=da.sel(time=slice(hist0, hist1)),
            offset=offset,
            freq='MS',
            window=3,
            dist='fisk',
            method='ML',
        )
        spei.name = "spei"
        # return spei
        da_list.append(spei) 
    spei_ds = xr.concat(da_list,dim="simulation")
    # assign water year coordinate
    water_year = (spei_ds.time.dt.month >= 10) + spei_ds.time.dt.year
    spei_ds.coords['water_year'] = water_year    
    return spei_ds

# now calculate number of drought years from SPEI
def drought_yrs(spei):   
    mod_dry_thresh = -1.0
    drought_duration_thresh = 6 # 3 months = short-term drought; 6+ = long-term
    num_dry_months = (spei <= mod_dry_thresh).groupby('water_year').sum('time')
    num_dry_years = (num_dry_months >= drought_duration_thresh).sum('water_year')
    # take model average
    num_dry_years_avg = num_dry_years.mean(dim=['simulation']).squeeze()   
    # make a nan mask
    nan_mask = spei.isel(simulation=0, time=-1).squeeze()
    # nan out grid points outside of the domain
    num_dry_years_avg = xr.where(np.isnan(nan_mask), x=np.nan, y=num_dry_years_avg)    
    return num_dry_years_avg

## Calculate water budget
wb_hist = calculate_wb(min_t_hist, max_t_hist, precip_hist)
wb_wl = calculate_wb(ds_minT, ds_maxT, ds_precip)
# Concatenate the historical and warming levels data
# apply new dummy time vector which follows the historical period
new_dummy_time = pd.date_range("2011-01-01", freq="1D", periods=10950)
wb_wl = wb_wl.assign_coords({'time': new_dummy_time})
# change simulation names so we can line them up
wb_wl = wb_wl.rename({"all_sims" : "simulation"})
wb_wl = wb_wl.assign_coords({'simulation': list(sim_name_dict.values())})
# And now concatenate the historical and warming levels datasets
wb_ds = xr.concat([wb_hist, wb_wl], dim="time")

# Calculate the SPEI
spei_ds = calculate_spei(wb_ds)

# Count the number of drought years for historical and warming level periods
spei_hist = spei_ds.sel(time=slice('1981-01-01','2010-12-31'))
drought_yrs_hist = drought_yrs(spei_hist)
spei_wl = spei_ds.sel(time=slice('2011-01-01','2040-12-31'))
drought_yrs_wl = drought_yrs(spei_wl)

# ----------------------------------------------------------------------------------------------------------------------
## Step 3: Calculate delta signal
# Difference between chronic (at 2.0°C warming level) and historical baseline (1981-2010)
ds_delta = drought_yrs_wl - drought_yrs_hist
ds_delta.name = "change_in_drought_years" # assign name so it can convert to pd.DataFrame
ds_delta = ck.load(ds_delta)
print("delta calculated")
# ----------------------------------------------------------------------------------------------------------------------
## Step 4: Reproject data to census tract projection
# load in census tract shapefile
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/" 
ca_boundaries = gpd.read_file(census_shp_dir)
# convert to area-preserving CRS
ca_boundaries = ca_boundaries.to_crs(crs=3310)
spei_df = reproject_to_tracts(ds_delta, ca_boundaries)

# ----------------------------------------------------------------------------------------------------------------------
## Step 5: Min-max standardization
# Using Cal-CRAI min-max standardization function, available in `utils.calculate_index.py`
spei_std = min_max_standardize(spei_df, col=ds_delta.name)

# ----------------------------------------------------------------------------------------------------------------------
## Step 6: Export data as csv
# Data will be exported via pcluster run

# clean up dataframes prior to export
spei_std = spei_std.drop(columns=['geometry'])

# export
bucket_name = 'ca-climate-index'
directory = '3_fair_data/index_data'

# warm nights
wn_fname = 'climate_drought_spei_metric.csv'
spei_std.to_csv(wn_fname)
upload_csv_aws([wn_fname], bucket_name, directory)
