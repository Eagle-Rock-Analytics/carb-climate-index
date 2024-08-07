#!/usr/bin/env python
# coding: utf-8

## Cal-CRAI metric: SPEI
# This script calculates the drought metric:
# `% change in probability that a water year is classified as having Moderate, Severe, or Extreme drought conditions via Standardized Precipitation Evapotranspiration Index (SPEI)` 
# from the Cal-Adapt: Analytics Engine. This notebook may be expanded upon for inclusion in cae-notebooks in the future. 
# **SPEI** will be added as an available data metric to climakitae as a part of this development. 
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
# **Runtime**: This notebook takes approximately ~1 hours to run due to data size, warming levels, and reprojection steps.
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
#        
#        *we will be using the Hargreaves and Samani (1985) version, so we use daily min and max temperatures*
# 2. Calibration Daily Water Budget
#     - Can be computed from Daily Water Budget over a given "calibration" time period
 
# ### Step 0: Import libraries
import climakitae as ck
from climakitae.explore import warming_levels 
from climakitae.util.utils import add_dummy_time_to_wl
from climakitae.util.utils import read_ae_colormap
import xarray as xr
import pandas as pd
from xclim.indices import (
    potential_evapotranspiration, 
    standardized_precipitation_evapotranspiration_index,
    standardized_precipitation_index
)
import holoviews as hv
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

## Step 1a) Chronic data (2.0deg WL)
wl = warming_levels()

# max air temperature
wl.wl_params.timescale = "daily"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.resolution = '3 km' # for pcluster run
wl.wl_params.variable = 'Maximum air temperature at 2m'
wl.wl_params.area_subset = "states"
wl.wl_params.cached_area = ["CA"]
wl.wl_params.warming_levels = ["2.0"]
wl.wl_params.anom = "No"
wl.calculate()
ds_maxT = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_maxT = add_dummy_time_to_wl(ds_maxT) # add time dimension back in, as this is removed by WL and is required for xclim functionality

# min air temperature
wl.wl_params.variable = 'Minimum air temperature at 2m'
wl.calculate()
ds_minT = wl.sliced_data["2.0"] # grab 2.0 degC data
ds_minT = add_dummy_time_to_wl(ds_minT) # add time dimension back in, as this is removed by WL and is required for xclim functionality

# precip
wl.wl_params.variable = 'Precipitation (total)'
wl.calculate()
ds_precip = wl.sliced_data["2.0"]
ds_precip = add_dummy_time_to_wl(ds_precip)
ds_precip = ds_precip.clip(min=1.)


# calculate PET
pet = potential_evapotranspiration(tasmin=ds_minT, tasmax=ds_maxT, method='HG85')
pet = pet * (60*60*24) # convert from per second to per day
pet.attrs['units'] = 'mm' # 1 mm = 1 kg/s

# calculate water budget
wb = ds_precip - pet
wb.attrs['units'] = 'mm/day'

wb_mon = wb.groupby('time.month').mean(dim='time')

da_list = []
for sim in wb.simulation.values:
    da = wb.sel(simulation=sim)
    wb_min = da.min().values
    da = da+abs(wb_min)
    da_list.append(da)
wb = xr.concat(da_list, dim='simulation')
wb = wb.chunk(dict(time=-1)).compute()


spei = standardized_precipitation_evapotranspiration_index(
    wb=wb, 
    wb_cal=wb,
    freq='MS',
    window=3,
    dist='gamma',
    method='APP',
)


water_year = (spei.time.dt.month >= 10) + spei.time.dt.year
spei.coords['water_year'] = water_year

mod_dry_thresh = -1.0
drought_duration_thresh = 6 # 3 months = short-term drought; 6+ = long-term
num_dry_months = (spei <= mod_dry_thresh).groupby('water_year').sum('time', skipna=True)
num_dry_years = (num_dry_months >= drought_duration_thresh).sum('water_year', skipna=True)

## Step 1b) Historical baseline data (1981-2010)
selections = ck.Select()
selections.timescale = 'daily'
selections.variable = 'Maximum air temperature at 2m'
selections.area_subset = 'states'
selections.cached_area = ['CA']
selections.scenario_historical=['Historical Climate']
selections.area_average = 'No'
selections.time_slice = (1981,2010) 
selections.resolution = '3 km'
max_t = selections.retrieve()

# now min temperature
selections.variable = 'Minimum air temperature at 2m'
min_t = selections.retrieve()

# also need precip
selections.variable = 'Precipitation (total)'
precip = selections.retrieve()
precip = precip.clip(min=1.)
# precip = precip.where((precip > 0.) | precip.isnull(), 0.)

# Calculate potential evapotranspiration and plot as a quick annual sum to make sure everything looks reasonable. We expect *O*(1000 mm/year). 

pet = potential_evapotranspiration(tasmin=min_t, tasmax=max_t, method='HG85')
pet = pet * (60*60*24) # convert from per second to per day
pet.attrs['units'] = 'mm' # 1 mm = 1 kg/s

pet_sum = pet.groupby('time.year').sum(dim='time', skipna=True)


# Calculate the water budget and plot monthly means as a quick check
wb = precip - pet
wb.attrs['units'] = 'mm/day'
wb_mon = wb.groupby('time.month').mean(dim='time')

# Compute and add the offset so water budget is never negative
da_list = []
for sim in wb.simulation.values:
    da = wb.sel(simulation=sim)
    wb_min = da.min().values
    da = da+abs(wb_min)
    da_list.append(da)
wb = xr.concat(da_list, dim='simulation')
wb = wb.chunk(dict(time=-1)).compute()

# Calculate 3-month SPEI and check results
spei = standardized_precipitation_evapotranspiration_index(
    wb=wb, 
    wb_cal=wb,
    freq='MS',
    window=3,
    dist='gamma',
    method='APP',
)

# Count number of water years featuring 6 or more months with SPEI < -1 (ie, 6 or more dry months in a year)
# assign water year coordinate
water_year = (spei.time.dt.month >= 10) + spei.time.dt.year
spei.coords['water_year'] = water_year

mod_dry_thresh = -1.0
drought_duration_thresh = 6 # 3 months = short-term drought; 6+ = long-term
num_dry_months = (spei <= mod_dry_thresh).groupby('water_year').sum('time', skipna=True)
num_dry_years = (num_dry_months >= drought_duration_thresh).sum('water_year', skipna=True)

drought_bins = [-8.41, -2, -1.5, -1.0, 0.99, 1.49, 1.99, 8.41]
drought_labels = ["Extremely Dry", "Very Dry",
                  "Moderately Dry", "Normal",
                  "Moderately Wet", "Very Wet",
                  "Extremely Wet"]
grp = spei.groupby_bins(spei, drought_bins, labels=drought_labels)
grp["Normal"].count() 

levels = [-8.41, -2, -1.5, -1.0, 0, 1., 1.5, 2., 8.41]
# wb = wb.chunk(dict(time=-1)).compute()
spei = standardized_precipitation_evapotranspiration_index(
    wb=wb, 
    wb_cal=wb,
    # .sel(time=slice('1981','2010')),
    freq='MS',
    window=1,
    dist='gamma',
    method='APP',
).compute()

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

df = ds_delta.to_dataframe().reset_index()

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
        # df.drop(columns=[col], inplace=True)
     
    return df

data_std = min_max_standardize(diss_gdf, cols_to_run_on=['change_ffwi_days'])

# -------------------------------------------------------------------------------------------------
## Step 5: Export data as csv
# Data will be exported via pcluster run
# clean up dataframes prior to export
data_std = data_std.drop(columns=['geometry'])

# export
data_std.to_csv('climate_drought_spei_metric.csv')