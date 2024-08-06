import pandas as pd
import os
import sys
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import boto3
import re
import numpy as np

# suppress pandas purely educational warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

sys.path.append(os.path.expanduser('../../'))
from scripts.utils.file_helpers import pull_csv_from_directory, upload_csv_aws, filter_counties
from scripts.utils.write_metadata import append_metadata

@append_metadata
def reproject_nlcd_impervious_lands(ds, ca_boundaries, run_code=True, varname=''):
    """
    Reprojects the CA-wide USGS impervious lands zarr to California Census Tract Coordinate Reference System, 
    then clips to these CA tracts, and uploads to AWS S3. This code differs from the 
    reproject_shapefile() function by utilizing dask-geopandas to manipulate large datasets and saving the result
    as 13 parquet files. 

    Note:
    This function assumes users have configured the AWS CLI such that their access key / secret key pair are stored in
    ~/.aws/credentials.
    See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html for guidance.
     
    Methods
    -------
    Use dask-geopandas to work with the large datasets
    
    Parameters
    ----------
    zarr_fname: string
        filename of the USGS impervious lands zarr
    ca_boundaries: 
        read-in gpd file of California Census Tracts
    run_code: bool
        if True, code will run. If false, just metadata file will be updated

    Script
    ------
    large_geospatial_reproject.ipynb    
    """
    s3_client = boto3.client('s3')  
    bucket_name = 'ca-climate-index' 
    var = 'natural_usgs_impervious'
    dest_f = in_fname.replace(
        in_fname.split('/')[-1],f"{var}.parquet.gzip")
    dest_f = re.sub(r'1_pull_data', '2b_reproject', dest_f)
                
    print('Data transformation: Reproject to standard coordinate reference system: 4269.')    
    print('Data transformation: sjoin large geodata with CA census tract boundaries data.')    
    print(
            "Data transformation: Saved as multiple parquet files because"
            +" the resulting dataset is too large to be saved as one file."
    )
    print(f"Parquets saved to: s3://ca-climate-index/2b_reproject/natural_systems/ecosystem_condition/usgs/")
        
    if run_code==True:
        orig_crs = ds.spatial_ref.attrs["crs_wkt"]
        cb_crs = ca_boundaries.crs
        ca_boundaries = ca_boundaries[["GEOID","geometry"]]

        da = ds.impervious_surface
        df = da.to_dask_dataframe()
        df = df[["impervious_surface","x","y"]]
        print('made dask df')

        for i in range(len(list(df.partitions))):
            print(f"reading in partition {i}")
            part_df = df.partitions[i].compute()
            part_df = part_df[part_df["impervious_surface"]!=127.0]
            gdf = gpd.GeoDataFrame(
                part_df, geometry=gpd.points_from_xy(part_df.x,part_df.y, crs=orig_crs)
            )
            gdf = gdf.to_crs(cb_crs)
            gdf = gdf.sjoin(ca_boundaries, how='inner', predicate='intersects')
            gdf = gdf.drop(columns=["index_right","x","y"])
            print(gdf)
            dest_f = dest_f.replace(
                dest_f.split('/')[-1],f"ca_clipped_{var}_{i}.parquet.gzip")
            gdf.to_parquet(dest_f, compression='gzip')
            
# open NLCD zarr from our S3 bucket
in_fname = 's3://ca-climate-index/1_pull_data/natural_systems/ecosystem_condition/usgs/nlcd_ca_developed_impervious.zarr'
ds = xr.open_zarr(in_fname)
# read in CA census tiger file
census_shp_dir = "s3://ca-climate-index/0_map_data/2021_tiger_census_tract/2021_ca_tract/"
ca_boundaries = gpd.read_file(census_shp_dir)
varname = 'test'

rdf = reproject_nlcd_impervious_lands(ds, ca_boundaries, run_code=False, varname=varname)