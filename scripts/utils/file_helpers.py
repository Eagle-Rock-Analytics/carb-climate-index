# Import libraries and functions
import zarr
import sys
import os
from bs4 import BeautifulSoup
import requests

sys.path.append(os.path.expanduser('~'))
from scripts.data_pull.manual_pull import aws_datasource_dirs

def to_zarr(ds, domain, aws_path, save_name):
    """Converts netcdf to zarr and sends to s3 bucket"""
    # first check that folder is not already available
    aws_path = aws_datasource_dirs(domain, datasource=aws_path)
    aws_path = "s3://ca-climate-index/"+aws_path
    filepath_zarr = aws_path+save_name+".zarr"
    # let xarray optimize chunks
    ds = ds.chunk(chunks="auto")
    try:
        ds.to_zarr(store=filepath_zarr, mode="w")
    except:
        return False

def list_webdir(url, ext=''):
    """Lists objects on a webpage"""
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]