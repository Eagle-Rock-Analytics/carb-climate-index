# Import libraries and functions
import zarr
from bs4 import BeautifulSoup
import requests

def to_zarr(ds, top_dir, domain, indicator, data_source, save_name):
    """Converts netcdf to zarr and sends to s3 bucket"""
    # first check that folder is not already available
    aws_path = '{0}/{1}/{2}/{3}/'.format(
        top_dir, domain, indicator, data_source
    )
    aws_path = "s3://ca-climate-index/"+aws_path
    filepath_zarr = aws_path+save_name+".zarr"
    # let xarray optimize chunks
    ds = ds.chunk(chunks="auto")
    ds.to_zarr(store=filepath_zarr, mode="w")

def list_webdir(url, ext=''):
    """Lists objects on a webpage"""
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def min_max_standardize(df, col_to_run_on):
    '''
    Calculates a dataframes min and max values based on a specified column, then calculates
    a min-max standardized value. Min, max, and standardized vaulue columns are created and
    added to the dataframe.

    Parameters
    ----------
    df: string
        Dataframe name   
    col_to_run_on: string
        Column within the string to calculate min, max, and standardize
    '''
    max_value = df[col_to_run_on].max()
    min_value = df[col_to_run_on].min()

    # Get min-max values, standardize, and add columns to df
    df['max_sum_value'] = max_value
    df['min_sum_value'] = min_value
    df['min_max_standardized'] = ((df[col_to_run_on] - min_value ) / (max_value - min_value))
     
    return df