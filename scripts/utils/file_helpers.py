# Import libraries and functions
import zarr
from bs4 import BeautifulSoup
import requests
import boto3
import zipfile
import io
import pandas as pd

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

def pull_csv_from_directory(bucket_name, directory, search_zipped=True):
    """
    Pulls CSV files from a specified directory in an S3 bucket.
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for CSV files.
    - search_zipped (bool): If True, search for CSV files within zip files. If False, search for CSV files directly.
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    # List objects in the specified directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)

    # Check if objects were found
    if 'Contents' in response:
        # Iterate through each object found
        for obj in response['Contents']:
            # Get the key (filename) of the object
            key = obj['Key']
            
            # Check if the object is a .zip file
            if search_zipped and key.endswith('.zip'):
                # Download the zip file into memory
                zip_object = s3.get_object(Bucket=bucket_name, Key=key)
                zip_data = io.BytesIO(zip_object['Body'].read())
                
                # Open the zip file
                with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                    # Iterate through each file in the zip
                    for file_name in zip_ref.namelist():
                        # Check if the file is a .csv file
                        if file_name.endswith('.csv'):
                            # Read the .csv file
                            with zip_ref.open(file_name) as csv_file:
                                # Convert the csv content to pandas DataFrame
                                df = pd.read_csv(csv_file)
                                # Save the DataFrame with a similar name as the .csv file
                                df_name = file_name[:-4]  # Remove .csv extension
                                df.to_csv(f"{df_name}.csv", index=False)
                                print(f"Saved DataFrame as '{df_name}.csv'")
                                # You can now manipulate df as needed
            elif not search_zipped and key.endswith('.csv'):
                # Directly download the CSV file
                csv_object = s3.get_object(Bucket=bucket_name, Key=key)
                csv_data = io.BytesIO(csv_object['Body'].read())
                # Convert the csv content to pandas DataFrame
                df = pd.read_csv(csv_data)
                # Save the DataFrame with a similar name as the .csv file
                df_name = key.split('/')[-1][:-4]  # Extract filename from key
                df.to_csv(f"{df_name}.csv", index=False)
                print(f"Saved DataFrame as '{df_name}.csv'")
                # You can now manipulate df as needed

    else:
        print("No objects found in the specified directory.")


def upload_csv_aws(file_names, bucket_name, directory):
    """
    Uploads CSV files to a specified directory in an S3 bucket.
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for CSV files.
    - file_names (str): .csv file to be uploaded to aws
    """  
    # Create an S3 client
    s3 = boto3.client('s3')
    # Iterate over each file name in the list
    for file_name in file_names:
        # Save the file to AWS S3 using the client
        with open(file_name, 'rb') as data:
            s3.upload_fileobj(data, bucket_name, f"{directory}/{file_name}")
            print(f"{file_name} uploaded to AWS")

    def filter_counties(df, county_column, county_list=None):
        '''
        Filter a df's county column to a list of established CA counties
        Parameters
        ----------
        df: dataframe
            name of the dataframe to be filtered
        
        column: string
            name of the county column within your dataframe

        county_list: list
            list of counties to be filtered for, if left blank the default list is CA counties shown below
        '''

        # Default county list if not provided
        if county_list is None:
            county_list = [
                    'alameda', 'alpine', 'amador', 'butte', 'calaveras', 'colusa', 'contra costa', 'del norte',
                    'el dorado', 'fresno', 'glenn', 'humboldt', 'imperial', 'inyo', 'kern', 'kings', 'lake', 'lassen',
                    'los angeles', 'madera', 'marin', 'mariposa', 'mendocino', 'merced', 'modoc', 'mono', 'monterey',
                    'napa', 'nevada', 'orange', 'placer', 'plumas', 'riverside', 'sacramento', 'san benito',
                    'san bernardino', 'san diego', 'san francisco', 'san joaquin', 'san luis obispo', 'san mateo',
                    'santa barbara', 'santa clara', 'santa cruz', 'shasta', 'sierra', 'siskiyou', 'solano', 'sonoma',
                    'stanislaus', 'sutter', 'tehama', 'trinity', 'tulare', 'tuolumne', 'ventura', 'yolo', 'yuba'
                ]
        
        # Convert county_list to lowercase for case-insensitive comparison
        county_list_lower = [county.lower() for county in county_list]
        
        # Filter rows where the value in the specified column matches any of the counties in the list
        filtered_df = df[df[county_column].str.lower().isin(county_list_lower)]
        
        return filtered_df