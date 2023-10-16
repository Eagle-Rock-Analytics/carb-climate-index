"""
This script uploads manually downloaded data to AWS bucket for the California Climate Risk Index.
Because files can be drag+dropped into an AWS bucket, this is somewhat redundant, but provide a
script here to complement the other pull scripts for this work. 

Notes:
1. Filetypes to prioritize: shapefile, netcdf, csv, geotiff(?)
"""

# Import libraries
import boto3
import argparse

# Set AWS credentials
s3 = boto3.resource('s3')
s3_cl = boto3.client('s3') # for lower-level processes
bucket_name = 'ca-climate-index'
raw_path = '1_pull_data/' # path to raw datafiles in AWS bucket


def aws_datasource_dirs(domain, datasource):
    """Creates a dir in the respective domain dir, if not already available"""
    bucket = s3.Bucket(bucket_name)

    # path to folder in aws
    datasource_dir = '{0}{1}/{2}/'.format(raw_path, domain, datasource)

    # check if folder already exists
    dirs = []
    for item in bucket.objects.filter(Prefix=raw_path+domain+'/'):
        d = str(item.key)
        dirs += [d]

    if datasource_dir not in dirs:
        print('Creating folder for {}'.format(datasource_dir))
        bucket.put_object(Key=datasource_dir)

    return datasource_dir


def manual_to_aws(domain, datasource, loc):
    """Uploads data that was manually downloaded to AWS bucket"""

    # first check that folder is not already available
    path_to_save = aws_datasource_dirs(domain, datasource)

    # extract the filename from path
    loc = loc.replace('\\', '/')
    fname = loc.split('/')[-1]

    # point to location of file(s) locally and upload to aws
    try:
        s3_cl.upload_file(
            loc,
            bucket_name,
            aws_datasource_dirs(domain, datasource)+fname
        )
        print('{0} saved to {1}'.format(fname, path_to_save))
    except Exception as e:
        print(e)


# Run functions
if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(
        description='Script uploads manually downloaded data to AWS bucket',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Define arguments
    parser.add_argument('domain', help='Options: built_environment, governance, natural_systems, society_economy, climate_risk')
    parser.add_argument('datasource', help='Organization of datasource')
    parser.add_argument('loc', help='Local path to filename to upload')

    # Parse out arguments for use
    args = parser.parse_args()
    domain = args.domain
    datasource = args.datasource
    loc = args.loc


    manual_to_aws(domain, datasource, loc)
