"""
This script uploads manually downloaded data to AWS bucket for the California Climate Risk Index.
Because files can be drag+dropped into an AWS bucket, this is somewhat redundant, but provide a
script here to complement the other pull scripts for this work. 

Notes:
1. Filetypes to prioritize: shapefile, netcdf, csv, geotiff(?)
"""

# Import libraries
import boto3

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


def manual_to_aws():
    """Uploads data that was manually downloaded to AWS bucket"""

    # user input on upload
    # domain options: built_environment, governance, natural_systems, society_economy, climate_risk
    domain_resp = input('Which domain to save to? ')
    if domain_resp == 'built_environment':
        domain = 'built_environment'
    elif domain_resp == 'society_economy':
        domain = 'society_economy'
    elif domain_resp == 'climate_risk':
        domain = 'climate_risk'
    elif domain_resp == 'governance':
        domain = 'governance'
    elif domain_resp == 'natural_systems':
        domain = 'natural_systems'
    else:
        print('Please pass a valid domain name')

    data_resp = input('What is the datasource? ')
    datasource = str(data_resp)

    loc_resp = input('Where is the file stored locally? Please pass the path and filename. ')
    loc = str(loc_resp)


    # first check that folder is not already available
    path_to_save = aws_datasource_dirs(domain, datasource)

    # extract the filename from path
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
    manual_to_aws()
