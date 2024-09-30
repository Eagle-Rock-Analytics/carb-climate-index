from datetime import datetime
from functools import wraps
import os
import boto3
import sys
s3_client = boto3.client('s3')

def make_metadata_files(df):
    """
    Function to make variable metadata from a csv
    """
    now = datetime.now()
    datestr = now.strftime("%B %d, %Y")
    varlist = list(df['Variable'].values)
    for var in varlist:    
        vardat = df.loc[df['Variable'] == var]
        meta_dict = vardat.set_index('Variable').to_dict()
        file_name = f"{var}_metadata.txt"
        metadata_path = "3_fair_data/metadata/"
        obj_name = f"{metadata_path}{file_name}"
        f = open(file_name, "w")
        f.write(
            "========"
            + " Metadata document "
            + "prepared for the California "
            + "Air Resources Board "
            + "'California Climate Risk "
            + "and Adaptation Index'"
            + " ========") 
        f.write("\n")
        f.write("\n")
        f.write(
            "Index and all accompanying documentation "
            + "developed by Eagle Rock Analytics, Inc."
        )
        f.write("\n")
        f.write("Website: https://www.eaglerockanalytics.com")
        f.write("\n")
        f.write("Github: https://github.com/Eagle-Rock-Analytics/carb-climate-index")
        f.write("\n")
        f.write(f"This document refers to the following variable: {var}")
        f.write("\n")        
        for key, val in zip(
            list(meta_dict.keys()),
            list(meta_dict.values())
        ):
            f.write(f"{key}: {val[var]}")
            f.write("\n")        
        f.write(
            "*** This document was generated on ("+ datestr+"), "
            +"to describe the steps used to incorporate this metric "
            +"into the California Climate Risk and Adaptation Index. ***")
        f.write("\n")
        f.write("*** All metrics used to calculate the California Climate Risk "
                +"and Adaptation Index are archived at "
                +"https://doi.org/10.5281/zenodo.13840187 ***")
        f.close()
        s3_client.upload_file(
            file_name, 'ca-climate-index', obj_name)
        print(f"Uploaded metadata file to {obj_name} in S3 bucket")
        os.remove(file_name)

def append_metadata(func):
    '''
    Decorator for a given function to pull its name, args, and kwargs
    and write to a metadata document. Adapted from 
    https://www.w3resource.com/python-exercises/decorator/python-decorator-exercise-1.php
    Note: varname is a required keyword argument, and must be supplied to the function! 
    '''
    def metadata_generator(*args, **kwargs):
        stdout_backup = sys.stdout
        metadata_path = "3_fair_data/metadata/"

        # Write the function parameters to file
        now = datetime.now()
        datestr = now.strftime("%B %d, %Y")
        var = kwargs['varname']       
        file_name = f"{var}_metadata.txt"

        if var != "test":
            # Download the metadata file from S3
            obj_name = f"{metadata_path}{file_name}"
            transform_prefix = func.__name__
            s3_client.download_file(
                'ca-climate-index', obj_name, file_name)
        
        with open(file_name, "a") as f:
            sys.stdout = f
            f.write("\n")
            f.write("======== Function applied to " + var + " ========")
            f.write("\n")
            f.write("\n")
            f.write(f"Function name: {func.__name__}")
            f.write("\n")
            f.write(f"Date function applied: {datestr}")
            f.write("\n")
            f.write(f"Function description: {func.__doc__}")
            f.write("\n")
            f.write(f"Function output statements:")
            f.write("\n")
            # Call the function
            result = func(*args, **kwargs)     
            
        sys.stdout = stdout_backup
        if var != "test":
            # Upload the file to S3
            s3_client.upload_file(
                file_name, 'ca-climate-index', obj_name)
            os.remove(file_name)
        return result
    return metadata_generator

