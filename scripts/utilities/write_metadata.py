from datetime import datetime
from functools import wraps
import os

metadata_path = os.path.expanduser('~/metadata/')

def make_metadata_files(df):
    """
    Function to make variable metadata from a csv
    """
    varlist = list(df['Variable'].values)
    for var in varlist:    
        vardat = df.loc[df['Variable'] == var]
        meta_dict = vardat.set_index('Variable').to_dict()
        f = open(
            f"{metadata_path}/{var}_metadata.txt", "w"
        )
        f.write(
            "========"
            + " Metadata document "
            + "prepared for the California "
            + "Air Resources Board "
            + "'California Climate Resilience "
            + "and Adaptation Index'"
            + " ========") 
        f.write("\n")
        f.write("\n")
        f.write(
            "Index and all accompanying documentation "
            + " developed by Eagle Rock Analytics, Inc."
        )
        f.write("\n")
        f.write(f"This document refers to the following variable: {var}")
        f.write("\n")        
        for key, val in zip(
            list(meta_dict.keys()),
            list(meta_dict.values())
        ):
            f.write(f"{key}: {val[var]}")
            f.write("\n")
        f.write("Github: https://github.com/Eagle-Rock-Analytics/carb-climate-index")
        f.close()

def append_metadata(func):
    '''
    Decorator for a given function to pull its name, args, and kwargs
    and write to a metadata document. Adapted from 
    https://www.w3resource.com/python-exercises/decorator/python-decorator-exercise-1.php
    Note: varname is a required keyword argument, and must be supplied to the function! 
    '''
    def metadata_generator(*args, **kwargs):
        # Call the function
        result = func(*args, **kwargs)

        # Write the function parameters to file
        now = datetime.now()
        datestr = now.strftime("%B %d, %Y")
        var = kwargs['varname']
        f = open(f"{metadata_path}/{var}_metadata.txt", "a")
        f.write("\n")
        f.write("\n")
        f.write("======== Function(s) applied to " + var + " ========")
        f.write("\n")
        f.write("\n")
        f.write(f"Function name: {func.__name__}")
        f.write("\n")
        f.write(f"Function description: {func.__doc__}")
        if args:
            f.write(f"Function arguments: {args}")
            f.write("\n")
        if len(kwargs)>1:
            f.write(f"Function keyword arguments: {args}")
            f.write("\n")
            for key, value in zip(list(kwargs.keys()), list(kwargs.values())):
                f.write(f"{key} = {value}")
                f.write("\n")
        f.write(f"Date function applied: {datestr}")
        f.write("\n")
        f.write("\n")
        f.close()
        return result
    return metadata_generator

