from datetime import datetime
from functools import wraps

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
        varname = kwargs['varname']
        f = open(varname + "_metadata.txt", "a")
        f.write("\n")
        f.write("\n")
        f.write("======== Function(s) applied to " + varname + " ========")
        f.write("\n")
        f.write("\n")
        f.write(f"Function name: {func.__name__}")
        f.write("\n")
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