#!/bin/bash -l

#SBATCH -J carb
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -o stdout/%J.out
#SBATCH -e err/%J.err

PYSCRIPT=/shared/carb-climate-index/scripts/data_metric_calc/climate_ae_spei.py 
# Run python script with a command line argument
conda run -n carb python3 ${PYSCRIPT}
