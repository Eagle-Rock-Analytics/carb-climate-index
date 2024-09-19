#!/bin/bash -l

#SBATCH -J carb_ffwi
#SBATCH -N 1
#SBATCH -c 48
#SBATCH -t 24:00:00
#SBATCH -p cri
#SBATCH -o stdout/%J.out
#SBATCH -e err/%J.err

PYSCRIPT=/shared/carb-climate-index/scripts/data_metric_calc/climate_ae_ffwi.py 
# Run python script with a command line argument
conda run -n carb python3 ${PYSCRIPT}
