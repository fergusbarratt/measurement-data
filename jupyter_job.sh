#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH -o logs/%j                 # output and error file name (%j expands to jobID)
#SBATCH -N 4                       # number of nodes requested
#SBATCH -n 4                       # total number of tasks to run in parallel
#SBATCH -p development             # queue (partition) 
#SBATCH -t 2:00:00                 # run time (hh:mm:ss) 

#Pick a random or predefined port
#port=$(shuf -i 6000-9999 -n 1)
port=8080
#Forward the picked port to the login1 on the same port. Here log-x is set to be the prince login node.
/usr/bin/ssh -N -f -R $port:localhost:$port login1

source /work2/08522/barratt/frontera/miniconda3/etc/profile.d/conda.sh && conda activate measurement_analysis

module list
module unload python3/3.7.0

ibrun dask-mpi --scheduler-file ~/dask-scheduler.json &

#Start the notebook
jupyter lab --no-browser --port $port
