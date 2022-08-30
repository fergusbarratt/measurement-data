#!/bin/bash
#SBATCH -o logs/%j                 # output and error file name (%j expands to jobID)
#SBATCH -N 128                     # number of nodes requested
#SBATCH -n 128                     # total number of tasks to run in parallel
#SBATCH -p normal                  # queue (partition)    
#SBATCH -t 5:00:00                 # run time (hh:mm:ss) 

module load launcher

python3 generate_param_sweep.py

export LAUNCHER_WORKDIR=/work2/08522/barratt/frontera/random_circuits/nonsymmetric/python/ed/measurement-data/sub/generate_data
export LAUNCHER_JOB_FILE=param_sweep

${LAUNCHER_DIR}/paramrun
