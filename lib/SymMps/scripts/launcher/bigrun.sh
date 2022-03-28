#!/bin/bash
#SBATCH -o logs/%j                 # output and error file name (%j expands to jobID)
#SBATCH -N 8                       # number of nodes requested
#SBATCH -n 8                       # total number of tasks to run in parallel
#SBATCH -p development             # queue (partition) 
#SBATCH -t 2:00:00                 # run time (hh:mm:ss) 

module load launcher

export LAUNCHER_WORKDIR=/work2/08522/barratt/frontera/random_circuits/symmetries/su2/SymMps/scripts/launcher/
export LAUNCHER_JOB_FILE=job_int_param_sweep

${LAUNCHER_DIR}/paramrun
