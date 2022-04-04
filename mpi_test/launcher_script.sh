#!/bin/bash
#SBATCH -o logs/%j                 # output and error file name (%j expands to jobID)
#SBATCH -N 20                      # number of nodes requested
#SBATCH -n 20                      # total number of tasks to run in parallel
#SBATCH -p development             # queue (partition) 
#SBATCH -t 2:00:00                 # run time (hh:mm:ss) 

module load launcher

export LAUNCHER_WORKDIR=/work2/08522/barratt/frontera/random_circuits/nonsymmetric/python/ed/measurement-data/mpi_test
export LAUNCHER_JOB_FILE=param_sweep

${LAUNCHER_DIR}/paramrun
