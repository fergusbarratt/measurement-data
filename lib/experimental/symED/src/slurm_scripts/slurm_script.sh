#!/bin/bash
# Request (N) 20*56=1120 cores, each of which runs a time evolution in Q, Q+1
#SBATCH -J sharp,12,1120
#SBATCH -o logs/%j                 # output and error file name (%j expands to jobID)
#SBATCH -N 20                      # number of nodes requested
#SBATCH -n 20                      # total number of tasks to run in parallel
#SBATCH -p development             # queue (partition) 
#SBATCH -t 2:00:00                 # run time (hh:mm:ss) 

module load launcher

export JULIA_PROJECT = "../../"
export JULIA_NUM_THREADS=56
export LAUNCHER_WORKDIR=/work2/08522/barratt/frontera/random_circuits/symmetric/u1/julia/SymED.jl/src/slurm_scripts
export LAUNCHER_JOB_FILE=run_many

${LAUNCHER_DIR}/paramrun

