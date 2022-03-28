#! /bin/bash
#BSUB -n 25
#BSUB -q long
#BSUB -W 6:00
#BSUB -R "rusage[mem=4000]" 
#BSUB -R "span[hosts=1]"
#BSUB -R "select[rh=8]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

julia --project=/home/fb20a/random_circuits/symmetric/SymED.jl --threads=25 /home/fb20a/random_circuits/symmetric/SymED.jl/src/run.jl
