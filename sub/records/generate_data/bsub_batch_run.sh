#! /bin/bash
#BSUB -n 20
#BSUB -q long
#BSUB -W 24:00
#BSUB -R "rusage[mem=10000]" 
#BSUB -R "span[hosts=1]"
#BSUB -R "select[rh=8]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

source /home/fb20a/miniconda3/etc/profile.d/conda.sh && conda activate measurement_analysis && python batch_run.py
