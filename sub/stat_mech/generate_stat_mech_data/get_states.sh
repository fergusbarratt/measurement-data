#! /bin/bash
#BSUB -n 25
#BSUB -q long
#BSUB -W 120:00
#BSUB -R "rusage[mem=2000]" 
#BSUB -R "span[hosts=1]"
#BSUB -R "select[rh=8]"

module load anaconda3/2019.03
source /share/pkg/anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate tebd
python get_states.py
