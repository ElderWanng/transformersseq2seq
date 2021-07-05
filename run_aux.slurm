#!/bin/bash
#SBATCH --job-name=BART XSUM WITH MULTINLI
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2-00:10:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay2/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.0-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
cd
source /ext3/env.sh
conda activate
cd /home/tw2112/codes/s2s
sh run_aux.sh
"
