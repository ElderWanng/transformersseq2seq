#!/bin/bash
#SBATCH --job-name=Ablation
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
cd
source /ext3/env.sh
conda activate trans
cd
cd wts/s2s/
sh Alation_only_task.sh
"
