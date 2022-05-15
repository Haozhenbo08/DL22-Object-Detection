#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err

singularity exec --nv \
--bind /scratch \
--overlay /scratch/rsc468/conda.ext3:ro \
--overlay /scratch/rsc468/unlabeled_112.sqsh \
--overlay /scratch/rsc468/labeled.sqsh \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate 
cd /home/rsc468/barlow-backbone-GCP.bis
python main.py
"
