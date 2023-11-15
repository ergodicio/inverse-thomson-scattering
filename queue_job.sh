#!/bin/bash
#SBATCH -A m4434_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
export AWS_ACCESS_KEY_ID="AKIARRO7HXV5G2SANILY"
export AWS_SECRET_ACCESS_KEY="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"
export BASE_TEMPDIR="$PSCRATCH/tmp/"
export MLFLOW_TRACKING_URI="$PSCRATCH/mlflow"

# copy job stuff over

module load cudnn/8.9.3_cuda12.lua
module load cudatoolkit/12.0.lua
module load conda
conda activate ts

cd /global/homes/a/amilder/inverse-thomson-scattering
srun python3 run.py
