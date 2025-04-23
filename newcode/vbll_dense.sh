#!/bin/bash
#SBATCH --job-name=newcode
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090
#SBATCH --output=out_slurms/vbll_dense_%j.out


cd /home/pyla/bayesian/from_notebook/vi-per

eval "$(conda shell.bash hook)"
conda activate bfn2

python src/main.py --config newcode/vbll_dense.json
