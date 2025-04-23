#!/bin/bash
#SBATCH --job-name=newcode
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx2080
#SBATCH --output=out_slurms/diag_%j.out

cd /home/pyla/bayesian/from_notebook/vi-per

eval "$(conda shell.bash hook)"
conda activate bfn2

SEEDS=(0)
LMAXS=(1 2 3 5 8 10 20 40)
BETAS=(0.1 0.5 1.0 2.0 5.0 10.0)
for SEED in "${SEEDS[@]}"
do
    # for LMAX in "${LMAXS[@]}"
    for BETA in "${BETAS[@]}"
    do
        # jq --argjson seed $SEED --argjson lmax $LMAX '.seed = $seed | .l_max = $lmax' newcode/diag.json > newcode/temp_diag.json
        jq --argjson seed $SEED --argjson beta $BETA '.seed = $seed | .beta = $beta' newcode/diag.json > newcode/tempp_diag.json
        python src/main.py --config newcode/tempp_diag.json
    done
done