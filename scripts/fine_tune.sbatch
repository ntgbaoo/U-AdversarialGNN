#!/bin/bash

# sbatch config
#SBATCH --account=cakcora
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --error=/home/bngo/error/$.err
#SBATCH --output=log_test.txt
#SBATCH --job-name=adversarialGNN
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ngot1@myumanitoba.ca

# Load modules
#module load anaconda/anaconda-2023.09  # Replace with the correct Anaconda module available on your cluster
# Create a new Conda environment with Python 3.8.19
conda activate AdversarialGNN
pip list
cd /home/bngo/U-AdversarialGNN
module load cuda/cuda-12.1
# Installing compatible versions
python experiments/hyper_tuning_each_split.py --model=GCN --split_idx=1 --dataset=ogbn-arxiv
# Deactivate virtual environment
#conda deactivate