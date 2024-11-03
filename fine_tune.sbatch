#!/bin/bash
#SBATCH --account=cakcora
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --error=rpwslurm-%J.err
#SBATCH --output=rpwslurm-%J.out
#SBATCH --job-name=adversarialGNN
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=ngot1@myumanitoba.ca

# Load modules
module load anaconda/anaconda-2023.09  # Replace with the correct Anaconda module available on your cluster
# Create a new Conda environment with Python 3.8.19
conda activate AdversarialGNN

cd /home/bngo/U-AdversarialGNN
module load cuda/cuda-12.1
# Installing compatible versions
python experiments/hyper_tuning_each_split.py --model=GNNGuard --split_idx=0 --dataset=ogbn-arxiv
# Deactivate virtual environment
conda deactivate