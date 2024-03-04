#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
# Thread count:
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
# memory in MB
#SBATCH --mem=64G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/you/OUAI_Demo/out.txt
#SBATCH --error=/home/you/OUAI_Demo/err.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=demo
#SBATCH --mail-user=your_email@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/you/OUAI_Demo/
#################################################

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip


. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/torch
wandb login your_api_key

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
main.py
