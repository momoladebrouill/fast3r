#!/bin/bash                                                

#SBATCH --account=ryan

#SBATCH --nodes=1 # number of nodes (don't forget to change it in the variable also                                    

#SBATCH --gres=gpu:1              # Number of GPUs (per node)                                                          

#SBATCH --mincpus=1                                        

#SBATCH --mem=0               # memory (per node)

#SBATCH --time=1-00:00            # time (DD-HH:MM)

#SBATCH --mail-user=ryan.bartolomeo@polymtl.ca

#SBATCH --mail-type=ALL 

#SBATCH --signal=SIGUSR1@120 # send a signal 120s before end to save the checkpoint

#SBATCH --job-name=NO_MEM_LIMIT


cd /home/travail/ryan/fast3r

echo
echo ---Activating FASTR
. /home/usagers/ryan/miniconda3/etc/profile.d/conda.sh
conda activate fast3r
echo inside
pwd


echo
echo ---Checking versions
pip freeze | grep "torch=="
nvcc --version

echo
echo ---Launching job
export experiment=super_long_training/super_long_training
#export SLURM_GPUS_PER_NODE=1
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=4871
export HYDRA_FULL_ERROR=1
export nodes=1


srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --rdzv-id=$RDZV_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    fast3r/train.py \
    paths.run_folder_name=$experiment_$SLURM_JOBID \
    +trainer.num_nodes=$nodes \
    +logger.wandb.name=$experiment_$SLURM_JOBID \
    experiment=$experiment \
    +partition_sizes=1.0
