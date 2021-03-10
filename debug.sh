#!/bin/sh

# General SLURM Specs
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --exclusive
#SBATCH --account=jzola
#SBATCH --time=00:10:00

# CUDA Specific Specs
# Need a node with a GPU
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100

# tasks per node I'm not sure we need
# Reduced to 1 for debugging purposes
#SBATCH --ntasks-per-node=1

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

module load cuda/11.0
ulimit -s unlimited


#Initial srun will trigger the SLURM prologue on the compute nodes.
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS
echo "Launch $1"
$1

echo "All Done!"
