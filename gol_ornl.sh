#!/bin/sh
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --time=72:00:00
#SBATCH --nodes=1 --gres=gpu:1 --constraint=V100
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="gol_test"
#SBATCH --output=test-gol.out
#SBATCH --mail-user=muhanned@buffalo.edu
#SBATCH --mail-type=ALL
##SBATCH --requeue
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.

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
echo "Launch gol-cuda"
./gol-cuda

echo "All Done!"
