#!/bin/bash
# The name of the job
#SBATCH -J train_slim

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=slurm-%x.%j.out

# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=1

# The maximum walltime of the job is 5 minutes
#SBATCH -t 01:00:00
#SBATCH --mem=316G

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1

# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=karljohan30@gmail.com

#SBATCH -A "bolt"

CUDA_HOME="/gpfs/space/software/cluster_software/spack/linux-centos7-x86_64/gcc-9.2.0/cuda-11.3.1-oqzddj7nezymwww6ennwec7qb6kktktw"

module load singularity

singularity run --nv ${HOME}/liso/liso_dev.sif ${HOME}/liso/train_slim.bash
