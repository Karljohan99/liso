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
#SBATCH -t 5-00:00:00
#SBATCH --mem=16G

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1

# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=karljohan30@gmail.com

#SBATCH -A "bolt"

module load singularity

singularity run --nv --bind /gpfs/space/projects:/mnt ${HOME}/liso/liso_tartu.sif ${HOME}/liso/train_slim_tartu.bash
