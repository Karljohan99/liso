#!/bin/bash
# The name of the job
#SBATCH -J prepare_data

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=slurm-%x.%j.out

# The job requires 1 compute node
#SBATCH -N 1

# The maximum walltime of the job
#SBATCH -t 1-00:00:00
#SBATCH --mem=16G

#SBATCH --cpus-per-task=4

# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=karljohan30@gmail.com

#SBATCH -A "bolt"

module load singularity

singularity run --nv ${HOME}/liso/liso_dev.sif ${HOME}/liso/prepare_tartu_data.bash
