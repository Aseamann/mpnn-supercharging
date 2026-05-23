#!/bin/bash
#SBATCH --partition=p_sdk94_1              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=supercharge         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=4            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --mem=32GB                   # Real memory (RAM) required (MB), 0 is the whole-node memory
#SBATCH --time=01:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm_sc.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm_sc.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

module use /projects/community/modulefiles
module load gcc/14.2.0-cermak

charge=$1

python protein_mpnn_supercharge.py -i pdb -o eGFPsc_pmpnn_${charge}.fa -t 0.3 -c ${charge} -n 10 --model v_48_020 --weights original -mhbond -u -v
