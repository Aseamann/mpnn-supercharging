#!/bin/bash
#SBATCH --partition=p_sdk94_1              # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=supercharge         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=4            # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --mem=32GB                   # Real memory (RAM) required (MB), 0 is the whole-node memory
#SBATCH --time=2-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm_sc.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm_sc.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

module use /projects/community/modulefiles
module load gcc/14.2.0-cermak

start=0.1
end=0.9
step=0.1

for val in $(seq $start $step $end); do
    echo "Running supercharge with temp: $val"
    python protein_mpnn_supercharge.py -i pdb -o eGFPsc_mpnn_${val//./}.fa -t $val -bottom -100 -top 100 -cat 1 -d 0.0 -n 10 --model v_48_010 --weights soluble -mhbond -v
done
