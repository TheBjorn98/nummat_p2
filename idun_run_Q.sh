#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000MB
#SBATCH --job-name="QTrajectory_Nummat_Neural_training"
#SBATCH --output=Qout.txt

module purge
module load GCCcore/.9.3.0
module load Python/3.8.2

python3 trajectory_training_Q.py
