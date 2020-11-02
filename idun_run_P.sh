#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000MB
#SBATCH --job-name="PTrajectory_Nummat_Neural_training"
#SBATCH --output=Pout.txt

module purge
module load GCCcore/.9.3.0
module load Python/3.8.2

python3 trajectory_training_P.py
