#!/bin/sh

#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000MB
#SBATCH --job-name="Job_2_Nummat_Neural_training"
#SBATCH --output=out.txt

module purge
module load GCCcore/.9.3.0
module load Python/3.8.2

python3 ../../../anntrainingtest.py 1 2 100 2 4 1 0.005 1000000 1e-10 Job_2
