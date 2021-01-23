#!/bin/bash -l

#SBATCH -p azad
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -J spadd
#SBATCH -o rmat_20_8_10.o%j

export OMP_NUM_THREADS=48
srun -N 1 -n 1 -c 48 ./exp_scaling
