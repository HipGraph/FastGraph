#!/bin/bash -l

#SBATCH -p azad
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J spadd
#SBATCH -o spadd.o%j

export OMP_NUM_THREADS=48
export FILE=hybrid.csv

echo "matrix,scale,d,k,thread,algorithm,total" > $FILE

./exp_scaling 16 8 10 0 >> $FILE
./exp_scaling 16 8 20 0 >> $FILE
./exp_scaling 16 8 40 0 >> $FILE
./exp_scaling 16 8 80 0 >> $FILE
./exp_scaling 18 8 10 0 >> $FILE
./exp_scaling 18 8 20 0 >> $FILE
./exp_scaling 18 8 40 0 >> $FILE
./exp_scaling 18 8 80 0 >> $FILE
./exp_scaling 20 8 10 0 >> $FILE
./exp_scaling 20 8 20 0 >> $FILE
./exp_scaling 20 8 40 0 >> $FILE
./exp_scaling 20 8 80 0 >> $FILE
./exp_scaling 16 8 10 1 >> $FILE
./exp_scaling 16 8 20 1 >> $FILE
./exp_scaling 16 8 40 1 >> $FILE
./exp_scaling 16 8 80 1 >> $FILE
./exp_scaling 18 8 10 1 >> $FILE
./exp_scaling 18 8 20 1 >> $FILE
./exp_scaling 18 8 40 1 >> $FILE
./exp_scaling 18 8 80 1 >> $FILE
./exp_scaling 20 8 10 1 >> $FILE
./exp_scaling 20 8 20 1 >> $FILE
./exp_scaling 20 8 40 1 >> $FILE
./exp_scaling 20 8 80 1 >> $FILE
