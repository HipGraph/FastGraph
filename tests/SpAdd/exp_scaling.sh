#!/bin/bash -l

#SBATCH -p azad
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J spadd
#SBATCH -o spadd.o%j

export OMP_NUM_THREADS=48

#export FILE=hash.csv
#export FILE=heap.csv
export FILE=mkl.csv
#export FILE=dense.csv
#export FILE=pairwise.csv
#export FILE=window.csv

#export BIN=./binary
#export BIN=./heap
export BIN=./mkl

echo "matrix,scale,d,k,thread,algorithm,total" > $FILE
#echo "matrix,scale,d,k,thread,algorithm,threshold,count,time" > $FILE
#echo "matrix,scale,d,k,thread,algorithm,window,symbolic,computation,total" > $FILE

$BIN 16 8 10 0 >> $FILE
$BIN 16 8 20 0 >> $FILE
$BIN 16 8 40 0 >> $FILE
$BIN 16 8 80 0 >> $FILE
$BIN 18 8 10 0 >> $FILE
$BIN 18 8 20 0 >> $FILE
$BIN 18 8 40 0 >> $FILE
$BIN 18 8 80 0 >> $FILE
$BIN 20 8 10 0 >> $FILE
$BIN 20 8 20 0 >> $FILE
$BIN 20 8 40 0 >> $FILE
$BIN 20 8 80 0 >> $FILE
$BIN 16 8 10 1 >> $FILE
$BIN 16 8 20 1 >> $FILE
$BIN 16 8 40 1 >> $FILE
$BIN 16 8 80 1 >> $FILE
$BIN 18 8 10 1 >> $FILE
$BIN 18 8 20 1 >> $FILE
$BIN 18 8 40 1 >> $FILE
$BIN 18 8 80 1 >> $FILE
$BIN 20 8 10 1 >> $FILE
$BIN 20 8 20 1 >> $FILE
$BIN 20 8 40 1 >> $FILE
$BIN 20 8 80 1 >> $FILE

$BIN 16 32 10 0 >> $FILE
$BIN 16 32 20 0 >> $FILE
$BIN 16 32 40 0 >> $FILE
$BIN 16 32 80 0 >> $FILE
$BIN 18 32 10 0 >> $FILE
$BIN 18 32 20 0 >> $FILE
$BIN 18 32 40 0 >> $FILE
$BIN 18 32 80 0 >> $FILE
$BIN 20 32 10 0 >> $FILE
$BIN 20 32 20 0 >> $FILE
$BIN 20 32 40 0 >> $FILE
$BIN 20 32 80 0 >> $FILE
$BIN 16 32 10 1 >> $FILE
$BIN 16 32 20 1 >> $FILE
$BIN 16 32 40 1 >> $FILE
$BIN 16 32 80 1 >> $FILE
$BIN 18 32 10 1 >> $FILE
$BIN 18 32 20 1 >> $FILE
$BIN 18 32 40 1 >> $FILE
$BIN 18 32 80 1 >> $FILE
$BIN 20 32 10 1 >> $FILE
$BIN 20 32 20 1 >> $FILE
$BIN 20 32 40 1 >> $FILE
$BIN 20 32 80 1 >> $FILE

#$BIN 20 64 40 0 >> $FILE
#$BIN 20 128 40 0 >> $FILE
