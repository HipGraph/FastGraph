#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

RSCALE=22
CSCALE=22
K=128
FILE="rmat-scaling.csv"
echo "matrix,row-scale,col-scale,d,k,thread,algorithm,total,nnz-in,nnz-out" > $FILE

for ALG in hash-regular-dynamic hash-sliding-dynamic pairwise-tree-dynamic pairwise-serial-dynamic mkl-serial mkl-tree
do
    for D in 16
    do
        for T in 48 24 12 1
        do
            echo ./$ALG $RSCALE $CSCALE $D $K 1 $T
            ./$ALG $RSCALE $CSCALE $D $K 1 $T >> $FILE
            if [ $? -eq 0 ]
            then
                echo OK
            else
                echo FAIL $?
            fi
            echo ---
        done
    done
done
