#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

RSCALE=22
CSCALE=10
K=128
FILE="er-scaling.csv"
#echo "matrix,row-scale,col-scale,d,k,thread,algorithm,total,nnz-in,nnz-out" > $FILE

#for ALG in hash-regular-static hash-sliding-static pairwise-tree-static pairwise-serial-static mkl-serial mkl-tree
for ALG in heap-static spa-static
do
    for D in 1024 16 2048
    do
        for T in 48 24 12 1
        do
            echo ./$ALG $RSCALE $CSCALE $D $K 0 $T
            ./$ALG $RSCALE $CSCALE $D $K 0 $T >> $FILE
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
