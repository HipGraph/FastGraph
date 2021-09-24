#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

FILE="window.csv"
echo "swindow,stime,cwindow,ctime,matrix,row-scale,col-scale,d,k,thread,algorithm,window,total,nnz-in,nnz-out" > $FILE

for RSCALE in 22
do
    for CSCALE in 10
    do
        for K in 128
        do
            for D in 64 16384
            do
                for T in 48
                do
                    for W in 4194304 2097152 1048576 262144 65536 32768 16384 8192 4096 2048 1024 512
                    do
                        echo ./window $RSCALE $CSCALE $D $K 0 $T $W
                        ./window $RSCALE $CSCALE $D $K 0 $T $W >> $FILE
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
        done
    done
done

for RSCALE in 22
do
    for CSCALE in 10
    do
        for K in 128
        do
            for D in 8 512 
            do
                for T in 48
                do
                    for W in 4194304 2097152 1048576 262144 65536 32768 16384 8192 4096 2048 1024 512
                    do
                        echo ./window $RSCALE $CSCALE $D $K 1 $T $W
                        ./window $RSCALE $CSCALE $D $K 1 $T $W >> $FILE
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
        done
    done
done

DATA_PATH="/N/u2/t/taufique/Data/r0_s"
for RSCALE in 22
do
    for CSCALE in 10
    do
        for K in 64
        do
            for D in 12345
            do
                for T in 48
                do
                    for W in 4194304 2097152 1048576 262144 65536 32768 16384 8192 4096 2048 1024 512
                    do
                        echo ./window $RSCALE $CSCALE $D $K 2 $T $DATA_PATH $W
                        ./window $RSCALE $CSCALE $D $K 2 $T $DATA_PATH $W >> $FILE
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
        done
    done
done
