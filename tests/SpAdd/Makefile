MKLROOT = /opt/intel/mkl
CC = g++ -fopenmp -std=c++11
INC = ${MKLROOT}/include
LIB = ${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

exp_scaling.o: exp_scaling.cpp
	$(CC) -I$(INC) -c $< -o $@

exp_scaling: exp_scaling.o
	$(CC) $^ -o $@ -L$(LIB)

all: exp_scaling

clean:
	rm exp_scaling.o exp_scaling
