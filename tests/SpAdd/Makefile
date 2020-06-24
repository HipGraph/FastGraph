#MKLROOT = /opt/intel/mkl
MKLROOT = /N/u2/t/taufique/intel/mkl
CC = g++ -fopenmp -std=c++11
#CC = g++ -fopenmp -std=c++11 -g -Og -mavx
INC = ${MKLROOT}/include
#LIB = ${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
LIB = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	  -lpthread -lm -ldl

exp_scaling.o: exp_scaling.cpp
	$(CC) -I$(INC) -c $< -o $@

exp_scaling: exp_scaling.o
	$(CC) $^ -o $@ $(LIB)

all: exp_scaling

clean:
	rm exp_scaling.o exp_scaling
