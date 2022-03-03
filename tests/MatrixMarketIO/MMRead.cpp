#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "../../include/CSC.h"
#include "../../include/COO.h"
#include "../../include/GAP/pvector.h"
#include "../../include/GAP/timer.h"
#include "../../include/CSC_adder.h"
#include "../../include/utils.h"

//#include "mmio.h"

int main(int argc, char* argv[]){
    std::string filename = std::string(argv[1]);
    COO<uint32_t, uint32_t, float> coo;
    coo.ReadMM(filename);
    CSC<uint32_t, float, uint32_t> csc(coo);

    csc.PrintInfo();
    std::cout<<"ewise"<<std::endl;
    //csc.ewiseApply(2);

    std::cout<<"deemapply here"<<std::endl;
    size_t n=csc.get_ncols();
    std::vector<int> column_vector(n, 2);
    //pvector vect(n,1);
    //pvector<int32_t> column_vector(n+1);
    
    //csc.dimApply(column_vector);
    csc.column_reduce(column_vector);
    //csc.deemApply(vect);
	return 0;

}
