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
    CSC<uint32_t, float, uint32_t> csc1(coo);


    
    //std::cout<<"ewise"<<std::endl;
    //csc.ewiseApply(2);
    
    //std::cout<<"DimApply1 here"<<std::endl;
    size_t n=csc.get_ncols();
    std::vector<int> column_vector(n, 2);
    //pvector vect(n,1);
    pvector<float> column_vector_1(n);
    //pvector<float> column_vector_1(n);


    pvector<float> column_reduce_vector(n);
    // pvector<float> column_reduce_vector(n);
    
    for(int j = 0; j < n; j++)
    {
		column_vector_1[j]=2;
	}
    
    //csc.dimApply(column_vector_1);

    //pvector

    //csc.dimApply(column_vector);
    //csc.dimApply1(column_vector_1);
    //csc.dimApply1(column_vector_1);


    // std::cout<<"Column reduce here "<<std::endl;
    // csc.column_reduce();
    //std::cout<<"Mat Addition here: "<<std::endl;
    //csc.matAddition_1(csc1);
    std::cout<<"Information for the 2 matrices.."<<std::endl;
    csc.PrintInfo();
    std::cout<<"Mat Addition here..."<<std::endl;
    //csc.matAddition(csc1);
    //csc.matAddition_1(csc1);
    //csc.matAddition_2(csc1);
    csc.matAddition_2(csc1);
    //csc.matAddition_3(csc1);
    //csc.PrintInfo();






    //column_reduce_vector=csc.column_reduce_1();
    //column_reduce_vector=csc.column_reduce_1();
	return 0;

}
