#include <iostream>
#include <vector>
#include <string>
#include <cmath>


#include "../../../csrc/common/CSC.h"
#include "../../../csrc/common/COO.h"
#include "../../../csrc/common/GAP/pvector.h"
#include "../../../csrc/common/GAP/timer.h"
#include "../../../csrc/common/CSC_adder.h"
#include "../../../csrc/common/utils.h"





using namespace std::chrono;





int main(int argc, char* argv[]){
    std::string filename = std::string(argv[1]);
    COO<uint32_t, uint32_t, float> coo;
    coo.ReadMM(filename);
    CSC<uint32_t, float, uint32_t> csc(coo);
    CSC<uint32_t, float, uint32_t> csc1(coo);


    std::cout<<"Information for the matrix.."<<std::endl;
    csc.PrintInfo();
    // std::cout<<"Mat Addition here..."<<std::endl;
    
    // auto start = high_resolution_clock::now();
    // csc.matAddition_2(csc1);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
 
    // std::cout << "Time taken by function:"<< duration.count() << " microseconds" <<std:: endl;
    std::cout<<"Column Reduce."<<std::endl;
    //csc.column_reduce();
    csc.column_reduce();

    //testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();
    
	return 0;

}
