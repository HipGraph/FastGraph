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

	return 0;

}
