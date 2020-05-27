#include <iostream>
#include <vector>


#include "../include/CSC.h" // need to check the relative paths for this section
#include "../include/COO.h"
#include "../include/GAP/pvector.h"
#include "../include/GAP/timer.h"
#include "../include/CSC_adder.h"


int main(){

	int k = 7; // number of matrices
	std::vector< CSC<int32_t, double, int32_t>* > vec(k);
	for(int i = 0; i < k; i++){
		COO<int32_t, int32_t, double> Acoo;
		Acoo.GenER(2,2,true);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
		vec[i] = new CSC<int32_t, double, int32_t>(Acoo);

		// Acoo.print_all();
		// std::cout<<std::endl;
		// vec[i]->print_all();
		// std::cout<<std::endl;
	}
	Timer clock;
	std::cout<<std::endl;
	clock.Start();
	CSC<int32_t, double, int32_t> result = add_vec_of_matrices<int32_t,int32_t, double,int32_t,int32_t> (vec);
	clock.Stop();
	std::cout<<"time for add_vec_of_matrices function in seconds = "<< clock.Seconds()<<std::endl;
	result.print_all();

	clock.Start();
	CSC<int32_t, double, int32_t> result_ls = add_vec_of_matrices_ls<int32_t,int32_t, double,int32_t,int32_t> (vec);
	clock.Stop();
	std::cout<<"time for add_vec_of_matrices_ls function in seconds = "<< clock.Seconds()<<std::endl;
	result_ls.print_all();

	return 0;

}