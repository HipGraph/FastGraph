#include <iostream>
#include <vector>

#include "../include/CSC.h" // need to check the relative paths for this section
#include "../include/COO.h"
#include "../include/GAP/pvector.h"
#include "../include/GAP/timer.h"
#include "../include/CSC_adder.h"
#include "../include/utils.h"

/*
intro: file to test CSC_adder.h
usage: find <int option> at the start of int main in this file and assign option to the function {id}(from CSC_adder.h in include folder) intended to be used

how to run:
while being in include folder and then type following commands

g++ ../tests/CSC_adder_io.h -fopenmp
./a.out < 1.in > 1.txt
diff 1.out 1.txt

where diff should not give any difference and

1.in has number_of_matrices as first line and then has the format between {.} for 1st matrix, while next (number_of_matrices-1) matrices follow same pattern

{<no_of_rows> <no_of_cols> <nnz>
<colptr_array>(space between each number)
<row_ids array>(space between each number)
<nz_vals array>(space between each number)
}// each line meaning a new line and no double new lines between any two matrices

1.out has the same format as above between {.} while having a new line at the end after nz_vals array

1.in, 1.out formats are taken care, if you use rand_matrix_gen.m

and finally, assign show_time to true for time of execution information, but mind that it will give out some diff if the above commands are still followed to test
*/

int main(){

	int option = 2;
/*
option = 
1 => unordered_map with push_back
2 => unordered_map with push_back and locks
3 => unordered_map with symbolic step
4 => heaps with symbolic step
5 => radix sort with symbolic step
6 => row_size_pvector_maintaining_sum with symbolic step
*/ 
	bool show_time = false;

	int k; // number of matrices
	std::cin>>k;


	std::vector< CSC<int32_t, int32_t, int32_t> > vec_1(k);
	std::vector< CSC<int32_t, int32_t, int32_t>* > vec(k);

	// below is method to use random matrices from COO.h
	// for(int i = 0; i < k; i++){
	// 	COO<int32_t, int32_t, double> Acoo;
	// 	Acoo.GenER(2,2,true);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
	// 	//CSC<int32_t, double, int32_t> Acsc(Acoo);
	// 	//vec[i] = &Acsc;
	// 	vec[i] = new CSC<int32_t, double, int32_t>(Acoo);

	// 	//Acoo.print_all();
	// 	//  std::cout<<std::endl;
	// 	//  vec[i]->print_all();
	// 	// std::cout<<std::endl;
	// }


	// input file as <no_of_rows> <no_of_cols> <nnz> <colptr_array> <row_ids> <nz_vals>
	int rows, cols, nnz;
	for(int i = 0; i < k; i++){
		std::cin >> rows >> cols >> nnz;
		CSC<int32_t, int32_t, int32_t> csc(rows, cols, nnz, true, true);
		pvector<int32_t> column_vector(cols+1);
		pvector<int32_t> row_vector(nnz);
		pvector<int32_t> value_vector(nnz);
		for(int j = 0; j < cols+1; j++){
			std::cin>>column_vector[j];
		}
		for(int j = 0; j < nnz; j++){
			std::cin>>row_vector[j];
		}
		for(int j = 0; j < nnz; j++){
			std::cin>>value_vector[j];
		}
		csc.nz_rows_pvector(&row_vector);
		csc.cols_pvector(&column_vector);
		csc.nz_vals_pvector(&value_vector);
		vec_1[i] = std::move(csc);
		vec[i] = &vec_1[i];

		//std::cout<<vec[i]->get_nrows()<<" "<<vec[i]->get_ncols()<<" "<<vec[i]->get_nnz()<<" "<<std::endl;
	}

	Timer clock;
	//std::cout<<std::endl;

	if(option == 1){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_1 = add_vec_of_matrices_1<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_1 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_1.print_all();
	}else if(option == 2){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_2 = add_vec_of_matrices_2<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_2 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_2.print_all();
	}else if(option == 3){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_3 = add_vec_of_matrices_3<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_3 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_3.print_all();
	}else if(option == 4){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_4 = add_vec_of_matrices_4<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_4 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_4.print_all();
	}else if(option == 5){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_5 = add_vec_of_matrices_5<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_5 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_5.print_all();
	}else if(option == 6){
		clock.Start();
		CSC<int32_t, int32_t, int32_t> result_6 = add_vec_of_matrices_6<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
		clock.Stop();
		if(show_time){
			std::cout<<"time for add_vec_of_matrices_6 function in seconds = "<< clock.Seconds()<<std::endl;
		}
		result_6.print_all();
	}else{

	}

	return 0;

}