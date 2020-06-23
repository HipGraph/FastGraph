#include <iostream>
#include <vector>
#include <string>

#include "../../include/CSC.h"
#include "../../include/COO.h"
#include "../../include/GAP/pvector.h"
#include "../../include/GAP/timer.h"
#include "../../include/CSC_adder.h"
#include "../../include/utils.h"

#include "mkl.h"
#include "mkl_spblas.h"

int main(){
	int x = 10; // scale of radom matrix
    int y = 4;  // average degree of random matrix
	bool weighted = true;

	int k = 10; // number of matrices

	std::vector< CSC<int32_t, int32_t, int32_t>* > vec;

    // below is method to use random matrices from COO.h
    for(int i = 0; i < k; i++){
        COO<int32_t, int32_t, int32_t> coo;
        coo.GenER(x,y,weighted);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
        //coo.GenRMAT(x,y,weighted);   //(x,y,true) Generate a weighted RMAT matrix with 2^x rows and columns and y nonzeros per column
        vec.push_back(new CSC<int32_t, int32_t, int32_t>(coo));
    }
    
    sparse_matrix_t A = NULL;
    struct matrix_descr Adsc;
    MKL_INT *mkl_A_colPtr;
    MKL_INT *mkl_A_rowIds;
    double *mkl_A_nzVal;

    pvector<int32_t>* csc_A_colPtr = vec[0]->get_colPtr(); 
    mkl_A_colPtr = (MKL_INT*) malloc( ( csc_A_colPtr->size() ) * sizeof(MKL_INT) );
    for(int i = 0; i < csc_A_colPtr->size(); i++){
        mkl_A_colPtr[i] = (*csc_A_colPtr)[i];
    }

    pvector<int32_t>* csc_A_rowIds = vec[0]->get_rowIds(); 
    mkl_A_rowIds = (MKL_INT*) malloc( ( csc_A_rowIds->size() ) * sizeof(MKL_INT) );
    for(int i = 0; i < csc_A_rowIds->size(); i++){
        mkl_A_rowIds[i] = (*csc_A_rowIds)[i];
    }

    pvector<int32_t>* csc_A_nzVals = vec[0]->get_nzVals(); 
    mkl_A_nzVals = (double*) malloc( ( csc_A_nzVals->size() ) * sizeof(double) );
    for(int i = 0; i < csc_A_nzVals->size(); i++){
        mkl_A_nzVals[i] = (double)(*csc_A_nzVals)[i];
    }

    sparse_status_t stat = mkl_sparse_d_create_csc (
        &A, 
        SPARSE_INDEX_BASE_ZERO, 
        vec[0]->get_nrows(), 
        vec[0]->get_ncols(), 
        mkl_A_colPtr, 
        mkl_A_colPtr + 1, 
        mkl_A_rowIds, 
        mkl_A_nzVals
    );
    if (stat == SPARSE_STATUS_SUCCESS){
        printf("MKL Sparse Matrix creation successful\n");
    }

    //sparse_matrix_t B = NULL;
    //sparse_matrix_t C = NULL;

    //Timer clock;
    //int threads[3] = {48, 24, 1};

    //for(int i = 0; i < 3; i++){
        //omp_set_num_threads(threads[i]);
        //std::cout << "Using " << threads[i] << " threads" << std::endl;

        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_1 = add_vec_of_matrices_1<int32_t, int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_1 function in seconds = "<< clock.Seconds()<<std::endl;
        
        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_2 = add_vec_of_matrices_2<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_2 function in seconds = "<< clock.Seconds()<<std::endl;

        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_3 = add_vec_of_matrices_3<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_3 function in seconds = "<< clock.Seconds()<<std::endl;

        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_4 = add_vec_of_matrices_4<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_4 function in seconds = "<< clock.Seconds()<<std::endl;

        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_5 = add_vec_of_matrices_5<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_5 function in seconds = "<< clock.Seconds()<<std::endl;

        //clock.Start();
        //CSC<int32_t, int32_t, int32_t> result_6 = add_vec_of_matrices_6<int32_t,int32_t, int32_t,int32_t,int32_t> (vec);
        //clock.Stop();
        //std::cout<<"time for add_vec_of_matrices_6 function in seconds = "<< clock.Seconds()<<std::endl;
    //}

	return 0;

}
