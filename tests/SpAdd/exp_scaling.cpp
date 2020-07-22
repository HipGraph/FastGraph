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
	int x = 20; // scale of radom matrix
    int y = 8;  // average degree of random matrix
	bool weighted = true;

	int k = 100; // number of matrices

	std::vector< CSC<int32_t, int32_t, int32_t>* > vec;

    // below is method to use random matrices from COO.h
    for(int i = 0; i < k; i++){
        COO<int32_t, int32_t, int32_t> coo;
        coo.GenER(x,y,weighted);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
        //coo.GenRMAT(x,y,weighted);   //(x,y,true) Generate a weighted RMAT matrix with 2^x rows and columns and y nonzeros per column
        vec.push_back(new CSC<int32_t, int32_t, int32_t>(coo));
    }
    
    Timer clock;

    /*
     *  Intel MKL specific codes
     * */
    //MKL_INT** mkl_rowPtr = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    //MKL_INT** mkl_colIds = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    //double** mkl_nzVals = (double**) malloc( k * sizeof(double*) );

    double** mkl_values = (double**) malloc( k * sizeof(double*) );
    MKL_INT** mkl_rows = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    MKL_INT** mkl_pointerB = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    MKL_INT** mkl_pointerE = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );

    sparse_matrix_t* mkl_csc_matrices = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
    sparse_matrix_t* mkl_csr_matrices = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
    sparse_matrix_t* mkl_sums = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );

    for(int i = 0; i < k; i++){
        mkl_values[i] = NULL;
        mkl_rows[i] = NULL;
        mkl_pointerB[i] = NULL;
        mkl_pointerE[i] = NULL;
        mkl_csc_matrices[i] = NULL;
        mkl_csr_matrices[i] = NULL;
        mkl_sums[i] = NULL;
    }

//#pragma omp parallel
    for(int i = 0; i < k; i++){
        //vec[i]->print_all();
        // For now
        // Treat created CSC matrix as CSR
        // And create MKL CSR matrix from that
        
        auto csc_nzVals = vec[i]->get_nzVals(); 
        mkl_values[i] = (double*) malloc( ( csc_nzVals->size() ) * sizeof(double) );
        for(int j = 0; j < csc_nzVals->size(); j++){
            mkl_values[i][j] = (double) (*csc_nzVals)[j];
        }

        auto csc_rowIds = vec[i]->get_rowIds(); 
        mkl_rows[i] = (MKL_INT*) malloc( ( csc_rowIds->size() ) * sizeof(MKL_INT) );
        for(int j = 0; j < csc_rowIds->size(); j++){
            mkl_rows[i][j] = (MKL_INT) (*csc_rowIds)[j];
        }

        auto csc_colPtr = vec[i]->get_colPtr();
        mkl_pointerB[i] = (MKL_INT*) malloc( ( csc_colPtr->size() ) * sizeof(MKL_INT) );
        mkl_pointerE[i] = (MKL_INT*) malloc( ( csc_colPtr->size() ) * sizeof(MKL_INT) );
        for(int j = 0; j < csc_colPtr->size(); j++){
            if(j == 0){
                mkl_pointerB[i][j] = (MKL_INT) (*csc_colPtr)[j];
            }
            else if(j == csc_colPtr->size()-1){
                mkl_pointerE[i][j-1] = (MKL_INT) (*csc_colPtr)[j];
            }
            else{
                mkl_pointerB[i][j] = (MKL_INT) (*csc_colPtr)[j];
                mkl_pointerE[i][j-1] = (MKL_INT) (*csc_colPtr)[j];
            }
        }

        printf("Creating MKL CSR matrix %d: ", i);
        sparse_status_t create_status = mkl_sparse_d_create_csc (
            &(mkl_csc_matrices[i]), 
            SPARSE_INDEX_BASE_ZERO, 
            vec[i]->get_nrows(), 
            vec[i]->get_ncols(), 
            mkl_pointerB[i], 
            mkl_pointerE[i], 
            mkl_rows[i], 
            mkl_values[i]
        );
        switch(create_status){
            case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        }
        printf("\n");

        printf("Converting MKL CSC matrix %d to CSR: ", i);
        sparse_status_t conv_status = mkl_sparse_convert_csr (
            mkl_csc_matrices[i],
            SPARSE_OPERATION_NON_TRANSPOSE,
            &(mkl_csr_matrices[i])
        );
        switch(conv_status){
            case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        }
        printf("\n");
    }
    printf("MKL sparse matrices created\n");




    int threads[3] = {48, 24, 1};

    for(int i = 0; i < 3; i++){
        omp_set_num_threads(threads[i]);
        mkl_set_num_threads(threads[i]);
        std::cout << "Using " << threads[i] << " threads" << std::endl;

        clock.Start();
        for(int i = 1; i < k; i++){
            sparse_status_t add_status;
            //printf("Adding matrix %d: ", i);
            if(i == 1){
                add_status = mkl_sparse_d_add(
                    SPARSE_OPERATION_NON_TRANSPOSE, 
                    mkl_csr_matrices[i-1], 
                    1.0, 
                    mkl_csr_matrices[i], 
                    &(mkl_sums[i])
                );
            }
            else{
                add_status = mkl_sparse_d_add(
                    SPARSE_OPERATION_NON_TRANSPOSE, 
                    mkl_sums[i-1], 
                    1.0, 
                    mkl_csr_matrices[i], 
                    &(mkl_sums[i])
                );
            }
            //switch(add_status){
                //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
                //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
                //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
                //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
                //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
                //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
                //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
            //}
            //printf("\n");
        }
        clock.Stop();
        std::cout << "time for MKL in seconds " << clock.Seconds() << std::endl;
        for (int i = 0; i < k; i++){
           if(mkl_sums[i] != NULL) mkl_sparse_destroy(mkl_sums[i]);
        }


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

        clock.Start();
        CSC<int32_t, int32_t, int32_t> result_3_1 = SpAddHash<int32_t,int32_t, int32_t,int32_t> (vec);
        clock.Stop();
        std::cout<<"time for SpAddHash function in seconds = "<< clock.Seconds()<<std::endl;

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
    }

    for (int i = 0; i < k; i++){
       if(mkl_values[i] != NULL) free(mkl_values[i]); 
       if(mkl_rows[i] != NULL) free(mkl_rows[i]); 
       if(mkl_pointerB[i] != NULL) free(mkl_pointerB[i]);
       if(mkl_pointerE[i] != NULL) free(mkl_pointerE[i]);
       if(mkl_csc_matrices[i] != NULL) mkl_sparse_destroy(mkl_csc_matrices[i]);
       if(mkl_csr_matrices[i] != NULL) mkl_sparse_destroy(mkl_csr_matrices[i]);
       //if(mkl_sums[i] != NULL) mkl_sparse_destroy(mkl_sums[i]);
    }
    if(mkl_values != NULL) free(mkl_values);
    if(mkl_rows != NULL) free(mkl_rows);
    if(mkl_pointerB != NULL) free(mkl_pointerB);
    if(mkl_pointerE != NULL) free(mkl_pointerE);
    if(mkl_csc_matrices != NULL) free(mkl_csc_matrices);
    if(mkl_csr_matrices != NULL) free(mkl_csr_matrices);
    if(mkl_sums != NULL) free(mkl_sums);

	return 0;

}
