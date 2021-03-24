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

#include "mkl.h"
#include "mkl_spblas.h"

int main(int argc, char* argv[]){
    int x = atoi(argv[1]); // scale of random matrix
    int y = atoi(argv[2]);  // average degree of random matrix
	bool weighted = true;

	int k = atoi(argv[3]);// number of matrices
    int type = atoi(argv[4]);
    //if(type == 0){
        //printf("Using %d ER matrices of scale %d, avg degree %d\n", k, x, y);
    //}
    //else{
        //printf("Using %d RMAT matrices of scale %d, avg degree %d\n", k, x, y);
    //}

	std::vector< CSC<int32_t, int32_t, int32_t>* > vec;

    // below is method to use random matrices from COO.h
    for(int i = 0; i < k; i++){
        COO<int32_t, int32_t, int32_t> coo;
        if(type == 0){
            coo.GenER(x,y,weighted, i);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
        }
        else{
            coo.GenRMAT(x,y,weighted, i);   //(x,y,true) Generate a weighted RMAT matrix with 2^x rows and columns and y nonzeros per column
        }
        vec.push_back(new CSC<int32_t, int32_t, int32_t>(coo));
        //vec[i]->print_all();
    }
    
    Timer clock;

    // MKL specific codes
     
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

        //printf("Creating MKL CSR matrix %d: ", i);
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
        //switch(create_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");

        //printf("Converting MKL CSC matrix %d to CSR: ", i);
        sparse_status_t conv_status = mkl_sparse_convert_csr (
            mkl_csc_matrices[i],
            SPARSE_OPERATION_NON_TRANSPOSE,
            &(mkl_csr_matrices[i])
        );
        //switch(conv_status){
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
    //printf("MKL sparse matrices created\n");
    //printf("\n");


    //std::vector<int> threads{1, 6, 12, 24, 48};
    //std::vector<int> threads{1, 16, 48};
    std::vector<int> threads{24};

    for(int i = 0; i < threads.size(); i++){
        omp_set_num_threads(threads[i]);
        mkl_set_num_threads(threads[i]);
        //std::cout << ">>> Using " << threads[i] << " threads" << std::endl;

        //clock.Start();
        //for(int j = 1; j < k; j++){
            //sparse_status_t add_status;
            ////printf("Adding matrix %d: ", j);
            //if(j == 1){
                //add_status = mkl_sparse_d_add(
                    //SPARSE_OPERATION_NON_TRANSPOSE, 
                    //mkl_csr_matrices[j-1], 
                    //1.0, 
                    //mkl_csr_matrices[j], 
                    //&(mkl_sums[j])
                //);
            //}
            //else{
                //add_status = mkl_sparse_d_add(
                    //SPARSE_OPERATION_NON_TRANSPOSE, 
                    //mkl_sums[j-1], 
                    //1.0, 
                    //mkl_csr_matrices[j], 
                    //&(mkl_sums[j])
                //);
            //}
            ////switch(add_status){
                ////case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
                ////case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
                ////case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
                ////case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
                ////case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
                ////case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
                ////case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
            ////}
            ////printf("\n");
        //}
        //clock.Stop();
        ////std::cout << "time for MKL in seconds " << clock.Seconds() << std::endl;
        //if(type == 0){
            //std::cout << "ER" << "," ;
        //}
        //else{
            //std::cout << "RMAT" << "," ;
        //}
        //std::cout << x << "," ;
        //std::cout << y << "," ;
        //std::cout << k << "," ;
        //std::cout << threads[i] << ",";
        //std::cout << "mkl_sparse_d_add" << ","; 
        //std::cout << clock.Seconds() << std::endl;
        
        CSC<int32_t, int32_t, int32_t> SpAddHash_out;

        //clock.Start();
        //std::vector<size_t> intermediate_nnz;
        //CSC<int32_t, int32_t, int32_t> SpAddHash_out = SpAdd<int32_t,int32_t, int32_t,int32_t> (vec[0], vec[1]);
        //intermediate_nnz.push_back(SpAddHash_out.get_nnz());
        //for (int j = 2; j < k; j++){
            //SpAddHash_out = SpAdd<int32_t,int32_t,int32_t,int32_t>(&SpAddHash_out, vec[j]);
            //intermediate_nnz.push_back(SpAddHash_out.get_nnz());
        //}
        //clock.Stop();
        //if(type == 0){
            //std::cout << "ER" << "," ;
        //}
        //else{
            //std::cout << "RMAT" << "," ;
        //}
        //std::cout << x << "," ;
        //std::cout << y << "," ;
        //std::cout << k << "," ;
        //std::cout << threads[i] << ",";
        //std::cout << "SpAdd" << ","; 
        //std::cout << clock.Seconds() << std::endl;
        ////std::cout<<"time for SpAdd function in seconds = "<< clock.Seconds()<<std::endl;
        ////std::ofstream fp;
        ////fp.open("pairwise-spadd.txt", std::ios::trunc);
        ////for (int j = 0 ; j < intermediate_nnz.size(); j++){
            ////fp << intermediate_nnz[j] << std::endl;
        ////}
        ////fp.close();
        ////SpAddHash_out.print_all();
        
        clock.Start(); 
        SpAddHash_out = SpMultiAdd<int32_t,int32_t, int32_t,int32_t> (vec,0);
        clock.Stop();
        if(type == 0){
            std::cout << "ER" << "," ;
        }
        else{
            std::cout << "RMAT" << "," ;
        }
        std::cout << x << "," ;
        std::cout << y << "," ;
        std::cout << k << "," ;
        std::cout << threads[i] << ",";
        std::cout << "SpMultiAddHash" << ","; 
        std::cout << clock.Seconds() << std::endl;
        //SpAddHash_out.print_all();
        
        //clock.Start(); 
        //SpAddHash_out = SpMultiAdd<int32_t,int32_t, int32_t,int32_t> (vec,1);
        //clock.Stop();
        //if(type == 0){
            //std::cout << "ER" << "," ;
        //}
        //else{
            //std::cout << "RMAT" << "," ;
        //}
        //std::cout << x << "," ;
        //std::cout << y << "," ;
        //std::cout << k << "," ;
        //std::cout << threads[i] << ",";
        //std::cout << "SpMultiAddHybrid" << ","; 
        //std::cout << clock.Seconds() << std::endl;

        //clock.Start(); 
        //SpAddHash_out = SpMultiAdd<int32_t,int32_t, int32_t,int32_t> (vec,2);
        //clock.Stop();
        //if(type == 0){
            //std::cout << "ER" << "," ;
        //}
        //else{
            //std::cout << "RMAT" << "," ;
        //}
        //std::cout << x << "," ;
        //std::cout << y << "," ;
        //std::cout << k << "," ;
        //std::cout << threads[i] << ",";
        //std::cout << "SpMultiAddHybrid2" << ","; 
        //std::cout << clock.Seconds() << std::endl;

        clock.Start(); 
        CSC<int32_t, int32_t, int32_t> SpAddHybrid_out = SpMultiAdd<int32_t,int32_t, int32_t,int32_t> (vec,4);
        clock.Stop();
        if(type == 0){
            std::cout << "ER" << "," ;
        }
        else{
            std::cout << "RMAT" << "," ;
        }
        std::cout << x << "," ;
        std::cout << y << "," ;
        std::cout << k << "," ;
        std::cout << threads[i] << ",";
        std::cout << "SpMultiAddHashSliding" << ","; 
        std::cout << clock.Seconds() << std::endl;
        //SpAddHybrid_out.print_all();

        auto SpAddHash_colPtr = SpAddHash_out.get_colPtr();
        auto SpAddHash_rowIds = SpAddHash_out.get_rowIds();
        auto SpAddHash_nzVals = SpAddHash_out.get_nzVals();
        auto SpAddHash_ncols = SpAddHash_out.get_ncols();
        auto SpAddHash_nrows = SpAddHash_out.get_nrows();
        auto SpAddHybrid_colPtr = SpAddHybrid_out.get_colPtr();
        auto SpAddHybrid_rowIds = SpAddHybrid_out.get_rowIds();
        auto SpAddHybrid_nzVals = SpAddHybrid_out.get_nzVals();
        auto SpAddHybrid_ncols = SpAddHybrid_out.get_ncols();
        auto SpAddHybrid_nrows = SpAddHybrid_out.get_nrows();
        if(SpAddHash_ncols == SpAddHybrid_ncols){
            bool flag = true;
            for(int32_t i = 0; i < SpAddHash_colPtr->size(); i++){
                if((*SpAddHash_colPtr)[i] != (*SpAddHybrid_colPtr)[i]){
                    printf("colPtr[%d]\tHash: %d - Hybrid: %d\n", i, (*SpAddHash_colPtr)[i], (*SpAddHybrid_colPtr)[i]);
                    flag = false;
                }
            }
            for(int32_t j = 0; j < SpAddHash_rowIds->size(); j++){
                if((*SpAddHash_rowIds)[j] != (*SpAddHybrid_rowIds)[j]){
                    printf("rowIds[%d]\tHash: %d - Hybrid: %d\n", j, (*SpAddHash_rowIds)[j], (*SpAddHybrid_rowIds)[j]);
                    flag = false;
                }
            }
            for(int32_t j = 0; j < SpAddHash_nzVals->size(); j++){
                if((*SpAddHash_nzVals)[j] != (*SpAddHybrid_nzVals)[j]){
                    printf("nzVals[%d]\tHash: %d - Hybrid: %d\n", j, (*SpAddHash_nzVals)[j], (*SpAddHybrid_nzVals)[j]);
                    flag = false;
                }
            }
            if(flag == true){
                printf("Everything matched! \n");
            }
        }
        else{
            printf("Number of columns not equal!!!\nAborting further check.\n");
        }


        //double t0, t1, t2, t3;
        //t0 = omp_get_wtime();
        //pvector<int32_t> nnzCPerCol = symbolicSpMultiAddHash<int32_t, int32_t, int32_t, int32_t, int32_t>(vec);
        //t1 = omp_get_wtime();
        //printf("Time for symbolic with pure hash: %lf\n", t1-t0);
        
        //t0 = omp_get_wtime();
        //pvector<int32_t> nnzCPerCol2 = symbolicSpMultiAddHashSliding2<int32_t, int32_t, int32_t, int32_t, int32_t>(vec);
        //t1 = omp_get_wtime();
        //printf("Time for symbolic with sliding hash: %lf\n", t1-t0);
        
        //for(int32_t i=0; i< nnzCPerCol.size(); i++)
        //{
            //if(nnzCPerCol[i] != nnzCPerCol2[i]) std::cout << "not equal" << std::endl;
        //}
        //printf("Symbolic Equal!\n");

        
        
        //printf("Transposing MKL output: ");
        //sparse_matrix_t *mkl_out = (sparse_matrix_t *) malloc( sizeof(sparse_matrix_t) );
        //sparse_status_t conv_status = mkl_sparse_convert_csr(
            //mkl_sums[k-1],
            //SPARSE_OPERATION_TRANSPOSE, // Transpose because it will make CSR matrix to be effectively CSC
            //mkl_out
        //);
        //switch(conv_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");

        //printf("Exporting MKL output: ");
        //sparse_index_base_t out_indexing;
        //MKL_INT out_nrows;
        //MKL_INT out_ncols;
        //MKL_INT *out_pointerB = NULL;
        //MKL_INT *out_pointerE = NULL;
        //MKL_INT *out_rows = NULL;
        //double *out_values = NULL;
        //sparse_status_t export_status = mkl_sparse_d_export_csr (
            //*mkl_out,
            //&out_indexing,
            //&out_nrows,
            //&out_ncols,
            //&out_pointerB,
            //&out_pointerE,
            //&out_rows,
            //&out_values
        //);
        //switch(export_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");
 
        //auto SpAddHash_colPtr = SpAddHash_out.get_colPtr();
        //auto SpAddHash_rowIds = SpAddHash_out.get_rowIds();
        //auto SpAddHash_nzVals = SpAddHash_out.get_nzVals();

        //printf("SpAdd vs MKL Output Comparison\n");
        //printf("==================================\n");
        //printf("Number of columns: %ld vs %ld\n", SpAddHash_colPtr->size()-1, out_ncols);

        //bool correct = true;
        //if ((*SpAddHash_colPtr)[0] != out_pointerB[0]) correct = false;
        //for (int i = 0; i < out_ncols; i++){
            //if ((*SpAddHash_colPtr)[i+1] != out_pointerE[i]){
                //correct = false;
                //break;
            //}
        //}
        //if(correct == false) printf("Column pointers did not match\n");
        //else printf("Column pointers matched\n");

        //for (int i = 0; i < out_pointerE[out_ncols-1] && correct; i++){
            //if( (*SpAddHash_rowIds)[i] != out_rows[i] ){
                //std::cout << "row id[" << i << "]: " << (*SpAddHash_rowIds)[i] << " - " << out_rows[i] << std::endl; 
                //correct = false;
                //break;
            //}
        //}
        //if(correct == false) printf("Row ids did not match\n");
        //else printf("Row ids matched\n");

        //for (int i = 0; i < out_pointerE[out_ncols-1] && correct; i++){
            ////std::cout << (*SpAddHash_nzVals)[i] << " vs " << out_values[i] << std::endl;
            //if( abs((*SpAddHash_nzVals)[i] - out_values[i]) > 1e3 ){
                //std::cout << "nz vals[" << i << "]: " << (*SpAddHash_nzVals)[i] << " - " << out_values[i] << std::endl; 
                //correct = false;
                //break;
            //}
        //}
        //if(correct == false) printf("Nonzeros did not match\n");
        //else printf("Nonzeros matched\n");
        //printf("===========================\n");
        
        //mkl_sparse_destroy(*mkl_out);
        //for (int i = 0; i < k; i++){
           //if(mkl_sums[i] != NULL) mkl_sparse_destroy(mkl_sums[i]);
        //}
        //printf("\n");
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
