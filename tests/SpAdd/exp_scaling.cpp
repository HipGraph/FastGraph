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

	std::vector< CSC<uint32_t, uint32_t, uint32_t>* > vec;
    std::vector< CSC<uint32_t, uint32_t, uint32_t>* > vec_temp;

    // below is method to use random matrices from COO.h
    for(int i = 0; i < k; i++){
        //omp_set_num_threads(1);
        COO<uint32_t, uint32_t, uint32_t> coo;
        if(type == 0){
            coo.GenER(x,y,weighted, i*999);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
        }
        else{
            coo.GenRMAT(x,y,weighted, i*999);   //(x,y,true) Generate a weighted RMAT matrix with 2^x rows and columns and y nonzeros per column
        }
        //omp_set_num_threads(48);

        vec.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
        vec_temp.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
        //vec[i]->print_all();
    }

    //COO<uint32_t, uint32_t, uint32_t> coo;
    //if(type == 0) coo.GenER(x,y,weighted, 20210409);   //(x,y,true) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column
    //else coo.GenRMAT(x,y,weighted, 20210409);
    //for(int i = 0; i < k; i++){
        //coo.RandReLabel(false);
        //vec.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
        ////vec_temp.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
    //}

    for (int j = 0; j < k; j++){
        printf("Matrix %d:", j);
        const pvector<uint32_t> *col_ptr_j = vec[j]->get_colPtr();
        const pvector<uint32_t> *row_ids_j = vec[j]->get_rowIds();
        const pvector<uint32_t> *nz_j = vec[j]->get_nzVals();

        pvector<uint32_t> nnzCol;
        uint32_t maxNnz = 0;
        uint32_t maxNnzI = 0;
        for (uint32_t i = 0; i < col_ptr_j->size()-1; i++){
            nnzCol.push_back((*col_ptr_j)[i+1] - (*col_ptr_j)[i]);
            if (nnzCol[i] > maxNnz){
                maxNnz = nnzCol[i];
                maxNnzI = i;
            }
        }

        printf("maxNnz: %d maxNnzI: %d", maxNnz, maxNnzI);
        //getStats<uint32_t>(nnzCol, true);
        printf("\n");
    }

    //for (int ji = 0; ji < k; ji++){
        //for (int jj = 0; jj < k; jj++){
            //std::cout << (*(vec[ji]) == *(vec[jj])) << " " ;
        //}
        //std::cout << std::endl;
    //}
    
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
    //std::vector<int> threads{48, 1, 12};
    std::vector<int> threads{48};

    int iterations = 1;

    for(int i = 0; i < threads.size(); i++){
        omp_set_num_threads(threads[i]);
        mkl_set_num_threads(threads[i]);

        //clock.Start();
        //for(int j = 1; j < k; j++){
            //sparse_status_t add_status;
            //sparse_status_t destroy_status;
            //printf("Adding matrix %d: ", j);
            //if(j == 1){
                //add_status = mkl_sparse_d_add(
                    //SPARSE_OPERATION_NON_TRANSPOSE, 
                    //mkl_csr_matrices[j-1], 
                    //1.0, 
                    //mkl_csr_matrices[j], 
                    //&(mkl_sums[j])
                //);
                //destroy_status = mkl_sparse_destroy(mkl_sums[j-1]);
            //}
            //else{
                //add_status = mkl_sparse_d_add(
                    //SPARSE_OPERATION_NON_TRANSPOSE, 
                    //mkl_sums[j-1], 
                    //1.0, 
                    //mkl_csr_matrices[j], 
                    //&(mkl_sums[j])
                //);
                //destroy_status = mkl_sparse_destroy(mkl_sums[j-1]);
            //}
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

        //sparse_matrix_t* mkl_temp = (sparse_matrix_t*) malloc( 1 * sizeof(sparse_matrix_t) );
        //double mkl_time = 0;
        //matrix_descr desc;
        //desc.type = SPARSE_MATRIX_TYPE_GENERAL;
        //for(int u = 0; u < k; u++){
            //if(mkl_sums[u] != NULL){
                //mkl_sparse_destroy(mkl_sums[u]);
            //}
        //}
        //free(mkl_sums);
        //mkl_sums = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
        //for(int u = 0; u < k; u++){
            //mkl_sums[u] = NULL;
        //}
        //int nIntermediate = k;
        //while(nIntermediate > 1){
            ////printf("MKL pairwise tree intermediate %d\n", nIntermediate);
            //int j = 0;
            //int idxf = j * 2 + 0;
            //int idxs = idxf;
            //if(idxs + 1 < nIntermediate) idxs++;
            //while(idxs < nIntermediate){
                //if(idxf < idxs){
                    //clock.Start();
                    //mkl_sparse_d_add(
                        //SPARSE_OPERATION_NON_TRANSPOSE, 
                        //mkl_csr_matrices[idxf], 
                        //1.0, 
                        //mkl_csr_matrices[idxs], 
                        //mkl_temp
                    //);
                    //clock.Stop();
                    //mkl_time += clock.Seconds();
                    //mkl_sparse_destroy(mkl_csr_matrices[idxf]);
                    //mkl_sparse_destroy(mkl_csr_matrices[idxs]);
                    //mkl_sparse_copy(*mkl_temp, desc, &(mkl_csr_matrices[j]));
                    //mkl_sparse_destroy(*mkl_temp);
                //}
                //else{
                    //clock.Start();
                    //mkl_sparse_copy(mkl_csr_matrices[idxf], desc, mkl_temp);
                    //clock.Stop();
                    ////mkl_time += clock.Seconds();
                    //mkl_sparse_destroy(mkl_csr_matrices[idxf]);
                    //mkl_sparse_copy(*mkl_temp, desc, &(mkl_csr_matrices[j]));
                    //mkl_sparse_destroy(*mkl_temp);
                //}
                //j++;
                //idxf = j * 2 + 0;
                //idxs = idxf;
                //if(idxs + 1 < nIntermediate) idxs++;
            //}
            //nIntermediate = j;
        //}
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
        //std::cout << "MKL" << ","; 
        //std::cout << mkl_time << std::endl;
        

        //CSC<uint32_t, uint32_t, uint32_t> OutPairwiseLinear;
        //double lin_time = 0;
        //clock.Start();
        //OutPairwiseLinear = SpAdd<uint32_t,uint32_t, uint32_t,uint32_t> (vec[0], vec[1]);
        //clock.Stop();
        //lin_time += clock.Seconds();
        //std::cout << vec[0]->get_nnz() << " + " << vec[1]->get_nnz() << " = " << OutPairwiseLinear.get_nnz() << " Time: " << clock.Seconds() << std::endl;
        //for (int j = 2; j < k; j++){
            //std::cout << vec[j]->get_nnz() << " + " << OutPairwiseLinear.get_nnz() << " = ";
            //clock.Start();
            //OutPairwiseLinear = SpAdd<uint32_t,uint32_t,uint32_t,uint32_t>(&OutPairwiseLinear, vec[j]);
            //clock.Stop();
            //lin_time += clock.Seconds();
            //std::cout << OutPairwiseLinear.get_nnz() << " Time: " << clock.Seconds() << std::endl;
        //}
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
        //std::cout << "SpAddPairwiseLinear" << ","; 
        //std::cout << lin_time << std::endl;

        //std::vector< CSC<uint32_t, uint32_t, uint32_t>* > tree(vec.begin(), vec.end());
        std::vector< CSC<uint32_t, uint32_t, uint32_t>* > tree(vec_temp.begin(), vec_temp.end());
        CSC<uint32_t, uint32_t, uint32_t> * temp1;
        CSC<uint32_t, uint32_t, uint32_t> * temp2;
        int nIntermediate = tree.size();
        int level = 0;
        double tree_time = 0;
        while(nIntermediate > 1){
            int j = 0;
            int idxf = j * 2 + 0;
            int idxs = idxf;
            if(idxs + 1 < nIntermediate) idxs++;
            while(idxs < nIntermediate){
                if(idxf < idxs){
                    temp1 = tree[idxf];
                    temp2 = tree[idxs];
                    clock.Start();
                    pvector<uint32_t> nnzCPerCol = symbolicSpAddRegular<uint32_t,uint32_t,uint32_t,uint32_t>(tree[idxf], tree[idxs]);
                    tree[j] = new CSC<uint32_t, uint32_t, uint32_t>(SpAddRegular<uint32_t,uint32_t,uint32_t,uint32_t>(tree[idxf], tree[idxs], nnzCPerCol));
                    clock.Stop();
                    tree_time += clock.Seconds();
                    std::cout << "level " << level << ": ";
                    std::cout << temp1->get_nnz() << " + " << temp2->get_nnz() << " = " << tree[j]->get_nnz() << " Time: " << clock.Seconds() << std::endl;
                    delete temp1;
                    delete temp2;
                }
                else{
                    tree[j] = tree[idxf];
                    std::cout << "level " << level << ": ";
                    std::cout << tree[idxf]->get_nnz() << " + " << 0 << " = " << tree[j]->get_nnz() << std::endl;
                }
                j++;
                idxf = j * 2 + 0;
                idxs = idxf;
                if(idxs + 1 < nIntermediate) idxs++;
            }
            nIntermediate = j;
            level++;
        }
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
        std::cout << "SpAddPairwiseTree" << ","; 
        std::cout << tree_time << std::endl;


        CSC<uint32_t, uint32_t, uint32_t> SpAddHash_out;
        pvector<uint32_t> nnzCPerCol;
        
        //clock.Start(); 
        //nnzCPerCol = symbolicSpMultiAddHash<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(vec);
        //SpAddHash_out = SpMultiAddHash<uint32_t,uint32_t, uint32_t,uint32_t> (vec, nnzCPerCol, true);
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
        //std::cout << "SpMultiAddHash (sorted)" << ","; 
        //std::cout << clock.Seconds() << std::endl;

        //clock.Start(); 
        //nnzCPerCol = symbolicSpMultiAddHash<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(vec);
        //SpAddHash_out = SpMultiAddHash<uint32_t,uint32_t, uint32_t,uint32_t> (vec, nnzCPerCol, false);
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
        //std::cout << "SpMultiAddHash (unsorted)" << ","; 
        //std::cout << clock.Seconds() << std::endl;
        
        double sliding_time = 0;
        for(int it = 0; it < iterations; it++){
            clock.Start(); 
            //CSC<uint32_t, uint32_t, uint32_t> SpAddHybrid_out = SpMultiAddHashSliding<uint32_t,uint32_t, uint32_t,uint32_t> (vec, 512, 512, true);
            SpAddHash_out = SpMultiAddHashSliding<uint32_t,uint32_t, uint32_t,uint32_t> (vec, 512, 512, true);
            clock.Stop();
            sliding_time += clock.Seconds();
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
            std::cout << "SpMultiAddHashSliding (sorted)" << ","; 
            std::cout << clock.Seconds() << std::endl;
            //std::cout << SpAddHybrid_out.get_nnz() << std::endl;
            //SpAddHybrid_out.print_all();
        }

        if(*(tree[0]) == SpAddHash_out) printf("Equal\n");
        else printf("Not equal\n");

        //double spa_time = 0;
        //CSC<uint32_t, uint32_t, uint32_t> SpAddSpA_out;
        //for(int it = 0; it < iterations; it++){
            //clock.Start(); 
            //SpAddSpA_out = SpMultiAddSpA<uint32_t,uint32_t, uint32_t,uint32_t> (vec);
            //clock.Stop();
            //spa_time += clock.Seconds();
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
            //std::cout << "SpMultiAddSpA" << ","; 
            //std::cout << clock.Seconds() << std::endl;
            ////std::cout << SpAddHybrid_out.get_nnz() << std::endl;
            ////SpAddHybrid_out.print_all();
        //}
        
        //clock.Start(); 
        //CSC<uint32_t, uint32_t, uint32_t> SpAddHeap_out = SpMultiAddHeap<uint32_t,uint32_t, uint32_t,uint32_t, uint32_t> (vec);
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
        //std::cout << "SpMultiAddHeap" << ","; 
        //std::cout << clock.Seconds() << std::endl;
        ////std::cout << SpAddHybrid_out.get_nnz() << std::endl;
        ////SpAddHybrid_out.print_all();


        //double t0, t1, t2, t3;
        //t0 = omp_get_wtime();
        //pvector<uint32_t> nnzCPerCol = symbolicSpMultiAddHash<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(vec);
        //t1 = omp_get_wtime();
        //printf("Time for symbolic with pure hash: %lf\n", t1-t0);
        
        //t0 = omp_get_wtime();
        //pvector<uint32_t> nnzCPerCol2 = symbolicSpMultiAddHashSliding<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(vec, 1024, 1024);
        //t1 = omp_get_wtime();
        //printf("Time for symbolic with sliding hash: %lf\n", t1-t0);
        
        //bool flag = true;
        //for(uint32_t i=0; i< nnzCPerCol.size(); i++)
        //{
            //if(nnzCPerCol[i] != nnzCPerCol2[i]){
                //flag = false;
               ////std::cout << "not equal" << std::endl;
            //}
        //}
        //if(flag == true) printf("Symbolic equal!\n");
        //if(flag == false) printf("Symbolic not equal!\n");

        
        
        //printf("Transposing MKL output: ");
        //sparse_matrix_t *mkl_out = (sparse_matrix_t *) malloc( sizeof(sparse_matrix_t) );
        //sparse_status_t conv_status = mkl_sparse_convert_csr(
            ////mkl_sums[k-1],
            //mkl_csr_matrices[0],
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
       //if(mkl_csc_matrices[i] != NULL) mkl_sparse_destroy(mkl_csc_matrices[i]);
       //if(mkl_csr_matrices[i] != NULL) mkl_sparse_destroy(mkl_csr_matrices[i]);
       //if(mkl_sums[i] != NULL) mkl_sparse_destroy(mkl_sums[i]);
    }
    if(mkl_values != NULL) free(mkl_values);
    if(mkl_rows != NULL) free(mkl_rows);
    if(mkl_pointerB != NULL) free(mkl_pointerB);
    if(mkl_pointerE != NULL) free(mkl_pointerE);
    if(mkl_csc_matrices != NULL) free(mkl_csc_matrices);
    if(mkl_csr_matrices != NULL) free(mkl_csr_matrices);
    if(mkl_sums != NULL) free(mkl_sums);

    //std::vector<int> threads{48};
    //std::vector<uint32_t> windowSizes{16, 32, 64, 128, 256, 512, 1024, 1024 * 2, 1024 * 4, 1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128, 1024 * 256, 1024 * 512, 1024 * 1024};
    ////std::vector<uint32_t> windowSizes{1024 * 16};

    //for(int t = 0; t < threads.size(); t++){
        //omp_set_num_threads(threads[t]);
        //mkl_set_num_threads(threads[t]);
        //for (int w = 0; w < windowSizes.size(); w++){
            ////if(type == 0){
                ////std::cout << "ER" << "," ;
            ////}
            ////else{
                ////std::cout << "RMAT" << "," ;
            ////}
            ////std::cout << x << "," ;
            ////std::cout << y << "," ;
            ////std::cout << k << "," ;
            ////std::cout << threads[t] << ",";
            ////std::cout << "SpMultiAddHash" << ","; 
            ////std::cout << windowSizes[w] << ",";
            ////clock.Start(); 
            ////pvector<uint32_t> nnzCPerCol = symbolicSpMultiAddHash<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(vec);
            ////SpMultiAddHash<uint32_t,uint32_t, uint32_t,uint32_t> (vec, nnzCPerCol, windowSizes[w], true);
            ////clock.Stop();
            ////std::cout << clock.Seconds() << std::endl;
            //////SpAddHash_out.print_all();
        
            //if(type == 0){
                //std::cout << "ER" << "," ;
            //}
            //else{
                //std::cout << "RMAT" << "," ;
            //}
            //std::cout << x << "," ;
            //std::cout << y << "," ;
            //std::cout << k << "," ;
            //std::cout << threads[t] << ",";
            //std::cout << "SpMultiAddHashSliding" << ","; 
            //std::cout << windowSizes[w] << ",";
            //clock.Start(); 
            //SpMultiAddHashSliding<uint32_t,uint32_t, uint32_t,uint32_t> (vec, windowSizes[w], windowSizes[w], false);
            //clock.Stop();
            //std::cout << clock.Seconds() << std::endl;
            ////std::cout << SpAddHybrid_out.get_nnz() << std::endl;
            ////SpAddHybrid_out.print_all();
            
        //}
    //}

	return 0;

}
