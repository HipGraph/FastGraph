#ifndef CSC_ADDER_H
#define CSC_ADDER_H

#include "CSC.h" // need to check the relative paths for this section
#include "GAP/pvector.h"
#include "GAP/timer.h"
#include "utils.h"

#include <vector> // needed while taking input
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iterator>
#include <assert.h>
#include <omp.h>
#include <queue>
#include <tuple>
#include <fstream>

/*
INDEX:
usage: CSC<RIT, VT, CPT> add_vec_of_matrices_{id}<RIT, CIT, VT, CPT, NM>(vector<CSC<RIT, VT, CPT> * > & )
where :
NM is type for number_of_matrices to merge

{id} = 
1 => unordered_map with push_back
2 => unordered_map with push_back and locks
3 => unordered_map with symbolic step
4 => heaps with symbolic step
5 => radix sort with symbolic step
6 => row_size_pvector_maintaining_sum with symbolic step

Note: ParallelPrefixSum from utils.h is used in here

*/

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_1(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices) // used array of pointers to pvectors(maybe more space needed) and push_back is used
{	


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}


	pvector<std::pair<VT, RIT> >* array_of_column_pointers = new pvector<std::pair<VT, RIT> >[num_of_columns]; 

	size_t count_for_nnz = 0;

	pvector<RIT> nz_per_column(num_of_columns,0);

#pragma omp parallel for reduction(+ : count_for_nnz)
	for(CIT i = 0; i < num_of_columns; i++){

		std::unordered_map<RIT, VT> umap;
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
			const pvector<VT> *nz_i = vec_of_matrices[k]->get_nzVals();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				umap[(*row_ids_i)[j] ] += (*nz_i)[j];
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;
			nz_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;
			delete nz_i;

		}

		for(auto iter = umap.begin(); iter != umap.end(); iter++){
			
				array_of_column_pointers[i].push_back(std::make_pair(iter->second, iter->first));
				count_for_nnz++;
			
		}
		nz_per_column[i] = array_of_column_pointers[i].size();
	} 
	// parallel programming ended

	pvector<VT> value_vector_for_csc(count_for_nnz);
	pvector<RIT> row_vector_for_csc(count_for_nnz);
	pvector<CPT> prefix_sum(num_of_columns+1);

	// prefix_sum[0] = 0;
	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + array_of_column_pointers[i-1].size();
	// }

	ParallelPrefixSum(nz_per_column, prefix_sum);

#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){
		size_t total_elements = array_of_column_pointers[i].size();
		for(size_t j = 0; j < total_elements; j++){
			row_vector_for_csc[(j + prefix_sum[i])] = ( (array_of_column_pointers[i]))[j].second;
			value_vector_for_csc[(j + prefix_sum[i])] = ( (array_of_column_pointers[i]))[j].first;
		}
	}// parallel programming ended

	delete[] array_of_column_pointers;

	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, count_for_nnz, false, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&prefix_sum);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	result_matrix_csc.sort_inside_column();

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}



//..........................................................................//





template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_2(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          // less space(stroed only what is required), maybe more time(since I used locks in omp prallel for and push_back)
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<VT> value_vector(0);
	pvector<RIT> row_vector(0);
	pvector<CIT> column_vector(0);

	pvector<RIT> nz_per_column(num_of_columns, 0); 
	size_t count_for_nnz = 0;

	omp_lock_t writelock;
	omp_init_lock(&writelock);

#pragma omp parallel for reduction(+ : count_for_nnz)
	for(CIT i = 0; i < num_of_columns; i++){

		std::unordered_map<RIT, VT> umap;
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
			const pvector<VT> *nz_i = vec_of_matrices[k]->get_nzVals();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				umap[(*row_ids_i)[j] ] += (*nz_i)[j];
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;
			nz_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;
			delete nz_i;
		}

		 omp_set_lock(&writelock);
		//#pragma openmp critical
		for(auto iter = umap.begin(); iter != umap.end(); iter++){
				value_vector.push_back(iter->second);
				row_vector.push_back(iter->first);
				column_vector.push_back(i);
				count_for_nnz++;
				nz_per_column[i] = nz_per_column[i]+1;
			
		}
		 omp_unset_lock(&writelock);

	} 
	omp_destroy_lock(&writelock);
	// parallel programming ended

	pvector<CPT> prefix_sum(num_of_columns+1);
	//pvector<CPT> column_vector_for_csc(num_of_columns+1);
	// prefix_sum[0] = 0;
	// column_vector_for_csc[0] = 0;

	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + nz_per_column[i-1];
	// 	column_vector_for_csc[i] = prefix_sum[i];
	// }
	ParallelPrefixSum(nz_per_column, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());



	pvector<VT> value_vector_for_csc(count_for_nnz);
	pvector<RIT> row_vector_for_csc(count_for_nnz);

	omp_init_lock(&writelock);

#pragma omp parallel for
	for(size_t i = 0; i < count_for_nnz; i++){
		CPT position;
		omp_set_lock(&writelock);						// time consuming step
		position = prefix_sum[column_vector[i]];
		prefix_sum[column_vector[i] ]++;
		omp_unset_lock(&writelock);

		value_vector_for_csc[position] = value_vector[i];
		row_vector_for_csc[position] = row_vector[i];
		
	}
	omp_destroy_lock(&writelock);


	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, count_for_nnz, false, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	result_matrix_csc.sort_inside_column();

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}



//..........................................................................//

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>

pvector<RIT> symbolic_add_vec_of_matrices(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();

	pvector<RIT> nz_per_column(num_of_columns);

#pragma omp parallel for 
	for(CIT i = 0; i < num_of_columns; i++){

		std::unordered_map<RIT, VT> umap;
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				umap[(*row_ids_i)[j] ] ++;
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;

		}
		nz_per_column[i] = umap.size();
	} 
	// parallel programming ended
	return std::move(nz_per_column);

}




template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHash(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{
    double t0, t1, t2, t3;
    CIT num_of_columns = vec_of_matrices[0]->get_ncols();
    NM number_of_matrices = vec_of_matrices.size();
    
    pvector<RIT> nz_per_column(num_of_columns);
    pvector<RIT> flops_per_column(num_of_columns, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < num_of_columns; i++){
        for(int k = 0; k < number_of_matrices; k++){
            const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<int64_t> prefix_sum(num_of_columns+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[num_of_columns];
    int64_t flops_per_thread_expected;
    
    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    int nthreads;
    pvector<CIT> splitters;
    pvector<double> ttimes;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            splitters.resize(nthreads);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), tid * flops_per_thread_expected) - prefix_sum.begin();
#pragma omp barrier
#pragma omp for schedule(static) nowait
        for (int t = 0; t < nthreads; t++) {
            CIT colStart = splitters[t];
            CIT colEnd = (t < nthreads-1) ? splitters[t+1] : num_of_columns;
            for(CIT i = colStart; i < colEnd; i++)
            {
                
                nz_per_column[i] = 0;
                size_t nnzcol = 0;
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    nnzcol += (vec_of_matrices[k]->get_colPtr(i+1) - vec_of_matrices[k]->get_colPtr(i));
                }
                    
                size_t htSize = minHashTableSize;
                while(htSize < nnzcol) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j] = -1;
                }
                
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
                    const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
                    for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++)
                    {
                        
                        RIT key = (*row_ids_i)[j];
                        RIT hash = (key*hashScale) & (htSize-1);
                        while (1) //hash probing
                        {
                            
                            if (globalHashVec[hash] == key) //key is found in hash table
                            {
                                break;
                            }
                            else if (globalHashVec[hash] == -1) //key is not registered yet
                            {
                                globalHashVec[hash] = key;
                                nz_per_column[i] ++;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
    printf("[Symbolic Hash] Time for parallel section: %lf\n", t1-t0);
    printf("[Symbolic Hash] Stats of parallel section timing:\n");
    getStats<double>(ttimes, true);
    // parallel programming ended
    //for (int i = 0 ; i < nz_per_column.size(); i++){
        //fp << nz_per_column[i] << std::endl;
    //}
    //fp.close();
    return std::move(nz_per_column);
    
}


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashSliding1(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{
    
    CIT num_of_columns = vec_of_matrices[0]->get_ncols();
    
    pvector<RIT> nz_per_column(num_of_columns);
    RIT nrows = vec_of_matrices[0]->get_nrows();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = 1024*2;
    const RIT hashScale = 107;
    NM number_of_matrices = vec_of_matrices.size();
#pragma omp parallel
    {
        std::vector<RIT> globalHashVec(minHashTableSize);
#pragma omp for
    for(CIT i = 0; i < num_of_columns; i++)
    {
        
        nz_per_column[i] = 0;
        size_t nnzcol = 0;
        for(NM k = 0; k < number_of_matrices; k++)
        {
            nnzcol += (vec_of_matrices[k]->get_colPtr(i+1) - vec_of_matrices[k]->get_colPtr(i));
        }
        if(nnzcol > maxHashTableSize)
        {
            size_t nparts = nnzcol/maxHashTableSize + 1;
            RIT nRowsPerPart = nrows / nparts;
             
            for(size_t p=0; p<nparts; p++)
            {
                RIT rowStart = p * nRowsPerPart;
                RIT rowEnd = (p == nparts-1) ? nrows : (p+1) * nRowsPerPart;
                size_t nnzcolpart = 0;
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = vec_of_matrices[k]->get_rowIds();
                   
                    auto first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                    auto last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                    nnzcolpart += last-first;
                }
                size_t htSize = minHashTableSize;
                while(htSize < nnzcolpart) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j] = -1;
                }
                
                
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = vec_of_matrices[k]->get_rowIds();
                    auto first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                    auto last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                    
                    for(; first<last; first++)
                    {
                        RIT key = *first;
                        RIT hash = (key*hashScale) & (htSize-1);
                        while (1) //hash probing
                        {
                            
                            if (globalHashVec[hash] == key) //key is found in hash table
                            {
                                break;
                            }
                            else if (globalHashVec[hash] == -1) //key is not registered yet
                            {
                                globalHashVec[hash] = key;
                                nz_per_column[i] ++;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            size_t htSize = minHashTableSize;
            while(htSize < nnzcol) //htSize is set as 2^n
            {
                htSize <<= 1;
            }
            if(globalHashVec.size() < htSize)
                globalHashVec.resize(htSize);
            for(size_t j=0; j < htSize; ++j)
            {
                globalHashVec[j] = -1;
            }
            
            for(NM k = 0; k < number_of_matrices; k++)
            {
                const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
                const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
                for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++)
                {
                    
                    RIT key = (*row_ids_i)[j];
                    RIT hash = (key*hashScale) & (htSize-1);
                    while (1) //hash probing
                    {
                        
                        if (globalHashVec[hash] == key) //key is found in hash table
                        {
                            break;
                        }
                        else if (globalHashVec[hash] == -1) //key is not registered yet
                        {
                            globalHashVec[hash] = key;
                            nz_per_column[i] ++;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (htSize-1);
                        }
                    }
                    //curptr[i]++;
                    //umap[(*row_ids_i)[j] ] ++;
                }
            }
        }
    }
    }
    return std::move(nz_per_column);
    
}


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashSliding(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{
    
    CIT num_of_columns = vec_of_matrices[0]->get_ncols();
    
    pvector<RIT> nz_per_column(num_of_columns);
    RIT nrows = vec_of_matrices[0]->get_nrows();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = 1024*8;
    const RIT hashScale = 107;
    NM number_of_matrices = vec_of_matrices.size();
    
#pragma omp parallel
    {
        std::vector<RIT> globalHashVec(maxHashTableSize);
#pragma omp for
    for(CIT i = 0; i < num_of_columns; i++)
    {
        
        nz_per_column[i] = 0;
        size_t nnzcol = 0;
        for(NM k = 0; k < number_of_matrices; k++)
        {
            nnzcol += (vec_of_matrices[k]->get_colPtr(i+1) - vec_of_matrices[k]->get_colPtr(i));
        }
        

        if(nnzcol > maxHashTableSize)
        {
            size_t nparts = nnzcol/maxHashTableSize + 1;
            RIT nRowsPerPart = nrows / nparts;
            
            pvector<RIT*> rowStart(number_of_matrices);
            pvector<RIT*> rowEnd(number_of_matrices);
            for(NM k = 0; k < number_of_matrices; k++)
            {
                const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = vec_of_matrices[k]->get_rowIds();
                rowStart[k] = rowIds->begin() + (*colPtr)[i];
            }
            
            for(size_t p=0; p<nparts; p++)
            {
                RIT end = (p == nparts-1) ? nrows : (p+1) * nRowsPerPart;
                size_t nnzcolpart = 0;
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = vec_of_matrices[k]->get_rowIds();
                    rowEnd[k] = std::lower_bound( rowStart[k], rowIds->begin() + (*colPtr)[i+1], end );
                    nnzcolpart += rowEnd[k]-rowStart[k];
                }
                size_t htSize = minHashTableSize;
                while(htSize < nnzcolpart) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j] = -1;
                }
                
                
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    for(; rowStart[k]<rowEnd[k]; rowStart[k]++)
                    {
                        RIT key = *rowStart[k];
                        RIT hash = (key*hashScale) & (htSize-1);
                        while (1) //hash probing
                        {
                            
                            if (globalHashVec[hash] == key) //key is found in hash table
                            {
                                break;
                            }
                            else if (globalHashVec[hash] == -1) //key is not registered yet
                            {
                                globalHashVec[hash] = key;
                                nz_per_column[i] ++;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            size_t htSize = minHashTableSize;
            while(htSize < nnzcol) //htSize is set as 2^n
            {
                htSize <<= 1;
            }
            if(globalHashVec.size() < htSize)
                globalHashVec.resize(htSize);
            for(size_t j=0; j < htSize; ++j)
            {
                globalHashVec[j] = -1;
            }
            
            for(NM k = 0; k < number_of_matrices; k++)
            {
                const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
                const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
                for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++)
                {
                    
                    RIT key = (*row_ids_i)[j];
                    RIT hash = (key*hashScale) & (htSize-1);
                    while (1) //hash probing
                    {
                        
                        if (globalHashVec[hash] == key) //key is found in hash table
                        {
                            break;
                        }
                        else if (globalHashVec[hash] == -1) //key is not registered yet
                        {
                            globalHashVec[hash] = key;
                            nz_per_column[i] ++;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (htSize-1);
                        }
                    }
                }
            }
        }
    }
    }
    return std::move(nz_per_column);
    
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashSliding2(std::vector<CSC<RIT, VT, CPT>* > &matrices)
{
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = 512 * 2;
    const RIT maxHashTableSizeSymbolic = 512 * 2;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    int nthreads;
    int nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<int64_t> flopsPerCol(ncols);
    pvector<int64_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel for
    for(CPT i = 0; i < ncols; i++){
        flopsPerCol[i] = 0;
        for(int k = 0; k < nmatrices; k++){
            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
            flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
        nWindowPerColSymbolic[i] = flopsPerCol[i] / maxHashTableSizeSymbolic + 1;
    }
    t3 = omp_get_wtime();
    printf("[symbolic sliding hash]\tTime for number of window calculation: %lf\n", t3-t2);
    

    pvector<int64_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<int64_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);
    printf("[symbolic sliding hash] stats of number of window per column\n");
    getStats<int64_t>(nWindowPerColSymbolic, true);
    printf("***\n\n");

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    int64_t flopsTot = prefixSumSymbolic[ncols];
    int64_t flopsPerThreadExpected;
    
    t1 = omp_get_wtime();
    printf("[symbolic sliding hash]\tTime before symbolic: %lf\n", t1-t0);

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            ttimes.resize(nthreads);
            splitters.resize(nthreads);
            flopsPerThreadExpected = flopsTot / nthreads;
            //printf("Expected flops per thread: %lld\n", flopsPerThreadExpected);
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), tid * flopsPerThreadExpected) - prefixSumSymbolic.begin();
        //printf("splitters[%d] %d: target %lld\n", tid, splitters[tid], tid * flopsPerThreadExpected);
#pragma omp barrier
#pragma omp for schedule(static) nowait
        for (int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            //printf("Thread %d processing %d columns: [%d, %d)\n", t, colEnd-colStart, colStart, colEnd);
            for(CPT i = colStart; i < colEnd; i++){
                int64_t nwindows = nWindowPerColSymbolic[i];
                RIT nrowsPerWindow = nrows / nwindows;
                nnzPerCol[i] = 0;
                nWindowPerCol[i] = 1;
                RIT runningSum = 0;
                for(size_t w=0; w < nwindows; w++){
                    // Determine the start and end row index
                    RIT rowStart = w * nrowsPerWindow;
                    RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;
                    
                    int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                    nnzPerWindowSymbolic[wIdx].first = rowStart;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    // Determine total number of input nonzeros in the window
                    size_t flopsWindow = 0;
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        
                        auto first = rowIds->begin();
                        auto last = rowIds->end();
                        int64_t startIdx, endIdx, midIdx;

                        //t2 = omp_get_wtime();
                        if(rowStart > 0){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            first = rowIds->begin() + endIdx;
                        }
                        else{
                            first = rowIds->begin() + (*colPtr)[i];
                        }

                        if(rowEnd < nrows){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            last = rowIds->begin() + endIdx;
                        }
                        else{
                            last = rowIds->begin() + (*colPtr)[i+1];
                        }
                        //t3 = omp_get_wtime();
                        //if(tid == 0) printf("custom lower bound: %lf\n", t3-t2);
                        
                        ////t2 = omp_get_wtime();
                        //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        ////t3 = omp_get_wtime();
                        ////if(tid == 0) printf("default lower bound: %lf\n", t3-t2);
                        
                        //first = (rowStart == 0) ? rowIds->begin() + (*colPtr)[i] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = (rowEnd == nrows) ? rowIds->begin() + (*colPtr)[i+1] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );

                        flopsWindow += last-first;
                    }
                    
                    // In worst case hash table may need to store all input nonzeros in the window
                    // So resize the hash table accordingly
                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }
                    
                    // For each matrix
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin();
                        auto last = rowIds->end();
                        int64_t startIdx, endIdx, midIdx;

                        //t2 = omp_get_wtime();
                        if(rowStart > 0){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            first = rowIds->begin() + endIdx;
                        }
                        else{
                            first = rowIds->begin() + (*colPtr)[i];
                        }

                        if(rowEnd < nrows){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            last = rowIds->begin() + endIdx;
                        }
                        else{
                            last = rowIds->begin() + (*colPtr)[i+1];
                        }

                        ////Determine the range of elements that fall within the window
                        //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        
                        //first = (rowStart == 0) ? rowIds->begin() + (*colPtr)[i] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = (rowEnd == nrows) ? rowIds->begin() + (*colPtr)[i+1] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        
                        for(; first<last; first++){
                            RIT key = *first;
                            RIT hash = (key*hashScale) & (htSize-1);
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash] == key) //key is found in hash table
                                {
                                    break;
                                }
                                else if (globalHashVec[hash] == -1) //key is not registered yet
                                {
                                    globalHashVec[hash] = key;
                                    nnzPerCol[i]++;
                                    nnzPerWindowSymbolic[wIdx].second++;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }
                        }
                    }

                    if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                        runningSum = nnzPerWindowSymbolic[wIdx].second;
                        nWindowPerCol[i]++;
                    }
                    else{
                        runningSum += runningSum + nnzPerWindowSymbolic[wIdx].second;
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
    printf("[symbolic sliding hash]\tTime for parallel section: %lf\n", t1-t0);
    printf("[symbolic sliding hash]\tstats for parallel section time:\n");
    getStats<double>(ttimes, true);
    printf("***\n\n");

    printf("[sliding hash] stats of number of window per column\n");
    getStats<RIT>(nWindowPerCol, true);
    printf("***\n\n");

    return std::move(nnzPerCol);
}


//..........................................................................//
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHash(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();
    

    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<RIT> nnzPerThread; // To record number of nnz processed by each thread 
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    auto nnzPerColStats = getStats<RIT>(nnzPerCol);
    int nthreads;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            ttimes.resize(nthreads);
            nnzPerThread.resize(nthreads);
            splitters.resize(nthreads);
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), tid * nnzCPerThreadExpected) - prefix_sum.begin();
#pragma omp barrier
        nnzPerThread[tid] = 0;
        std::vector< std::pair<RIT,VT>> globalHashVec(minHashTableSize);
#pragma omp for schedule(static) nowait 
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = t < nthreads-1 ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] != 0){
                    //----------- preparing the hash table for this column -------
                    size_t htSize = minHashTableSize;
                    while(htSize < nnzPerCol[i])
                    {
                        htSize <<= 1;
                    }   
                    if(globalHashVec.size() < htSize)
                        globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j)
                    {
                        globalHashVec[j].first = -1;
                    }
                
                    //----------- add this column form all matrices -------
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                        {
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }   
                        }
                        //nnzPerThread[tid] += (*colPtr)[i+1] - (*colPtr)[i]; // Would cause false sharing
                    }
               
                    if(sorted)
                    {
                        size_t index = 0;
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                globalHashVec[index++] = globalHashVec[j];
                            }
                        }
                    
                        // try radix sort
                        //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                        std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                    
                    
                        for (size_t j=0; j < index; ++j)
                        {
                            CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                            CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                            prefix_sum[i] ++;
                        }
                    }
                    else
                    {
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                                CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                                prefix_sum[i] ++;
                            }
                        }
                    }
                }
            }
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }

    t1 = omp_get_wtime();
    printf("[Hash]\tTime for parallel section: %lf\n", t1-t0);
    printf("[Hash]\tStats for parallel section timing:\n");
    getStats<double>(ttimes, true);

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

/*
 * Hash + SpA Hybrid by simply processing each dense column in a single thread using SPA
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHybrid(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();
    
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    pvector<RIT> nnzPerColSparse(nnzPerCol.begin(), nnzPerCol.end());
    pvector<RIT> nnzPerColDense(nnzPerCol.begin(), nnzPerCol.end());
    pvector<CPT> skippedColList;
    t0 = omp_get_wtime();
#pragma omp parallel for
    for (CPT i = 0; i < ncols; i++){
        if(nnzPerCol[i] > 2048){
            nnzPerColSparse[i] = 0;
        }
        else{
            nnzPerColDense[i] = 0;
        }
    }
    t1 = omp_get_wtime();
    //printf("Time for skip calculation %lf\n", t1-t0);
    
    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefixSum);
    pvector<CPT> prefixSumSparse(ncols+1);
    ParallelPrefixSum(nnzPerColSparse, prefixSumSparse);
    pvector<CPT> prefixSumDense(ncols+1);
    ParallelPrefixSum(nnzPerColDense, prefixSumDense);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCTotSparse = prefixSumSparse[ncols];
    CPT nnzCTotDense = prefixSumDense[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerThreadSparseExpected;
    CPT nnzCPerThreadDenseExpected;
    int nthreads;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            nnzCPerThreadSparseExpected = nnzCTotSparse / nthreads;
            nnzCPerThreadDenseExpected = nnzCTotDense / nthreads;
            ttimes.resize(nthreads);
            splitters.resize(nthreads);
        }
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
        std::vector< std::pair<RIT,VT> > spa(nrows);
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSumSparse.begin(), prefixSumSparse.end(), tid * nnzCPerThreadSparseExpected) - prefixSumSparse.begin();
#pragma omp barrier
#pragma omp for schedule(static) nowait 
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] != 0 && nnzPerCol[i] <= 2048){
                    //----------- preparing the hash table for this column -------
                    size_t htSize = minHashTableSize;
                    while(htSize < nnzPerCol[i]){
                        htSize <<= 1;
                    }   
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j].first = -1;
                    }
                
                    //----------- add this column form all matrices -------
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }   
                        }
                    }
               
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j){
                        if (globalHashVec[j].first != -1){
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
                
                    std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                
                    for (size_t j=0; j < index; ++j){
                        CrowIds[prefixSum[i]] = globalHashVec[j].first;
                        CnzVals[prefixSum[i]] = globalHashVec[j].second;
                        prefixSum[i] ++;
                    }
                }
            }
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSumDense.begin(), prefixSumDense.end(), tid * nnzCPerThreadDenseExpected) - prefixSumDense.begin();
#pragma omp barrier
#pragma omp for schedule(static) nowait 
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] > 2048) {
                    for (RIT j = 0; j < nrows; j++){
                        spa[j].first = -1;
                    }
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                        {
                            RIT key = (*rowIds)[j];
                            VT curval = (*nzVals)[j];
                            if (spa[key].first == -1){
                                spa[key].first = key;
                                spa[key].second = curval;
                            }
                            else{
                                spa[key].second += curval;
                            }
                        }
                    }
                    RIT index = 0;
                    for (size_t j=0; j < nrows; ++j){
                        if (spa[j].first != -1){
                            spa[index++] = spa[j];
                        }
                    }
                
                    for (size_t j=0; j < index; ++j)
                    {
                        CrowIds[prefixSum[i]] = spa[j].first;
                        CnzVals[prefixSum[i]] = spa[j].second;
                        prefixSum[i] ++;
                    }
                }
            }
        }   
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    return std::move(sumMat);
}

/*
 * Hash + SpA Hybrid with row wise distribution of work for dense columns
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHybrid2(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();
    
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    pvector<RIT> nnzPerColSparse(nnzPerCol.begin(), nnzPerCol.end());
    pvector<RIT> nnzPerColDense(nnzPerCol.begin(), nnzPerCol.end());
    pvector<CPT> skippedColList;
    t0 = omp_get_wtime();
#pragma omp parallel for
    for (CPT i = 0; i < ncols; i++){
        if(nnzPerCol[i] > 2048){
            nnzPerColSparse[i] = 0;
        }
        else{
            nnzPerColDense[i] = 0;
        }
    }
    t1 = omp_get_wtime();
    //printf("Time for skip calculation %lf\n", t1-t0);
    
    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefixSum);
    pvector<CPT> prefixSumSparse(ncols+1);
    ParallelPrefixSum(nnzPerColSparse, prefixSumSparse);
    pvector<CPT> prefixSumDense(ncols+1);
    ParallelPrefixSum(nnzPerColDense, prefixSumDense);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCTotSparse = prefixSumSparse[ncols];
    CPT nnzCTotDense = prefixSumDense[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerThreadSparseExpected;
    CPT nnzCPerThreadDenseExpected;
    int nthreads;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    std::vector< std::pair<RIT,VT> > spa(nrows);
    std::vector< std::vector <std::pair<RIT,VT> > > denseOutputPerThread;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            nnzCPerThreadSparseExpected = nnzCTotSparse / nthreads;
            nnzCPerThreadDenseExpected = nnzCTotDense / nthreads;
            ttimes.resize(nthreads);
            splitters.resize(nthreads);
            denseOutputPerThread.resize(nthreads);
        }
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSumSparse.begin(), prefixSumSparse.end(), tid * nnzCPerThreadSparseExpected) - prefixSumSparse.begin();
#pragma omp barrier
#pragma omp for schedule(static) nowait 
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] != 0 && nnzPerCol[i] <= 2048){
                    //----------- preparing the hash table for this column -------
                    size_t htSize = minHashTableSize;
                    while(htSize < nnzPerCol[i]){
                        htSize <<= 1;
                    }   
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j].first = -1;
                    }
                
                    //----------- add this column form all matrices -------
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }   
                        }
                    }
               
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j){
                        if (globalHashVec[j].first != -1){
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
                
                    std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                
                    for (size_t j=0; j < index; ++j){
                        CrowIds[prefixSum[i]] = globalHashVec[j].first;
                        CnzVals[prefixSum[i]] = globalHashVec[j].second;
                        prefixSum[i] ++;
                    }
                }
            }
        }

        // Rows of each previously skipped columns will be split between threads
        for (CPT i = 0; i < ncols; i++){
            if(nnzPerCol[i] > 2048){
                // Determine matrix having densest pattern for this particular column
                int kDense = 0; // Matrix
                RIT nnzDense = (*(matrices[0]->get_colPtr()))[i+1] - (*(matrices[0]->get_colPtr()))[i]; // Density
                for (int k = 0; k < nmatrices; k++){
                    RIT nnz = (*(matrices[k]->get_colPtr()))[i+1] - (*(matrices[k]->get_colPtr()))[i];
                    if(nnz > nnzDense){
                        nnzDense = nnz;
                        kDense = k;
                    }
                }

                // Use the figured out dense pattern to balance load between threads
                // Divide rows between threads
                const pvector<CPT> *colPtrDense = matrices[kDense]->get_colPtr();
                const pvector<RIT> *rowIdsDense = matrices[kDense]->get_rowIds();
                const pvector<VT> *nzValsDense = matrices[kDense]->get_nzVals();
                RIT nnzPerThreadDense = nnzDense / nthreads; 

                // For the matrix having densest pattern at current column, determine the range of row indices for current thread
                CPT colStartDense = (*colPtrDense)[i]; // Get index in the row pointer array of the matrix from where current column starts
                CPT colEndDense = (*colPtrDense)[i+1]; // Get index in the row pointer array of the matrix from where next column starts if there is any
                RIT rowIdsDenseThreadStartIdx = (colStartDense + tid * nnzPerThreadDense);
                RIT rowIdsDenseThreadEndIdx = (tid < nthreads-1) ? (colStartDense + (tid+1) * nnzPerThreadDense) : colEndDense;
                RIT rowStartDense = 0;
                RIT rowEndDense = nrows;
                if (tid > 0){
                    if(rowIdsDenseThreadStartIdx < (*colPtrDense)[ncols]) {
                        rowStartDense = (*rowIdsDense)[rowIdsDenseThreadStartIdx];
                    }
                }
                if (tid < nthreads-1){
                    if(rowIdsDenseThreadEndIdx < (*colPtrDense)[ncols]) {
                        rowEndDense = (*rowIdsDense)[rowIdsDenseThreadEndIdx];
                    }
                }

                // Each thread initializes range of the SpA array that it owns
                for (RIT j = rowStartDense; j < rowEndDense; j++) spa[j].first = -1;

                // Figure out the range of rows for the current column of each matrix that current thread would process
                pvector< std::pair<RIT,RIT> > rowRanges(nmatrices); 
                RIT nnzProcessedInRange = 0;
                for (int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    rowRanges[k].first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStartDense ) - rowIds->begin();
                    rowRanges[k].second = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEndDense ) - rowIds->begin();
                    for (RIT j = rowRanges[k].first; j < rowRanges[k].second; j++){
                        RIT rowId = (*rowIds)[j];
                        VT value = (*nzVals)[j];
                        if (spa[rowId].first == -1){
                            spa[rowId].first = rowId;
                            spa[rowId].second = value;
                            nnzProcessedInRange++;
                        }
                        else{
                            spa[rowId].second += value;
                        }
                    }
                }
                RIT index = 0;
                denseOutputPerThread[tid].resize(nnzProcessedInRange);
                for (RIT j=rowStartDense; j < rowEndDense; j++){
                    if (spa[j].first != -1){
                        denseOutputPerThread[tid][index++] = spa[j];
                    }
                }
#pragma omp barrier
                // Wait untill all threads are done for this column
                // Copy contents to output using only one thread
                if(tid == 0){
                    for(int t = 0; t < nthreads; t++){
                        for(RIT index = 0; index < denseOutputPerThread[t].size(); index++){
                            CrowIds[prefixSum[i]] = denseOutputPerThread[t][index].first;
                            CnzVals[prefixSum[i]] = denseOutputPerThread[t][index].second;
                            prefixSum[i]++;
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    return std::move(sumMat);
}

/*
 * Hash + SpA Hybrid with row wise distribution of work for dense columns
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHybrid3(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3, t4, t5;
    int nmatrices = matrices.size();
    int densityThreshold = 8 * 1024;
    int windowSize = 8 * 1024;
    int nthreads;
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    pvector<RIT> nnzPerColSparse(nnzPerCol.begin(), nnzPerCol.end());
    pvector<RIT> nnzPerColDense(nnzPerCol.begin(), nnzPerCol.end());
#pragma omp parallel for
    for (CPT i = 0; i < ncols; i++){
        if(nnzPerCol[i] > densityThreshold){
            nnzPerColSparse[i] = 0;
        }
        else{
            nnzPerColDense[i] = 0;
        }
    }
    
    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefixSum);
    pvector<CPT> prefixSumSparse(ncols+1);
    ParallelPrefixSum(nnzPerColSparse, prefixSumSparse);
    pvector<CPT> prefixSumDense(ncols+1);
    ParallelPrefixSum(nnzPerColDense, prefixSumDense);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCTotSparse = prefixSumSparse[ncols];
    CPT nnzCTotDense = prefixSumDense[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerThreadSparseExpected;
    CPT nnzCPerThreadDenseExpected;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    pvector< std::pair<bool,VT> > spa(nrows);
    pvector< std::vector <std::pair<RIT,VT> > > denseOutputPerThread;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            nnzCPerThreadSparseExpected = nnzCTotSparse / nthreads;
            nnzCPerThreadDenseExpected = nnzCTotDense / nthreads;
            ttimes.resize(nthreads);
            splitters.resize(nthreads);
            denseOutputPerThread.resize(nthreads);
        }
    }
    t1 = omp_get_wtime();
    printf("Time for checkpoint 1: %lf\n", t1-t0);

    RIT colProcessed = 0;
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
        splitters[tid] = std::lower_bound(prefixSumSparse.begin(), prefixSumSparse.end(), tid * nnzCPerThreadSparseExpected) - prefixSumSparse.begin();
#pragma omp barrier
#pragma omp for reduction(+: colProcessed) schedule(static) nowait
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] <= densityThreshold){
                    colProcessed++;
                    //----------- preparing the hash table for this column -------
                    size_t htSize = minHashTableSize;
                    while(htSize < nnzPerCol[i]){
                        htSize <<= 1;
                    }   
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j].first = -1;
                    }
                
                    //----------- add this column form all matrices -------
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }   
                        }
                    }
               
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j){
                        if (globalHashVec[j].first != -1){
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
                
                    std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                
                    for (size_t j=0; j < index; ++j){
                        CrowIds[prefixSum[i]] = globalHashVec[j].first;
                        CnzVals[prefixSum[i]] = globalHashVec[j].second;
                        prefixSum[i] ++;
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    printf("Processed sparse columns %d\n", colProcessed);
    printf("Time for checkpoint 2: %lf\n", t1-t0);
    printf("Stats of sparse portion:\n");
    getStats<double>(ttimes, true);
    
    colProcessed = 0;
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        for(CPT i = 0; i < ncols; i++){
            if(nnzPerCol[i] > densityThreshold){
                if(tid == 0) colProcessed++;
                //denseOutputPerThread[tid].clear();
                RIT nRowsPerThread = nrows / nthreads;
                RIT rowStart = tid * nRowsPerThread;
                RIT rowEnd = (tid == nthreads-1) ? nrows : (tid+1) * nRowsPerThread;
                pvector< std::pair<RIT,RIT> > rowRanges(nmatrices); 
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    rowRanges[k].first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart ) - rowIds->begin();
                    rowRanges[k].second = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd ) - rowIds->begin();
                }
                while(rowStart < rowEnd){
                    RIT windowStart = rowStart;
                    RIT windowEnd = std::min(windowStart + windowSize, rowEnd);
                    for (RIT j = windowStart; j < windowEnd; j++) spa[j].first = false;
                
                    for (int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                        for (RIT j = rowRanges[k].first; j < rowRanges[k].second; j++){
                            RIT rowId = (*rowIds)[j];
                            VT value = (*nzVals)[j];
                            if( rowId >= windowEnd ) {
                                rowRanges[k].first = j;
                                break;
                            }
                            else{
                                if (spa[rowId].first == false){
                                    spa[rowId].first = true;
                                    spa[rowId].second = value;
                                }
                                else{
                                    spa[rowId].second += value;
                                }

                                if(j == rowRanges[k].second - 1){
                                    rowRanges[k].first = rowRanges[k].second;
                                }
                            }
                        }
                    }
                    //for(RIT j = windowStart; j < windowEnd; j++){
                        //if(spa[j].first != false) denseOutputPerThread[tid].push_back(std::pair<RIT,VT>(j,spa[j].second));
                    //}
                    rowStart += windowSize;
                }

//#pragma omp barrier
                //RIT offset = (tid == 0) ? 0 : denseOutputPerThread[tid-1].size();
                //for(RIT idx = 0; idx < denseOutputPerThread[tid].size(); idx++){
                    //CrowIds[prefixSum[i] + offset + idx] = denseOutputPerThread[tid][idx].first;
                    //CnzVals[prefixSum[i] + offset + idx] = denseOutputPerThread[tid][idx].second;
                //}
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    printf("Processed dense columns %d\n", colProcessed);
    printf("Time for checkpoint 3: %lf\n", t1-t0);
    printf("Stats of dense portion:\n");
    getStats<double>(ttimes, true);

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

/*
 * Sliding hash
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHybrid4(std::vector<CSC<RIT, VT, CPT>* > & matrices, bool sorted=true)
{
    
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = 512 * 2;
    const RIT maxHashTableSizeSymbolic = 512 * 4;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    int nthreads;
    int nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<int64_t> flopsPerCol(ncols);
    pvector<int64_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel for
    for(CPT i = 0; i < ncols; i++){
        flopsPerCol[i] = 0;
        for(int k = 0; k < nmatrices; k++){
            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
            flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
        nWindowPerColSymbolic[i] = flopsPerCol[i] / maxHashTableSizeSymbolic + 1;
    }
    t3 = omp_get_wtime();
    //printf("[Hybrid4]\tTime for number of window calculation: %lf\n", t3-t2);
    

    pvector<int64_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<int64_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    int64_t flopsTot = prefixSumSymbolic[ncols];
    int64_t flopsPerThreadExpected;
    
    t1 = omp_get_wtime();
    printf("[Sliding Hash]\tTime before symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of number of windows of symbolic:\n");
    getStats<int64_t>(nWindowPerColSymbolic, true);

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            ttimes.resize(nthreads);
            splitters.resize(nthreads);
            flopsPerThreadExpected = flopsTot / nthreads;
            //printf("Expected flops per thread: %lld\n", flopsPerThreadExpected);
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), tid * flopsPerThreadExpected) - prefixSumSymbolic.begin();
        //printf("splitters[%d] %d: target %lld\n", tid, splitters[tid], tid * flopsPerThreadExpected);
#pragma omp barrier
#pragma omp for schedule(static) nowait
        for (int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            //printf("Thread %d processing %d columns: [%d, %d)\n", t, colEnd-colStart, colStart, colEnd);
            for(CPT i = colStart; i < colEnd; i++){
                int64_t nwindows = nWindowPerColSymbolic[i];
                RIT nrowsPerWindow = nrows / nwindows;
                nnzPerCol[i] = 0;
                nWindowPerCol[i] = 0;
                RIT runningSum = 0;
                for(size_t w=0; w < nwindows; w++){
                    // Determine the start and end row index
                    RIT rowStart = w * nrowsPerWindow;
                    RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;
                    
                    int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                    nnzPerWindowSymbolic[wIdx].first = rowStart;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    // Determine total number of input nonzeros in the window
                    size_t flopsWindow = 0;
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin();
                        auto last = rowIds->end();
                        int64_t startIdx, endIdx, midIdx;

                        //t2 = omp_get_wtime();
                        if(rowStart > 0){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            first = rowIds->begin() + endIdx;
                        }
                        else{
                            first = rowIds->begin() + (*colPtr)[i];
                        }

                        if(rowEnd < nrows){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            last = rowIds->begin() + endIdx;
                        }
                        else{
                            last = rowIds->begin() + (*colPtr)[i+1];
                        }
                       
                        //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        //first = (rowStart == 0) ? rowIds->begin() + (*colPtr)[i] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = (rowEnd == nrows) ? rowIds->begin() + (*colPtr)[i+1] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );

                        flopsWindow += last-first;
                    }
                    
                    // In worst case hash table may need to store all input nonzeros in the window
                    // So resize the hash table accordingly
                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }
                    
                    // For each matrix
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin();
                        auto last = rowIds->end();
                        int64_t startIdx, endIdx, midIdx;

                        //t2 = omp_get_wtime();
                        if(rowStart > 0){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            first = rowIds->begin() + endIdx;
                        }
                        else{
                            first = rowIds->begin() + (*colPtr)[i];
                        }

                        if(rowEnd < nrows){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            last = rowIds->begin() + endIdx;
                        }
                        else{
                            last = rowIds->begin() + (*colPtr)[i+1];
                        }

                        // Determine the range of elements that fall within the window
                        //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        //first = (rowStart == 0) ? rowIds->begin() + (*colPtr)[i] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        //last = (rowEnd == nrows) ? rowIds->begin() + (*colPtr)[i+1] : std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        
                        for(; first<last; first++){
                            RIT key = *first;
                            RIT hash = (key*hashScale) & (htSize-1);
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash] == key) //key is found in hash table
                                {
                                    break;
                                }
                                else if (globalHashVec[hash] == -1) //key is not registered yet
                                {
                                    globalHashVec[hash] = key;
                                    nnzPerCol[i]++;
                                    nnzPerWindowSymbolic[wIdx].second++;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }
                        }
                    }
                    if (w == 0){
                        nWindowPerCol[i] = 1;
                        runningSum = nnzPerWindowSymbolic[wIdx].second;
                    }
                    else{
                        if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                            runningSum = nnzPerWindowSymbolic[wIdx].second;
                            nWindowPerCol[i]++;
                        }
                        else{
                            runningSum = runningSum + nnzPerWindowSymbolic[wIdx].second;
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
    printf("[Sliding Hash]\tTime for symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of time consumed by threads:\n");
    getStats<double>(ttimes, true);

    //size_t maxWindowSymbolic=0;
    //size_t minWindowSymbolic=ncols;
    //size_t avgWindowSymbolic=0;
    //size_t maxWindow=0;
    //size_t minWindow = ncols;
    //size_t avgWindow=0;
////#pragma omp parallel for reduction
    //for (CPT i = 0; i < ncols; i++){
        //maxWindowSymbolic = std::max(maxWindowSymbolic, nnzPerWindowPerColSymbolic[i].size());
        //minWindowSymbolic = std::min(minWindowSymbolic, nnzPerWindowPerColSymbolic[i].size());
        //avgWindowSymbolic = avgWindowSymbolic + nnzPerWindowPerColSymbolic[i].size();
        //maxWindow = std::max(maxWindow, nnzPerWindowPerCol[i].size());
        //minWindow = std::min(minWindow, nnzPerWindowPerCol[i].size());
        //avgWindow = avgWindow + nnzPerWindowPerCol[i].size();
    //}
    //avgWindowSymbolic /= ncols;
    //avgWindow /= ncols;

    //printf("Window statistics for symbolic: max %d, min %d, avg %d\n", maxWindowSymbolic, minWindowSymbolic, avgWindowSymbolic);
    //printf("Window statistics for computation: max %d, min %d, avg %d\n", maxWindow, minWindow, avgWindow);
    
    pvector<RIT> prefixSumWindow(ncols+1, 0);
    ParallelPrefixSum(nWindowPerCol, prefixSumWindow);

    //for(CPT i = 0; i < ncols; i++){
        //printf("%d -> %d\n", nWindowPerColSymbolic[i], nWindowPerCol[i]);
    //}

    printf("[Sliding Hash]\tStats of number of windows:\n");
    getStats<RIT>(nWindowPerCol, true);

    pvector< std::pair<RIT, RIT> > nnzPerWindow(prefixSumWindow[ncols]);
    
    t0 = omp_get_wtime();
#pragma omp for schedule(static)
    for(int t = 0; t < nthreads; t++){
        CPT colStart = splitters[t];
        CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
        for(CPT i = colStart; i < colEnd; i++){
            int64_t nwindows = nWindowPerColSymbolic[i];
            int64_t wsIdx = prefixSumWindowSymbolic[i];
            int64_t wcIdx = prefixSumWindow[i];
            nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
            nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
            for(size_t w=1; w < nwindows; w++){
                wsIdx = prefixSumWindowSymbolic[i] + w;
                if(nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second > maxHashTableSize){
                    wcIdx++;
                    nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
                    nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
                }
                else{
                    nnzPerWindow[wcIdx].second = nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second;
                }
            }
        }
    }
    t1 = omp_get_wtime();
    printf("[Hybrid4]\tTime for collapsing windows: %lf\n", t1-t0);

    //for (CPT i = 0; i < ncols; i++){
        //printf("Column %d: ", i);
        //printf("%d windows ", nWindowPerColSymbolic[i]);
        //for(int w = 0; w < nWindowPerColSymbolic[i]; w++){
            //int wIdx = prefixSumWindowSymbolic[i]+w;
            //printf("(start: %d, nnz:%d) ", nnzPerWindowSymbolic[wIdx].first, nnzPerWindowSymbolic[wIdx].second);
        //}
        //printf(" -> ");
        //printf("%d windows ", nWindowPerCol[i]);
        //for(int w = 0; w < nWindowPerCol[i]; w++){
            //int wIdx = prefixSumWindow[i]+w;
            //printf("(start: %d, nnz:%d) ", nnzPerWindow[wIdx].first, nnzPerWindow[wIdx].second);
        //}
        //printf("\n");
    //}

    pvector<CPT> prefixSum(ncols+1, 0);
    ParallelPrefixSum(nnzPerCol, prefixSum);

    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
        splitters[tid] = std::lower_bound(prefixSum.begin(), prefixSum.end(), tid * nnzCPerThreadExpected) - prefixSum.begin();
#pragma omp barrier
#pragma omp for schedule(static) nowait
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                RIT nwindows = nWindowPerCol[i];
                pvector< std::pair<RIT, RIT> > rowIdsRange(nmatrices);
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    rowIdsRange[k].first = (*colPtr)[i];
                    rowIdsRange[k].second = (*colPtr)[i+1];
                }
                for (int w = 0; w < nwindows; w++){
                    RIT wIdx = prefixSumWindow[i] + w;
                    RIT rowStart = nnzPerWindow[wIdx].first;
                    RIT rowEnd = (w == nWindowPerCol[i]-1) ? nrows : nnzPerWindow[wIdx+1].first;
                    RIT nnzWindow = nnzPerWindow[wIdx].second;

                    size_t htSize = minHashTableSize;
                    while(htSize < nnzWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize)
                        globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j)
                    {
                        globalHashVec[j].first = -1;
                    }

                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                        while( (rowIdsRange[k].first < rowIdsRange[k].second) && ((*rowIds)[rowIdsRange[k].first] < rowEnd) ){
                            RIT j = rowIdsRange[k].first;
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }

                            rowIdsRange[k].first++;
                        } 
                    }
                    //if(tid == 0) printf("Thread %d finished window %d out of %d windows of column %d in range [%d,%d)\n", tid, w, nwindows, i, colStart, colEnd);
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j){
                        if (globalHashVec[j].first != -1){
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
            
                    std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                
                    for (size_t j=0; j < index; ++j){
                        CrowIds[prefixSum[i]] = globalHashVec[j].first;
                        CnzVals[prefixSum[i]] = globalHashVec[j].second;
                        prefixSum[i] ++;
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    printf("[Hybrid4]\tTime for computation: %lf\n", t1-t0);
    printf("Stats of time consumed by threads:\n");
    getStats<double>(ttimes, true);

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddSpA(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    //printf("Using SpMultiAddSpA\n");
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();
    

    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<RIT> nnzPerThread; // To record number of nnz processed by each thread 
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    auto nnzPerColStats = getStats<RIT>(nnzPerCol);
    int nthreads;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            ttimes.resize(nthreads);
            nnzPerThread.resize(nthreads);
            splitters.resize(nthreads);
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), tid * nnzCPerThreadExpected) - prefix_sum.begin();
#pragma omp barrier
        nnzPerThread[tid] = 0;
        std::vector< std::pair<RIT,VT> > spa(nrows);
#pragma omp for schedule(static) nowait 
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = t < nthreads-1 ? splitters[t+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                for (RIT j = 0; j < nrows; j++){
                    spa[j].first = -1;
                }
                for(int k = 0; k < nmatrices; k++)
                {
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                
                    for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                    {
                        RIT key = (*rowIds)[j];
                        VT curval = (*nzVals)[j];
                        if (spa[key].first == -1){
                            spa[key].first = key;
                            spa[key].second = curval;
                        }
                        else{
                            spa[key].second += curval;
                        }
                    }
                }
                size_t index = 0;
                for (size_t j=0; j < nrows; ++j){
                    if (spa[j].first != -1){
                        spa[index++] = spa[j];
                    }
                }
            
                for (size_t j=0; j < index; ++j)
                {
                    CrowIds[prefix_sum[i]] = spa[j].first;
                    CnzVals[prefix_sum[i]] = spa[j].second;
                    prefix_sum[i] ++;
                }
            }
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }

    t1 = omp_get_wtime();

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddSpA2(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3, t4, t5;
    int nmatrices = matrices.size();
    int nthreads;
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefixSum);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    pvector<double> ttimes; // To record time taken by each thread

    std::vector< std::pair<bool,VT> > spa(nrows);
    pvector< std::vector <std::pair<RIT,VT> > > denseOutputPerThread;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            ttimes.resize(nthreads);
            denseOutputPerThread.resize(nthreads);
        }
    }
    t1 = omp_get_wtime();
    printf("Time for checkpoint 1: %lf\n", t1-t0);
    
    //int windowSize = 128*1024;
    int windowSize = 2;
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        for(CPT i = 0; i < ncols; i++){
            if(nnzPerCol[i] > 0){
                denseOutputPerThread[tid].clear();
                RIT nRowsPerThread = nrows / nthreads;
                RIT rowStart = tid * nRowsPerThread;
                RIT rowEnd = (tid == nthreads-1) ? nrows : (tid+1) * nRowsPerThread;
                pvector< std::pair<RIT,RIT> > rowRanges(nmatrices); 
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    rowRanges[k].first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart ) - rowIds->begin();
                    rowRanges[k].second = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd ) - rowIds->begin();
                }
                while(rowStart < rowEnd){
                    RIT windowStart = rowStart;
                    RIT windowEnd = std::min(windowStart + windowSize, rowEnd);
                    for (RIT j = windowStart; j < windowEnd; j++) spa[j].first = false;
                
                    for (int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                        for (RIT j = rowRanges[k].first; j < rowRanges[k].second; j++){
                            RIT rowId = (*rowIds)[j];
                            VT value = (*nzVals)[j];
                            if( rowId >= windowEnd ) {
                                rowRanges[k].first = j;
                                break;
                            }
                            else{
                                if (spa[rowId].first == false){
                                    spa[rowId].first = true;
                                    spa[rowId].second = value;
                                }
                                else{
                                    spa[rowId].second += value;
                                }

                                if(j == rowRanges[k].second - 1){
                                    rowRanges[k].first = rowRanges[k].second;
                                }
                            }
                        }
                    }
                    for(RIT j = windowStart; j < windowEnd; j++){
                        if(spa[j].first != false) denseOutputPerThread[tid].push_back(std::pair<RIT,VT>(j,spa[j].second));
                    }
                    rowStart += windowSize;
                }

#pragma omp barrier
                RIT offset = (tid == 0) ? 0 : denseOutputPerThread[tid-1].size();
                for(RIT idx = 0; idx < denseOutputPerThread[tid].size(); idx++){
                    CrowIds[prefixSum[i] + offset + idx] = denseOutputPerThread[tid][idx].first;
                    CnzVals[prefixSum[i] + offset + idx] = denseOutputPerThread[tid][idx].second;
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("Time for checkpoint 3: %lf\n", t1-t0);
    printf("Stats of dense portion:\n");
    getStats<double>(ttimes, true);

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    return std::move(sumMat);
}

/*
 *  Estimates nnumber of non-zeroes when adding two CSC matrices in regular way
 *  Assumes that entries of each column are sorted according to the order of row id
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
pvector<RIT> symbolicSpAddRegular(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B){
    double t0, t1, t3, t4;

    t0 = omp_get_wtime();

    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();
    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();
    
    pvector<RIT> nnzCPerCol(ncols);
    
    t1 = omp_get_wtime();
    //printf("[symbolicSpAddRegular] Time taken before parallel section %lf\n", (t1-t0));
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
#pragma omp for schedule(dynamic, 100)
        // Process each column in parallel
        for(CIT i = 0; i < ncols; i++){
            RIT ArowsStart = (*AcolPtr)[i];
            RIT ArowsEnd = (*AcolPtr)[i+1];
            RIT BrowsStart = (*BcolPtr)[i];
            RIT BrowsEnd = (*BcolPtr)[i+1];
            RIT Aptr = ArowsStart;
            RIT Bptr = BrowsStart;
            nnzCPerCol[i] = 0;
            
            while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                if (Aptr >= ArowsEnd){
                    // Entries of A has finished
                    // Increment nnzCPerCol[i]
                    // Increment BPtr
                    nnzCPerCol[i]++;
                    Bptr++;
                }
                else if (Bptr >= BrowsEnd){
                    // Entries of B has finished
                    // Increment nnzCPerCol[i]
                    // Increment APtr
                    nnzCPerCol[i]++;
                    Aptr++;
                }
                else {
                    if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                        // Increment nnzCPerCol[i]
                        // Increment APtr 
                        nnzCPerCol[i]++;
                        Aptr++;
                    }
                    else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                        // Increment nnzCPerCol[i]     
                        // Increment BPtr
                        nnzCPerCol[i]++;
                        Bptr++;
                    }
                    else{
                        // Increment nnzCPerCol[i]
                        // Increment APtr, BPtr 
                        nnzCPerCol[i]++;
                        Aptr++;
                        Bptr++;
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        //printf( "[symbolicSpAddRegular] Time taken by thread %d is %f\n", omp_get_thread_num(), ttime );
    }
    t1 = omp_get_wtime();
    //printf("[symbolicSpAddRegular] Time taken for parallel section %lf\n", (t1-t0));
    return std::move(nnzCPerCol);
}

/*
 *  Adds two CSC matrices in regular way (the way merge operation of MergeSort works)
 *  Assumes that entries of each column are sorted according to the order of row id
 *  Assumes that all sanity checks are done before, so do not perform any
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpAddRegular(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B, pvector<RIT> & nnzCPerCol){
    double t0, t1, t3, t4;

    t0 = omp_get_wtime();

    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();
    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();

    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzCPerCol, prefixSum);
    CSC<RIT, VT, CPT> C(nrows, ncols, prefixSum[ncols], false, true);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    
    CPT nnzCTot = prefixSum[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<RIT> nnzPerThread; // To record number of nnz processed by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    int nthreads;

    t1 = omp_get_wtime();
    //printf("[SpAddRegular] Time taken before parallel section %lf\n", (t1-t0));
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nnzCPerThreadExpected = nnzCTot / nthreads;
            ttimes.resize(nthreads);
            nnzPerThread.resize(nthreads);
            splitters.resize(nthreads);
        }
#pragma omp barrier
        splitters[tid] = std::lower_bound(prefixSum.begin(), prefixSum.end(), tid * nnzCPerThreadExpected) - prefixSum.begin();
#pragma omp barrier
        nnzPerThread[tid] = 0;
#pragma omp for schedule(static) nowait
        for(int t = 0; t < nthreads; t++){
            CPT colStart = splitters[t];
            CPT colEnd = t < nthreads-1 ? splitters[t+1] : ncols;
            for (CPT i = colStart; i < colEnd; i++){
                RIT ArowsStart = (*AcolPtr)[i];
                RIT ArowsEnd = (*AcolPtr)[i+1];
                RIT BrowsStart = (*BcolPtr)[i];
                RIT BrowsEnd = (*BcolPtr)[i+1];
                RIT Aptr = ArowsStart;
                RIT Bptr = BrowsStart;
                RIT Cptr = prefixSum[i];

                while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                    if (Aptr >= ArowsEnd){
                        // Entries of A has finished
                        // Copy the entry of BPtr to the CPtr
                        // Increment BPtr and CPtr
                        CrowIds[Cptr] = (*BrowIds)[Bptr];
                        CnzVals[Cptr] = (*BnzVals)[Bptr];
                        Bptr++;
                        Cptr++;
                    }
                    else if (Bptr >= BrowsEnd){
                        // Entries of B has finished
                        // Copy the entry of APtr to the CPtr
                        // Increment APtr and CPtr
                        CrowIds[Cptr] = (*ArowIds)[Aptr];
                        CnzVals[Cptr] = (*AnzVals)[Aptr];
                        Aptr++;
                        Cptr++;
                    }
                    else {
                        if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                            // Copy the entry of APtr to the CPtr
                            // Increment APtr and CPtr
                            CrowIds[Cptr] = (*ArowIds)[Aptr];
                            CnzVals[Cptr] = (*AnzVals)[Aptr];
                            Aptr++;
                            Cptr++;
                        }
                        else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                            // Copy the entry of BPtr to the CPtr
                            // Increment BPtr and CPtr
                            CrowIds[Cptr] = (*BrowIds)[Bptr];
                            CnzVals[Cptr] = (*BnzVals)[Bptr];
                            Bptr++;
                            Cptr++;
                        }
                        else{
                            // Sum the entries of APtr and BPtr then store at CPtr
                            // Increment APtr, BPtr and CPtr
                            CrowIds[Cptr] = (*ArowIds)[Aptr];
                            CnzVals[Cptr] = (*AnzVals)[Aptr] + (*BnzVals)[Bptr];
                            Aptr++;
                            Bptr++;
                            Cptr++;
                        }
                    }
                }
                //nnzPerThread[tid] += (ArowsEnd - ArowsStart) + (BrowsEnd - BrowsStart);
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("[SpAddRegular] Time taken for parallel section %lf\n", (t1-t0));

    t0 = omp_get_wtime();
    
    //printf("[SpAddRegular] Stats of time consumed by threads\n");
    //getStats<double>(ttimes);
    //printf("[SpAddRegular] Stats of nnz processed by threads\n");
    //getStats<RIT>(nnzPerThread);


    C.cols_pvector(&CcolPtr);
    C.nz_rows_pvector(&CrowIds);
    C.nz_vals_pvector(&CnzVals);

    t1 = omp_get_wtime();
    //printf("[SpAddRegular] Time taken after parallel section %lf\n", (t1-t0));

    return std::move(C);
}

/*
 *  Routing function to add two CSC matrices
 *  Checks different criteria to decide which implementation to use
 *  Returns another CSC matrix
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpAdd(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B, bool inputSorted=true, bool outputSorted=true){
    double t0, t1, t2, t3; 

    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();

    // ---------- checking if matrices can be added ------------------
    if( (nrows != B->get_nrows()) || (ncols != B->get_ncols()) ){
        std::cerr << " Can not be added as matrix dimensions do not agree. Returning an empty matrix. \n";
        return CSC<RIT, VT, CPT>();
    }

    std::vector<CSC<RIT, VT, CPT>*> matrices(2);
    matrices[0] = A;
    matrices[1] = B;
    t0 = omp_get_wtime();
    //pvector<RIT> nnzCPerCol = symbolicSpMultiAddHash<RIT, CIT, VT, CPT, int32_t>(matrices);
    pvector<RIT> nnzCPerCol = symbolicSpAddRegular<RIT, CIT, VT, CPT>(A, B);
    t1 = omp_get_wtime();
    //printf("[SpAdd] Time taken by symbolic: %lf\n", t1-t0);

    //printf("[SpAdd] Stats of nnzCPerCol\n");
    //getStats<RIT>(nnzCPerCol);

    if(inputSorted){
        // Use SpAddRegular
        return SpAddRegular<RIT, CIT, VT, CPT>(A, B, nnzCPerCol);
        //return SpMultiAddHash<RIT, CIT, VT, CPT>(matrices, nnzCPerCol);
    }
    else{
        // To Do: Probably using SpAddHash would be better      
        // Temporarily using regular one for the sake of semantic correctness of the function
        return SpAddRegular<RIT, CIT, VT, CPT>(A, B, nnzCPerCol);
    } 
    
}

/*
 *  Routing function to add multiple CSC matrices
 *  Takes a vector of CSC matrices as input
 *  Checks different criteria to decide which implementation to use
 *  Returns another CSC matrix
 *
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAdd(std::vector<CSC<RIT, VT, CPT>* > & matrices, int version, bool inputSorted=true, bool outputSorted=true){

    int nmatrices = matrices.size();
    
    if(nmatrices == 0) return CSC<RIT, VT, CPT>();
    double t0, t1, t3, t4;

    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    for(int i = 1; i < nmatrices; i++)
    {
        if( (ncols != matrices[i]->get_ncols()) || (nrows != matrices[i]->get_nrows()))
        {
            std::cerr << " Can not be added as matrix dimensions do not agree. Returning an empty matrix. \n";
            return CSC<RIT, VT, CPT>();
        }
    }

    if(version != 4) {
        t0 = omp_get_wtime();
        pvector<RIT> nnzCPerCol = symbolicSpMultiAddHash<RIT, CIT, VT, CPT, int32_t>(matrices);
        t1 = omp_get_wtime();
        printf("Time for symbolic: %lf\n", t1-t0);
        if(version == 0) return SpMultiAddHash<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        else if(version == 1) return SpMultiAddHybrid<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        else if(version == 2) return SpMultiAddHybrid2<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        else if(version == 3) return SpMultiAddHybrid3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        else return SpMultiAddHybrid3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
    }
    else {
        return SpMultiAddHybrid4<RIT, CIT, VT, CPT>(matrices, true);
    }
    
    //t0 = omp_get_wtime();
    //pvector<RIT> nnzCPerCol1 = symbolicSpMultiAddHashSliding<RIT, CIT, VT, CPT, int32_t>(matrices);
    //t1 = omp_get_wtime();
    //printf("Time for symbolic sliding new: %lf\n", t1-t0);
    
    //t0 = omp_get_wtime();
    //pvector<RIT> nnzCPerCol2 = symbolicSpMultiAddHashSliding1<RIT, CIT, VT, CPT, int32_t>(matrices);
    //t1 = omp_get_wtime();
    //printf("Time for symbolic sliding old: %lf\n", t1-t0);
    
    //for(CIT i=0; i< ncols; i++)
    //{
        //if(nnzCPerCol[i] != nnzCPerCol1[i]) std::cout << "not equal" << std::endl;
    //}
    //printf("Symbolic Equal!\n");
    
    
    //return SpMultiAddSpA3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
}


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_3(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          // optimised, used symbolic step to calculate the space needed and so no push_backs
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<RIT> nz_per_column = symbolic_add_vec_of_matrices<RIT, CIT, VT, CPT, NM>(vec_of_matrices);

	pvector<CPT> prefix_sum(num_of_columns+1);
	// pvector<CPT> column_vector_for_csc(num_of_columns+1);
	// prefix_sum[0] = 0;
	// column_vector_for_csc[0] = 0;

	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + nz_per_column[i-1];
	// 	column_vector_for_csc[i] = prefix_sum[i];
	// }

	ParallelPrefixSum(nz_per_column, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());

	pvector<VT> value_vector_for_csc(prefix_sum[num_of_columns]);
	pvector<RIT> row_vector_for_csc(prefix_sum[num_of_columns]);


#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){

		std::unordered_map<RIT, VT> umap;
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
			const pvector<VT> *nz_i = vec_of_matrices[k]->get_nzVals();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				umap[(*row_ids_i)[j] ] += (*nz_i)[j];
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;
			nz_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;
			delete nz_i;
		}

		for(auto iter = umap.begin(); iter != umap.end(); iter++){
			row_vector_for_csc[prefix_sum[i] ] = iter->first;
			value_vector_for_csc[prefix_sum[i] ] = iter->second;
			prefix_sum[i] ++;
		}
	}  // parallel programming ended



	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, column_vector_for_csc[num_of_columns], false, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	result_matrix_csc.sort_inside_column();

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}






//..............................use of hash functions are done by here............................................//


template <typename RIT, typename VT, typename NM>
struct heap_help{
	RIT hh_row_number;
	VT hh_element;
	NM hh_matrix_number;

	heap_help() : hh_row_number(0), hh_element(0), hh_matrix_number(0){}
	heap_help(RIT temp1, VT temp2, NM temp3) : hh_row_number(temp1), hh_element(temp2), hh_matrix_number(temp3){}

	template <typename RIT_1, typename VT_1, typename NM_1>
	bool operator< (const heap_help<RIT_1, VT_1, NM_1 > & other) const{
		return this->hh_row_number > other.hh_row_number; // < here is inverted on purpose so as use .top() of priority queue to extract minimum
	}
};



//..........................................................................//



template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_4(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          //used HEAPS here. optimised, used symbolic step to calculate the space needed and so no push_backs
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<RIT> nz_per_column = symbolic_add_vec_of_matrices<RIT, CIT, VT, CPT, NM>(vec_of_matrices); 

	pvector<CPT> prefix_sum(num_of_columns+1);
	// pvector<CPT> column_vector_for_csc(num_of_columns+1);
	// prefix_sum[0] = 0;
	// column_vector_for_csc[0] = 0;

	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + nz_per_column[i-1];
	// 	column_vector_for_csc[i] = prefix_sum[i];
	// }
	ParallelPrefixSum(nz_per_column, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());


	pvector<VT> value_vector_for_csc(prefix_sum[num_of_columns]);
	pvector<RIT> row_vector_for_csc(prefix_sum[num_of_columns]);


#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){

			std::priority_queue<heap_help<RIT, VT, NM> > pq;
			NM number_of_matrices = vec_of_matrices.size();

			pvector<RIT> current_index_of_specific_matrix(number_of_matrices,0);

			for(NM j = 0; j < number_of_matrices; j++){

				const pvector<CPT> *col_ptr_j = vec_of_matrices[j]->get_colPtr();
				const pvector<RIT> *row_ids_j = vec_of_matrices[j]->get_rowIds();
				const pvector<VT> *nz_j = vec_of_matrices[j]->get_nzVals();

				if( (*col_ptr_j)[i] + current_index_of_specific_matrix[j] < (*col_ptr_j)[i+1] )
				{
					RIT temp1_row = (*row_ids_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
					VT temp1_value = (*nz_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
					pq.push(heap_help<RIT, VT, NM> (temp1_row, temp1_value, j));
					current_index_of_specific_matrix[j]++;
				}

				col_ptr_j = nullptr;
				row_ids_j = nullptr;
				nz_j = nullptr;

				delete col_ptr_j;
				delete row_ids_j;
				delete nz_j;
			}

			bool first_entry_check = false;
			while(! pq.empty()){
				heap_help<RIT, VT, NM> temp1 = pq.top(); // T as long double
				pq.pop();
				if(!first_entry_check)
				{
					value_vector_for_csc[prefix_sum[i]] = temp1.hh_element;
					row_vector_for_csc[prefix_sum[i]] = temp1.hh_row_number;
					prefix_sum[i]++;
					first_entry_check = true;
				}
				else
				{
					if(row_vector_for_csc[prefix_sum[i] - 1] == temp1.hh_row_number)
					{
						value_vector_for_csc[prefix_sum[i] - 1] += temp1.hh_element;
					}
					else
					{
						value_vector_for_csc[prefix_sum[i]] = temp1.hh_element;
						row_vector_for_csc[prefix_sum[i]] = temp1.hh_row_number;
						prefix_sum[i]++;
					}
				}

				NM matrix_removed_from_pq = temp1.hh_matrix_number;
				const pvector<CPT> *col_ptr_j = vec_of_matrices[matrix_removed_from_pq]->get_colPtr();
				const pvector<RIT> *row_ids_j = vec_of_matrices[matrix_removed_from_pq]->get_rowIds();
				const pvector<VT> *nz_j = vec_of_matrices[matrix_removed_from_pq]->get_nzVals();

				if( (*col_ptr_j)[i] + current_index_of_specific_matrix[matrix_removed_from_pq] < (*col_ptr_j)[i+1] )
				{
					RIT temp1_row = (*row_ids_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[matrix_removed_from_pq]] ;
					VT temp1_value = (*nz_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[matrix_removed_from_pq]] ;

					pq.push(heap_help<RIT, VT, NM> (temp1_row, temp1_value, matrix_removed_from_pq));
					current_index_of_specific_matrix[matrix_removed_from_pq]++;
				}

				col_ptr_j = nullptr;
				row_ids_j = nullptr;
				nz_j = nullptr;

				delete col_ptr_j;
				delete row_ids_j;
				delete nz_j;

			} //end while


	}  // parallel programming ended



	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, column_vector_for_csc[num_of_columns], true, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}


//..........................................................................//


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>

pvector<std::tuple<RIT,size_t,RIT> > symbolic_add_vec_of_matrices_modified(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)   // gives a tuple of rows in final sum, total elements in a column over all matrices, max row number in a column all required for radix sort.
{

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();

	pvector<std::tuple<RIT,size_t,RIT> > nz_per_column(num_of_columns);

#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){

		RIT max_row_number = 0;
		size_t total_elements = 0;
		std::unordered_map<RIT, VT> umap;
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				umap[(*row_ids_i)[j] ] ++;
				max_row_number = std::max(max_row_number, (*row_ids_i)[j]);
			}

			total_elements += (*col_ptr_i)[i+1] - (*col_ptr_i)[i];

			col_ptr_i = nullptr;
			row_ids_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;

		}
		std::get<0>(nz_per_column[i]) = umap.size();
		std::get<2>(nz_per_column[i]) = max_row_number;
		std::get<1>(nz_per_column[i]) = total_elements;

	} 
	// parallel programming ended
	return std::move(nz_per_column);

}



//..........................................................................//


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

void count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon)
{


	size_t num_of_elements = all_elements.size();

	pvector<std::pair<RIT, VT> > temp_array(num_of_elements);
	size_t count[10] = {0};
	size_t index_for_count;

	for(size_t i = 0; i < num_of_elements; i++){
		index_for_count = ((all_elements[i].first)/expon)%10;
		count[index_for_count]++;
	}

	for(int i = 1; i < 10; i++){
		count[i] += count[i-1];
	}

	for(size_t i = num_of_elements-1; i > 0; i--){
		index_for_count = ((all_elements[i].first)/expon)%10;
		temp_array[count[index_for_count] -1] = all_elements[i];
		count[index_for_count]--;
	}


	index_for_count = ((all_elements[0].first)/expon)%10;
	temp_array[count[index_for_count] -1] = all_elements[0];

	all_elements = std::move(temp_array);

	return;
}


//..........................................................................//



template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_5(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          //used RADIX SORT here. optimised, used symbolic step to calculate the space needed and so no push_backs
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<std::tuple<RIT, size_t, RIT> > nz_per_column = symbolic_add_vec_of_matrices_modified<RIT, CIT, VT, CPT, NM>(vec_of_matrices);  // has nnz in sum pre column, total elements in all matrices per column, max element in all rows in all matrices per column as tuple

	pvector<RIT> nz_per_column_in_sum(num_of_columns);

#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){
		nz_per_column_in_sum[i] = std::get<0>(nz_per_column[i]);
	} //parallel ended

	pvector<CPT> prefix_sum(num_of_columns+1);
	// pvector<CPT> column_vector_for_csc(num_of_columns+1);
	// prefix_sum[0] = 0;
	// column_vector_for_csc[0] = 0;

	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + std::get<0>(nz_per_column[i-1]);
	// 	column_vector_for_csc[i] = prefix_sum[i];
	// }
	ParallelPrefixSum(nz_per_column_in_sum, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());


	pvector<VT> value_vector_for_csc(prefix_sum[num_of_columns]);
	pvector<RIT> row_vector_for_csc(prefix_sum[num_of_columns]);


#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){

		pvector<std::pair<RIT, VT> > all_elements(std::get<1>(nz_per_column[i]) );
		NM number_of_matrices = vec_of_matrices.size();
		size_t current_index = 0;

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
			const pvector<VT> *nz_i = vec_of_matrices[k]->get_nzVals();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				all_elements[current_index] = std::pair<RIT, VT>((*row_ids_i)[j], (*nz_i)[j]);
				current_index++;
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;
			nz_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;
			delete nz_i;
		}


		RIT max_row_number = std::get<2>(nz_per_column[i]);
		for(size_t expon = 1; max_row_number/expon > 0; expon *= 10){
			count_sort<RIT, CIT, VT, CPT, NM>(all_elements, expon);
		}


		size_t num_of_elements = all_elements.size();
		VT sum = 0;
		for(size_t j = 0; j < num_of_elements; j++){
			if(j == 0){
				value_vector_for_csc[prefix_sum[i]] = all_elements[j].second;
				row_vector_for_csc[prefix_sum[i]] = all_elements[j].first;
				prefix_sum[i]++;
			}else{
				if(all_elements[j].first == row_vector_for_csc[prefix_sum[i] - 1]){
					value_vector_for_csc[prefix_sum[i] - 1] += all_elements[j].second;
				}else{
					value_vector_for_csc[prefix_sum[i]] = all_elements[j].second;
					row_vector_for_csc[prefix_sum[i]] = all_elements[j].first;
					prefix_sum[i]++;
				}
			}
		}


	}  // parallel programming ended



	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, column_vector_for_csc[num_of_columns], true, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}





//..........................................................................//



template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_6(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          // used long array(in the sense of maintaining a vector of row size for every column) here.optimised, used symbolic step to calculate the space needed and so no push_backs
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<RIT> nz_per_column = symbolic_add_vec_of_matrices<RIT, CIT, VT, CPT, NM>(vec_of_matrices); 

	pvector<CPT> prefix_sum(num_of_columns+1);
	// pvector<CPT> column_vector_for_csc(num_of_columns+1);
	// prefix_sum[0] = 0;
	// column_vector_for_csc[0] = 0;

	// for(CIT i = 1; i < num_of_columns+1; i++){
	// 	prefix_sum[i] = prefix_sum[i-1] + nz_per_column[i-1];
	// 	column_vector_for_csc[i] = prefix_sum[i];
	// }

	ParallelPrefixSum(nz_per_column, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());


	pvector<VT> value_vector_for_csc(prefix_sum[num_of_columns]);
	pvector<RIT> row_vector_for_csc(prefix_sum[num_of_columns]);


#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){

		pvector<VT> row_size_vector(num_of_rows, 0);
		pvector<bool> is_occupied(num_of_rows, false);
		NM number_of_matrices = vec_of_matrices.size();

		for(NM k = 0; k < number_of_matrices; k++){

			const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
			const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
			const pvector<VT> *nz_i = vec_of_matrices[k]->get_nzVals();

			for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++){
				row_size_vector[(*row_ids_i)[j] ] += (*nz_i)[j];
				is_occupied[(*row_ids_i)[j] ] = true;
			}

			col_ptr_i = nullptr;
			row_ids_i = nullptr;
			nz_i = nullptr;

			delete col_ptr_i;
			delete row_ids_i;
			delete nz_i;
		}

		for(RIT j = 0; j < num_of_rows; j++){
			if(is_occupied[j]){
				value_vector_for_csc[prefix_sum[i] ] = row_size_vector[j];
				row_vector_for_csc[prefix_sum[i] ] = j;
				prefix_sum[i]++;
			}
		}


	}  // parallel programming ended



	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, column_vector_for_csc[num_of_columns], true, true);
	result_matrix_csc.nz_rows_pvector(&row_vector_for_csc);
	result_matrix_csc.cols_pvector(&column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(&value_vector_for_csc);

	clock.Stop();
	//PrintTime("CSC Creation Time", clock.Seconds());

	return std::move(result_matrix_csc);

}

#endif
