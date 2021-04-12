#ifndef CSC_ADDER_H
#define CSC_ADDER_H

#include "CSC.h" // need to check the relative paths for this section
#include "GAP/pvector.h"
#include "GAP/timer.h"
#include "utils.h"
#include "radixSort.h"

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
    int64_t flops_per_split_expected;
    
    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    CIT nthreads;
    CIT nsplits;
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
            nsplits = std::min( nthreads*4, num_of_columns );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : num_of_columns;
            for(CIT i = colStart; i < colEnd; i++)
            {
                nz_per_column[i] = 0;

                size_t htSize = minHashTableSize;
                while(htSize < flops_per_column[i]) //htSize is set as 2^n
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
    //printf("[Symbolic Hash] Time for parallel section: %lf\n", t1-t0);
    //printf("[Symbolic Hash] Stats of parallel section timing:\n");
    //getStats<double>(ttimes, true);
    //printf("---\n");
    
    // parallel programming ended
    //for (int i = 0 ; i < nz_per_column.size(); i++){
        //fp << nz_per_column[i] << std::endl;
    //}
    //fp.close();
    return std::move(nz_per_column);
    
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashSliding(std::vector<CSC<RIT, VT, CPT>* > &matrices, const RIT windowSizeSymbolic, const RIT windowSize){
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = windowSize;
    const RIT maxHashTableSizeSymbolic = windowSizeSymbolic;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<size_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
            nWindowPerColSymbolic[i] = (flopsPerCol[i] / maxHashTableSizeSymbolic) + 1;
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<size_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    size_t cacheL1 = 32 * 1024;
    size_t elementsToFitL1 = cacheL1 / sizeof( std::pair<RIT,RIT> ); // 32KB L1 cache / 8B element size = 4096 elements needed to fit cache line
    size_t padding = std::max(elementsToFitL1, nmatrices); 
    pvector< std::pair< RIT, RIT > > rowIdsRange(padding * nthreads); // Padding to avoid false sharing

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        size_t tid = omp_get_thread_num();
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), s * flopsPerSplitExpected) - prefixSumSymbolic.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                nnzPerCol[i] = 0;
                nWindowPerCol[i] = 1;
                size_t nwindows = nWindowPerColSymbolic[i];
                if (nwindows == 1){
                    RIT rowStart = 0;
                    RIT  rowEnd = nrows;
                    size_t wIdx = prefixSumWindowSymbolic[i];

                    nnzPerWindowSymbolic[wIdx].first = 0;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    size_t flopsWindow = flopsPerCol[i];
                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }

                    for(size_t k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin() + (*colPtr)[i];
                        auto last = rowIds->begin() + (*colPtr)[i+1];
                        
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
                }
                else{
                    RIT nrowsPerWindow = nrows / nwindows;
                    RIT runningSum = 0;
                    for(size_t w = 0; w < nwindows; w++){
                        RIT rowStart = w * nrowsPerWindow;
                        RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;

                        int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                        nnzPerWindowSymbolic[wIdx].first = rowStart;
                        nnzPerWindowSymbolic[wIdx].second = 0;

                        size_t flopsWindow = 0;

                        for(int k = 0; k < nmatrices; k++){
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + (*colPtr)[i];
                            auto last = rowIds->begin() + (*colPtr)[i+1];
                            size_t startIdx, endIdx, midIdx;

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
                                //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
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
                                //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                            }
                            rowIdsRange[tid * padding + k].first = first - rowIds->begin();
                            rowIdsRange[tid * padding + k].second = last - rowIds->begin();

                            flopsWindow += last-first;
                        }

                        size_t htSize = minHashTableSize;
                        while(htSize < flopsWindow) //htSize is set as 2^n
                        {
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j){
                            globalHashVec[j] = -1;
                        }

                        for(int k = 0; k < nmatrices; k++)
                        {
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + rowIdsRange[tid  * padding + k].first;
                            auto last = rowIds->begin() + rowIdsRange[tid * padding + k].second;
                            
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
                            //nWindowPerCol[i] = 1;
                            runningSum = nnzPerWindowSymbolic[wIdx].second;
                        }
                        else{
                            if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                                nWindowPerCol[i]++;
                                runningSum = nnzPerWindowSymbolic[wIdx].second;
                            }
                            else{
                                runningSum = runningSum + nnzPerWindowSymbolic[wIdx].second;
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
    printf("[Sliding Hash]\tTime for symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of time consumed by threads:\n");
    getStats<double>(ttimes, true);
    printf("---\n");

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
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerSplitExpected;
    //auto nnzPerColStats = getStats<RIT>(nnzPerCol);
    CIT nthreads;
    CIT nsplits;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;

    pvector<double> colTimes(ncols);

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(minHashTableSize);
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols);
            ttimes.resize(nthreads);
            splitters.resize(nsplits);
            nnzCPerThreadExpected = nnzCTot / nthreads;
            nnzCPerSplitExpected = nnzCTot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * nnzCPerSplitExpected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic) 
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = s < nsplits-1 ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                //double tc = omp_get_wtime();
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
                        //integerSort<RIT>(globalHashVec.data(), index);
                    
                    
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
                //colTimes[i] = omp_get_wtime() - tc;
            }
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("[Hash]\tTime for parallel section: %lf\n", t1-t0);
    //printf("[Hash]\tStats for parallel section timing:\n");
    //getStats<double>(ttimes, true);
    //printf("***\n\n");

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    //double denseTime = 0;
    //CPT denseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] > densityThreshold){
            //denseTime += colTimes[i];
            //denseCount += 1;
        //}
    //}

    //double sparseTime = 0;
    //CPT sparseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] <= densityThreshold){
            //sparseTime += colTimes[i];
            //sparseCount += 1;
        //}
    //}
    
    //std::cout << denseCount << "," << denseTime << "," << sparseCount << "," << sparseTime << ",";

    clock.Stop();
    return std::move(sumMat);
}

/*
 * Sliding hash
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHashSliding(std::vector<CSC<RIT, VT, CPT>* > & matrices, const RIT windowSizeSymbolic, const RIT windowSize, bool sorted=true)
{
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = windowSize;
    const RIT maxHashTableSizeSymbolic = windowSizeSymbolic;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<size_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
            nWindowPerColSymbolic[i] = (flopsPerCol[i] / maxHashTableSizeSymbolic) + 1;
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<size_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    size_t cacheL1 = 32 * 1024;
    size_t elementsToFitL1 = cacheL1 / sizeof( std::pair<RIT,RIT> ); // 32KB L1 cache / 8B element size = 4096 elements needed to fit cache line
    size_t padding = std::max(elementsToFitL1, nmatrices); 
    pvector< std::pair< RIT, RIT > > rowIdsRange(padding * nthreads); // Padding to avoid false sharing

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        size_t tid = omp_get_thread_num();
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), s * flopsPerSplitExpected) - prefixSumSymbolic.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                nnzPerCol[i] = 0;
                nWindowPerCol[i] = 1;
                size_t nwindows = nWindowPerColSymbolic[i];
                if (nwindows == 1){
                    RIT rowStart = 0;
                    RIT  rowEnd = nrows;
                    size_t wIdx = prefixSumWindowSymbolic[i];

                    nnzPerWindowSymbolic[wIdx].first = 0;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    size_t flopsWindow = flopsPerCol[i];
                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }

                    for(size_t k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin() + (*colPtr)[i];
                        auto last = rowIds->begin() + (*colPtr)[i+1];
                        
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
                }
                else{
                    RIT nrowsPerWindow = nrows / nwindows;
                    RIT runningSum = 0;
                    for(size_t w = 0; w < nwindows; w++){
                        RIT rowStart = w * nrowsPerWindow;
                        RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;

                        int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                        nnzPerWindowSymbolic[wIdx].first = rowStart;
                        nnzPerWindowSymbolic[wIdx].second = 0;

                        size_t flopsWindow = 0;

                        for(int k = 0; k < nmatrices; k++){
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + (*colPtr)[i];
                            auto last = rowIds->begin() + (*colPtr)[i+1];
                            size_t startIdx, endIdx, midIdx;

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
                                //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
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
                                //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                            }
                            rowIdsRange[tid * padding + k].first = first - rowIds->begin();
                            rowIdsRange[tid * padding + k].second = last - rowIds->begin();

                            flopsWindow += last-first;
                        }

                        size_t htSize = minHashTableSize;
                        while(htSize < flopsWindow) //htSize is set as 2^n
                        {
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j){
                            globalHashVec[j] = -1;
                        }

                        for(int k = 0; k < nmatrices; k++)
                        {
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + rowIdsRange[tid  * padding + k].first;
                            auto last = rowIds->begin() + rowIdsRange[tid * padding + k].second;
                            
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
                            //nWindowPerCol[i] = 1;
                            runningSum = nnzPerWindowSymbolic[wIdx].second;
                        }
                        else{
                            if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                                nWindowPerCol[i]++;
                                runningSum = nnzPerWindowSymbolic[wIdx].second;
                            }
                            else{
                                runningSum = runningSum + nnzPerWindowSymbolic[wIdx].second;
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
    //printf("%lf,", t1-t0);
#ifdef DEBUG
    printf("[Sliding Hash]\tTime for symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of time consumed by threads:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("---\n");
    
    pvector<RIT> prefixSumWindow(ncols+1, 0);
    ParallelPrefixSum(nWindowPerCol, prefixSumWindow);

    //printf("[Sliding Hash]\tStats of number of windows:\n");
    //getStats<RIT>(nWindowPerCol, true);

    pvector< std::pair<RIT, RIT> > nnzPerWindow(prefixSumWindow[ncols]);
    
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
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
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;
    
    pvector<double> colTimes(ncols);
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSum.begin(), prefixSum.end(), s * nnzCPerSplitExpected) - prefixSum.begin();
        }
#pragma omp for schedule(dynamic)
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                //double tc = omp_get_wtime();
                RIT nwindows = nWindowPerCol[i];
                if(nwindows > 1){
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                        rowIdsRange[padding * tid + k].first = (*colPtr)[i];
                        rowIdsRange[padding * tid + k].second = (*colPtr)[i+1];
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

                            while( (rowIdsRange[padding * tid + k].first < rowIdsRange[padding * tid + k].second) && ((*rowIds)[rowIdsRange[padding * tid + k].first] < rowEnd) ){
                                //printf("Thread %d, Column %d, Window %d\n", tid, i, w);
                                RIT j = rowIdsRange[padding * tid + k].first;
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

                                rowIdsRange[padding * tid + k].first++;
                            } 
                        }

                        if(sorted){
                            size_t index = 0;
                            for (size_t j=0; j < htSize; ++j){
                                if (globalHashVec[j].first != -1){
                                    globalHashVec[index++] = globalHashVec[j];
                                }
                            }
                    
                            //std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                            integerSort<RIT>(globalHashVec.data(), index);
                        
                            for (size_t j=0; j < index; ++j){
                                CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                prefixSum[i] ++;
                            }
                        }
                        else{
                            for (size_t j=0; j < htSize; ++j)
                            {
                                if (globalHashVec[j].first != -1)
                                {
                                    CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                    CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                    prefixSum[i] ++;
                                }
                            }
                        }
                    }
                    
                }
                else{
                    RIT wIdx = prefixSumWindow[i];
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

                    for(size_t k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                        for( RIT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
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
                    if(sorted){
                        size_t index = 0;
                        for (size_t j=0; j < htSize; ++j){
                            if (globalHashVec[j].first != -1){
                                globalHashVec[index++] = globalHashVec[j];
                            }
                        }
                
                        //std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                        integerSort<RIT>(globalHashVec.data(), index);
                    
                        for (size_t j=0; j < index; ++j){
                            CrowIds[prefixSum[i]] = globalHashVec[j].first;
                            CnzVals[prefixSum[i]] = globalHashVec[j].second;
                            prefixSum[i] ++;
                        }
                    }
                    else{
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                prefixSum[i] ++;
                            }
                        }
                    }
                }
                //colTimes[i] = omp_get_wtime() - tc;
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("%lf,", t1-t0);

#ifdef DEBUG
    printf("[Sliding Hash]\tTime for computation: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of parallel section timings:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("***\n\n");
    
    //double denseTime = 0;
    //CPT denseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] > windowSize){
            //denseTime += colTimes[i];
            //denseCount += 1;
        //}
    //}

    //double sparseTime = 0;
    //CPT sparseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] <= windowSize){
            //sparseTime += colTimes[i];
            //sparseCount += 1;
        //}
    //}
    
    //std::cout << denseCount << "," << denseTime << "," << sparseCount << "," << sparseTime << ",";

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddSpA(std::vector<CSC<RIT, VT, CPT>* > & matrices)
{
    double t0, t1, t2, t3;

    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();

    pvector<CPT> nnzPerCol(ncols);
    pvector<CPT> flops_per_column(ncols, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < ncols; i++){
        for(int k = 0; k < nmatrices; k++){
            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[ncols];
    int64_t flops_per_thread_expected;
    int64_t flops_per_split_expected;
    
    CIT nthreads;
    CIT nsplits;
    pvector<CIT> splitters;
    pvector<double> ttimes;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector< RIT > globalHashVec(nrows);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min( nthreads*4, ncols );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CIT i = colStart; i < colEnd; i++){
                nnzPerCol[i] = 0;
                for(size_t j=0; j < nrows; ++j) globalHashVec[j] = -1;
            
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                
                    for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                    {
                        RIT key = (*rowIds)[j];
                        if (globalHashVec[key] == -1) {
                            globalHashVec[key] = key;
                            nnzPerCol[i]++;
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();

    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(nrows);
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * nnzCPerSplitExpected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic) 
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = s < nsplits-1 ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] != 0){
                    for(size_t j=0; j < nrows; ++j) globalHashVec[j].first = -1;
                
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                        {
                            RIT key = (*rowIds)[j];
                            VT curval = (*nzVals)[j];
                            if (globalHashVec[key].first == key) //key is found in hash table
                            {
                                globalHashVec[key].second += curval;
                            }
                            else if (globalHashVec[key].first == -1) //key is not registered yet
                            {
                                globalHashVec[key].first = key;
                                globalHashVec[key].second = curval;
                            }
                        }
                    }
               
                    for (size_t j=0; j < nrows; ++j){
                        if (globalHashVec[j].first != -1){
                            CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                            CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                            prefix_sum[i]++;
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

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

/*
 *  Estimates nnumber of non-zeroes when adding two CSC matrices in regular way
 *  Assumes that entries of each column are sorted according to the order of row id
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
pvector<RIT> symbolicSpAddRegular(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B){
    double t0, t1, t3, t4;

    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();
    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();
    
    pvector<RIT> nnzCPerCol(ncols);
    
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
//template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
//CSC<RIT, VT, CPT> SpMultiAdd(std::vector<CSC<RIT, VT, CPT>* > & matrices, int version, bool inputSorted=true, bool outputSorted=true){

    //int nmatrices = matrices.size();
    
    //if(nmatrices == 0) return CSC<RIT, VT, CPT>();
    //double t0, t1, t3, t4;

    //CIT ncols = matrices[0]->get_ncols();
    //RIT nrows = matrices[0]->get_nrows();
    
    //for(int i = 1; i < nmatrices; i++)
    //{
        //if( (ncols != matrices[i]->get_ncols()) || (nrows != matrices[i]->get_nrows()))
        //{
            //std::cerr << " Can not be added as matrix dimensions do not agree. Returning an empty matrix. \n";
            //return CSC<RIT, VT, CPT>();
        //}
    //}

    //if(version != 4) {
        //t0 = omp_get_wtime();
        //pvector<RIT> nnzCPerCol = symbolicSpMultiAddHash<RIT, CIT, VT, CPT, int32_t>(matrices);
        //t1 = omp_get_wtime();
        ////printf("Time for symbolic: %lf\n", t1-t0);
        //if(version == 0) return SpMultiAddHash<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        //else if(version == 1) return SpMultiAddHybrid<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        //else if(version == 2) return SpMultiAddHybrid2<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        //else if(version == 3) return SpMultiAddHybrid3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
        //else return SpMultiAddHybrid3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
    //}
    //else {
        //return SpMultiAddHybrid4<RIT, CIT, VT, CPT>(matrices, true);
    //}
    
    ////t0 = omp_get_wtime();
    ////pvector<RIT> nnzCPerCol1 = symbolicSpMultiAddHashSliding<RIT, CIT, VT, CPT, int32_t>(matrices);
    ////t1 = omp_get_wtime();
    ////printf("Time for symbolic sliding new: %lf\n", t1-t0);
    
    ////t0 = omp_get_wtime();
    ////pvector<RIT> nnzCPerCol2 = symbolicSpMultiAddHashSliding1<RIT, CIT, VT, CPT, int32_t>(matrices);
    ////t1 = omp_get_wtime();
    ////printf("Time for symbolic sliding old: %lf\n", t1-t0);
    
    ////for(CIT i=0; i< ncols; i++)
    ////{
        ////if(nnzCPerCol[i] != nnzCPerCol1[i]) std::cout << "not equal" << std::endl;
    ////}
    ////printf("Symbolic Equal!\n");
    
    
    ////return SpMultiAddSpA3<RIT, CIT, VT, CPT>(matrices, nnzCPerCol, true);
//}

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


template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
CSC<RIT, VT, CPT> SpMultiAddHeap(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices){
    double t0, t1, t2, t3;
	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();
    NM number_of_matrices = vec_of_matrices.size();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

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

    pvector<CPT> prefix_sum(num_of_columns+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[num_of_columns];
    int64_t flops_per_thread_expected;
    int64_t flops_per_split_expected;
    
    CIT nthreads;
    CIT nsplits;
    pvector<CIT> splitters;
    pvector<double> ttimes;
	pvector<RIT> nz_per_column(num_of_columns, 0); 
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::priority_queue<heap_help<RIT, VT, NM> > pq;
        pvector<RIT> current_index_of_specific_matrix(number_of_matrices);
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min( nthreads*4, num_of_columns );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : num_of_columns;
            for(CIT i = colStart; i < colEnd; i++){
                for(NM j = 0; j < number_of_matrices; j++){
                    current_index_of_specific_matrix[j] = 0;
                    const pvector<CPT> *col_ptr_j = vec_of_matrices[j]->get_colPtr();
                    const pvector<RIT> *row_ids_j = vec_of_matrices[j]->get_rowIds();
                    const pvector<VT> *nz_j = vec_of_matrices[j]->get_nzVals();

                    if( (*col_ptr_j)[i] + current_index_of_specific_matrix[j] < (*col_ptr_j)[i+1] ){
                        RIT temp1_row = (*row_ids_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
                        VT temp1_value = (*nz_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
                        pq.push(heap_help<RIT, VT, NM> (temp1_row, temp1_value, j));
                        current_index_of_specific_matrix[j]++;
                    }
                }

                bool flag = false;
                while(!pq.empty()){
                    nz_per_column[i]++;
                    RIT r = pq.top().hh_row_number;
                    while(!pq.empty() && r == pq.top().hh_row_number){
                        heap_help<RIT, VT, NM> elem = pq.top();
                        pq.pop();
                        NM j = elem.hh_matrix_number;
                        current_index_of_specific_matrix[j]++;

                        const pvector<CPT> *col_ptr_j = vec_of_matrices[j]->get_colPtr();
                        const pvector<RIT> *row_ids_j = vec_of_matrices[j]->get_rowIds();
                        const pvector<VT> *nz_j = vec_of_matrices[j]->get_nzVals();

                        if( (*col_ptr_j)[i] + current_index_of_specific_matrix[j] < (*col_ptr_j)[i+1] ){
                            RIT temp1_row = (*row_ids_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
                            VT temp1_value = (*nz_j)[(*col_ptr_j)[i] + current_index_of_specific_matrix[j]] ;
                            pq.push(heap_help<RIT, VT, NM> (temp1_row, temp1_value, j));
                        }
                    }
                }
            }
        }
    }


	ParallelPrefixSum(nz_per_column, prefix_sum);
	pvector<CPT> column_vector_for_csc(prefix_sum.begin(), prefix_sum.end());
	pvector<VT> value_vector_for_csc(prefix_sum[num_of_columns]);
	pvector<RIT> row_vector_for_csc(prefix_sum[num_of_columns]);

    int64_t nnz_tot = prefix_sum[num_of_columns];
    int64_t nnz_per_thread_expected = nnz_tot / nthreads;
    int64_t nnz_per_split_expected = nnz_tot / nsplits;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::priority_queue<heap_help<RIT, VT, NM> > pq;
        pvector<RIT> current_index_of_specific_matrix(number_of_matrices);
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : num_of_columns;
            for(CIT i = colStart; i < colEnd; i++){
                for(NM j = 0; j < number_of_matrices; j++){
                    current_index_of_specific_matrix[j] = 0;
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
                }

                bool first_entry_check = false;
                while(! pq.empty()){
                    heap_help<RIT, VT, NM> temp1 = pq.top(); // T as long double
                    pq.pop();
                    if(!first_entry_check){
                        value_vector_for_csc[prefix_sum[i]] = temp1.hh_element;
                        row_vector_for_csc[prefix_sum[i]] = temp1.hh_row_number;
                        prefix_sum[i]++;
                        first_entry_check = true;
                    }
                    else{
                        if(row_vector_for_csc[prefix_sum[i] - 1] == temp1.hh_row_number){
                            value_vector_for_csc[prefix_sum[i] - 1] += temp1.hh_element;
                        }
                        else{
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

                } //end while
            }
        }
    }


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
