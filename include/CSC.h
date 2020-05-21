#ifndef _CSC_H_
#define _CSC_H_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <random>


#include "COO.h"
#include "utils.h"
#include "GAP/timer.h"
#include "GAP/util.h"
#include "GAP/pvector.h"
#include "GAP/platform_atomics.h"

#include <numeric>

/*
 We heavily use pvector (equivalent to std::vector).
Since pvector uses size_t for indexing, we will also stick to  size_t for indexing. That means, nnz, nrows, ncols are all of size_t.
What should we do for value type for array of offsets such as colptr?
size_t is definitely safe because of the above argument.
However, it may take more space for tall and skinny matrices.
Hence, we use an optional type CPT for such cases. By default, CPT=size_t
*/



// RIT: Row Index Type
// VT: Value Type
// CPT: Column pointer type (use only for very tall and skinny matrices)
template <typename RIT, typename VT=double, typename CPT=size_t>
class CSC
{
public:
    CSC(): nrows_(0), ncols_(0), nnz_(0), isColSorted_(false) {}
    //COO(nrows, ncols, nnz, isWeighted): nrows_(nrows), ncols_(ncols), nnz_(nnz), sort_type_(UNSORTED), isWeighted_(isWeighted); {NzList.resize(nnz_);}
    template <typename CIT>
    CSC(COO<RIT, CIT, VT> & cooMat);
    
    template <typename AddOp>
    void MergeDuplicateSort(AddOp binop);
    void PrintInfo();
    
private:
    size_t nrows_;
    size_t ncols_;
    size_t nnz_;
 
    pvector<CPT> colPtr_;
    pvector<RIT> rowIds_;
    pvector<VT> nzVals_;
    bool isWeighted_;
    bool isColSorted_;
};


template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::PrintInfo()
{
    std::cout << "CSC matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl;
}
// Construct an CSC object from COO
// This will be a widely used function
// Optimize this as much as possible
template <typename RIT, typename VT, typename CPT>
template <typename CIT>
CSC<RIT, VT, CPT>::CSC(COO<RIT, CIT, VT> & cooMat)
{
    Timer t;
    t.Start();
    nrows_ = cooMat.nrows();
    ncols_ = cooMat.ncols();
    nnz_ = cooMat.nnz();
    isWeighted_ = cooMat.isWeighted();
    cooMat.BinByCol(colPtr_, rowIds_, nzVals_);
    MergeDuplicateSort(std::plus<VT>());
    isColSorted_ = true;
    t.Stop();
    PrintTime("CSC Creation Time", t.Seconds());
}



template <typename RIT, typename VT, typename CPT>
template <typename AddOp>
void CSC<RIT, VT, CPT>::MergeDuplicateSort(AddOp binop)
{
    pvector<RIT> sqNnzPerCol(ncols_);
#pragma omp parallel
    {
        pvector<std::pair<RIT, VT>> tosort;
#pragma omp for
        for(size_t i=0; i<ncols_; i++)
        {
            size_t nnzCol = colPtr_[i+1]-colPtr_[i];
            sqNnzPerCol[i] = 0;
            
            if(nnzCol>0)
            {
                if(tosort.size() < nnzCol) tosort.resize(nnzCol);
                
                for(size_t j=0, k=colPtr_[i]; j<nnzCol; ++j, ++k)
                {
                    tosort[j] = std::make_pair(rowIds_[k], nzVals_[k]);
                }
                
                //TODO: replace with radix or another integer sorting
                sort(tosort.begin(), tosort.end());
                
                size_t k = colPtr_[i];
                rowIds_[k] = tosort[0].first;
                nzVals_[k] = tosort[0].second;
                
                // k points to last updated entry
                for(size_t j=1; j<nnzCol; ++j)
                {
                    if(tosort[j].first != rowIds_[k])
                    {
                        rowIds_[++k] = tosort[j].first;
                        nzVals_[k] = tosort[j].second;
                    }
                    else
                    {
                        nzVals_[k] = binop(tosort[j].second, nzVals_[k]);
                    }
                }
                sqNnzPerCol[i] = k-colPtr_[i]+1;
          
            }
        }
    }
    
    
    // now squeze
    // need another set of arrays
    // Think: can we avoid this extra copy with a symbolic step?
    pvector<CPT>sqColPtr;
    ParallelPrefixSum(sqNnzPerCol, sqColPtr);
    nnz_ = sqColPtr[ncols_];
    pvector<RIT> sqRowIds(nnz_);
    pvector<VT> sqNzVals(nnz_);
#pragma omp parallel for
    for(size_t i=0; i<ncols_; i++)
    {
        size_t srcStart = colPtr_[i];
        size_t srcEnd = colPtr_[i] + sqNnzPerCol[i];
        size_t destStart = sqColPtr[i];
        std::copy(rowIds_.begin()+srcStart, rowIds_.begin()+srcEnd, sqRowIds.begin()+destStart);
        std::copy(nzVals_.begin()+srcStart, nzVals_.begin()+srcEnd, sqNzVals.begin()+destStart);
    }
    
    // now replace (just pointer swap)
    colPtr_.swap(sqColPtr);
    rowIds_.swap(sqRowIds);
    nzVals_.swap(sqNzVals);
    
}



#endif
