#ifndef _CSR_H_
#define _CSR_H_

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <random>
#include <vector>


#include "COO.h"
#include "utils.h"
#include "GAP/timer.h"
#include "GAP/util.h"
#include "GAP/pvector.h"
#include "GAP/platform_atomics.h"

#include <numeric>
//#include "gtest/gtest.h"



/*
 We heavily use pvector (equivalent to std::vector).
Since pvector uses size_t for indexing, we will also stick to  size_t for indexing. That means, nnz, nrows, ncols are all of size_t.
What should we do for value type for array of offsets such as colptr?
size_t is definitely safe because of the above argument.
However, it may take more space for tall and skinny matrices.
Hence, we use an optional type RPT for such cases. By default, RPT=size_t
*/



// CIT: Row Index Type
// VT: Value Type
// RPT: Row pointer type 
template <typename CIT, typename VT=double, typename RPT=size_t>
class CSR
{
public:
	CSR(): nrows_(0), ncols_(0), nnz_(0), isRowSorted_(false) {}

	CSR(size_t nrows, CIT ncols, size_t nnz,bool row_sort_bool, bool isWeighted): nrows_(nrows), ncols_(ncols), nnz_(nnz), isRowSorted_(row_sort_bool), isWeighted_(isWeighted) 
    {
		rowPtr_.resize(nrows_+1); 
        colIds_.resize(nnz); 
        nzVals_.resize(nnz);
    }  // added by abhishek

	//written by Shardul
	template <typename RIT>
	CSR(COO<RIT, CIT, VT> & cooMat);
	
	template <typename AddOp>
	void MergeDuplicateSort(AddOp binop);
	void PrintInfo();

	const pvector<CIT>* get_colIds(); // added by abhishek
	const pvector<RPT>* get_rowPtr(); 
	const pvector<VT>* get_nzVals();
    
    const RPT get_rowPtr(size_t idx);

	//Copy constructor, written by Shardul
	CSR<CIT, VT, RPT>(CSR<CIT, VT, RPT> &&other): nrows_(other.nrows_),ncols_(other.ncols_),nnz_(other.nnz_),isWeighted_(other.isWeighted_),isRowSorted_(other.isRowSorted_)   // added by abhishek
	{
		rowPtr_.resize(nrows_+1); colIds_.resize(nnz_); nzVals_.resize(nnz_);
		colIds_ = std::move(other.colIds_);
		rowPtr_ = std::move(other.rowPtr_);
		nzVals_ = std::move(other.nzVals_);
	}

	//Overloaded equal to operator
	CSR<CIT, VT, RPT>& operator= (CSR<CIT, VT, RPT> && other){ // added by abhishek
		nrows_ = other.nrows_;
		ncols_ = other.ncols_;
		nnz_ = other.nnz_;
		isWeighted_ = other.isWeighted_;
		isRowSorted_ = other.isRowSorted_;
		rowPtr_.resize(nrows_+1); colIds_.resize(nnz_); nzVals_.resize(nnz_);
		colIds_ = std::move(other.colIds_);
		rowPtr_ = std::move(other.rowPtr_);
		nzVals_ = std::move(other.nzVals_);
		return *this;
	}
    
    bool operator== (const CSR<CIT, VT, RPT> & other);


	size_t get_ncols(); // added by abhishek
	size_t get_nrows();
	size_t get_nnz() ;

	void rows_pvector(pvector<RPT>* row_pointer) { rowPtr_ = std::move(*(row_pointer));} // added by abhishek
	void nz_cols_pvector(pvector<CIT>* column_pointer) {colIds_ = std::move(*(column_pointer));}
	void nz_vals_pvector(pvector<VT>* value_pointer) {nzVals_ = std::move(*(value_pointer));}

	

	void print_all(); // added by abhishek
    
	
	// written by Shardul
	void ewiseApply(VT scalar);


	
	// written by Shardul
	template <typename T>
	void dimApply(pvector<T> &mul_vector);


	// written by Shardul
	void row_reduce();

	

private:
	size_t nrows_;
	size_t ncols_;
	size_t nnz_;
 
	pvector<CIT> colIds_;
	pvector<RPT> rowPtr_;
	pvector<VT> nzVals_;
	bool isWeighted_;
	bool isRowSorted_;
};





template <typename CIT, typename VT, typename RPT>
const pvector<RPT>* CSR<CIT, VT, RPT>::get_rowPtr()
{
	return &rowPtr_;
}

template <typename CIT, typename VT, typename RPT>
const RPT CSR<CIT, VT, RPT>::get_rowPtr(size_t idx)
{
    return rowPtr_[idx];
}

template <typename CIT, typename VT, typename RPT>
const pvector<CIT>* CSR<CIT, VT, RPT>::get_colIds()
{
	return &colIds_;
}

template <typename RIT, typename VT, typename RPT>
const pvector<VT>* CSR<RIT, VT, RPT>::get_nzVals()
{
	return &nzVals_;
} 



template <typename RIT, typename VT, typename RPT>
size_t CSR<RIT, VT, RPT>:: get_ncols()
{
	return ncols_;
}

template <typename RIT, typename VT, typename RPT>
size_t CSR<RIT, VT, RPT>:: get_nrows()
{
	return nrows_;
}

template <typename RIT, typename VT, typename RPT>
size_t CSR<RIT, VT, RPT>:: get_nnz()
{
	return nnz_;
}

template <typename RIT, typename VT, typename RPT>
void CSR<RIT, VT, RPT>::print_all()
{
	//std::cout << "CSR matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl<<"column_pointer_array"<<std::endl;
	std::cout<< nrows_<<" "<<ncols_<<" "<<nnz_<<std::endl;
	
	for(size_t i = 0; i < colIds_.size(); i++){
		std::cout<<colIds_[i];
		if(i != ncols_){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
	for(size_t i = 0; i < rowPtr_.size(); i++){
		std::cout<<rowPtr_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	

}



//TODO: need parallel code
template <typename CIT, typename VT, typename RPT>
bool CSR<CIT, VT, RPT>::operator==(const CSR<CIT, VT, RPT> & rhs)
{
    if(nnz_ != rhs.nnz_ || nrows_  != rhs.nrows_ || ncols_ != rhs.ncols_) return false;
    bool same = std::equal(rowPtr_.begin(), rowPtr_.begin()+nrows_+1, rhs.rowPtr_.begin());
    same = same && std::equal(colIds_.begin(), colIds_.begin()+nnz_, rhs.colIds_.begin());
    ErrorTolerantEqual<VT> epsilonequal(EPSILON);
    same = same && std::equal(nzVals_.begin(), nzVals_.begin()+nnz_, rhs.nzVals_.begin(), epsilonequal );
    return same;
}

template <typename RIT, typename VT, typename RPT>
void CSR<RIT, VT, RPT>::PrintInfo()
{
	std::cout << "CSR matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl;
}

// Construct an CSR object from COO
// This will be a widely used function 
// Optimize this as much as possible
template <typename CIT, typename VT, typename RPT>
template <typename RIT>
CSR<CIT, VT, RPT>::CSR(COO<RIT, CIT, VT> & cooMat)
{
	Timer t;
	t.Start();
	nrows_ = cooMat.nrows();
	ncols_ = cooMat.ncols();
	nnz_ = cooMat.nnz();
	isWeighted_ = cooMat.isWeighted();
	cooMat.BinByRow(rowPtr_, colIds_, nzVals_);
	MergeDuplicateSort(std::plus<VT>());
	isRowSorted_ = true;
	t.Stop();
}


template <typename CIT, typename VT, typename RPT>
void CSR<CIT, VT, RPT>::ewiseApply(VT scalar)
{
	std::cout<<"Scalar"<<scalar<<std::endl;
	std::cout<<"\n"<<std::endl;
	std::cout<<"Rows, columns and non zero values"<<std::endl;
	std::cout<< nrows_<<" "<<ncols_<<" "<<nnz_<<std::endl;
	std::cout<<"nzvals"<<nzVals_.size()<<std::endl;


	for(size_t i = 0; i < nzVals_.size(); i++){
		nzVals_[i]=nzVals_[i]*scalar;
		
	}

	std::cout<<"Nonzero values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}

}


template <typename CIT, typename VT, typename RPT>
void CSR<CIT, VT, RPT>::row_reduce()
{
	
	size_t n=get_nrows();
	std::cout<<"n:"<<n<<std::endl;
	pvector<float> result_vector(n);
	std::cout<<"rowIds here"<<std::endl;
	std::cout<<"result_vector size here"<<result_vector.size()<<std::endl;
	for(size_t i = 0; i < n; i++)
	{
		std::cout<<rowPtr_[i]<<std::endl;
		result_vector[i]=0;
	}
	std::cout<<"Indices before rowreduce"<<std::endl;
	for(size_t i = 0; i < rowPtr_.size()-1; i++)
	{
		std::cout<<"i:"<<i<<std::endl;
		for(size_t j=rowPtr_[i];j<rowPtr_[i+1];j++)
		{
			//std::cout<<"j:"<<i<<std::endl;
			result_vector[i]=result_vector[i]+nzVals_[j];
			
		}
	}
	std::cout<<"Final Result"<<std::endl;
	//std::cout<<"rowid size: "<<rowPtr_.size()<<std::endl;
	for(size_t i = 0; i < rowPtr_.size()-1; i++)
	{
		std::cout<<result_vector[i]<<std::endl;
	}
	
}

//Multiply each non zero value with a vector
template <typename CIT, typename VT, typename RPT>
template<typename T>
void CSR<CIT, VT, RPT>::dimApply(pvector<T> &mul_vector)
{
	for(size_t i = 0; i < colIds_.size(); i++)
	{
		
		for(size_t j=colIds_[i];j<colIds_[i+1];j++)
		{
			nzVals_[j]=nzVals_[j]*mul_vector[i];
		}
	}
	std::cout<<"Nonzero values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
}







template <typename CIT, typename VT, typename RPT>
template <typename AddOp>
void CSR<CIT, VT, RPT>::MergeDuplicateSort(AddOp binop)
{
	pvector<CIT> sqNnzPerRow(nrows_);
#pragma omp parallel
	{
		pvector<std::pair<CIT, VT>> tosort;
#pragma omp for
        for(size_t i=0; i<nrows_; i++)
        {
            size_t nnzRow = rowPtr_[i+1]-rowPtr_[i];
            sqNnzPerRow[i] = 0;
            
            if(nnzRow>0)
            {
                if(tosort.size() < nnzRow) tosort.resize(nnzRow);
                
                for(size_t j=0, k=rowPtr_[i]; j<nnzRow; ++j, ++k)
                {
                    tosort[j] = std::make_pair(colIds_[k], nzVals_[k]);
                }
                
                //TODO: replace with radix or another integer sorting
                sort(tosort.begin(), tosort.begin()+nnzRow);
                
                size_t k = rowPtr_[i];
                colIds_[k] = tosort[0].first;
                nzVals_[k] = tosort[0].second;
                
                // k points to last updated entry
                for(size_t j=1; j<nnzRow; ++j)
                {
                    if(tosort[j].first != colIds_[k])
                    {
                        colIds_[++k] = tosort[j].first;
                        nzVals_[k] = tosort[j].second;
                    }
                    else
                    {
                        nzVals_[k] = binop(tosort[j].second, nzVals_[k]);
                    }
                }
                sqNnzPerRow[i] = k-rowPtr_[i]+1;
          
            }
        }
    }
    
    
    // now squeze
    // need another set of arrays
    // Think: can we avoid this extra copy with a symbolic step?
    pvector<RPT>sqRowPtr;
    ParallelPrefixSum(sqNnzPerRow, sqRowPtr);
    nnz_ = sqRowPtr[nrows_];
    pvector<CIT> sqColIds(nnz_);
    pvector<VT> sqNzVals(nnz_);

#pragma omp parallel for
	for(size_t i=0; i<nrows_; i++)
	{
		size_t srcStart = rowPtr_[i];
		size_t srcEnd = rowPtr_[i] + sqNnzPerRow[i];
		size_t destStart = sqRowPtr[i];
		std::copy(colIds_.begin()+srcStart, colIds_.begin()+srcEnd, sqColIds.begin()+destStart);
		std::copy(nzVals_.begin()+srcStart, nzVals_.begin()+srcEnd, sqNzVals.begin()+destStart);
	}
	
	// now replace (just pointer swap)
	
	rowPtr_.swap(sqRowPtr);
	colIds_.swap(sqColIds);
	nzVals_.swap(sqNzVals);
	
}

#endif