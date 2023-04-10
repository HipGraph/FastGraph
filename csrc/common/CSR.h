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
Hence, we use an optional type CPT for such cases. By default, CPT=size_t
*/



// RIT: Row Index Type
// VT: Value Type
// CPT: Column pointer type (use only for very tall and skinny matrices)
template <typename RIT, typename VT=double, typename CPT=size_t>
class CSR
{
public:
	CSR(): nrows_(0), ncols_(0), nnz_(0), isColSorted_(false) {}

	CSR(RIT nrows, size_t ncols, size_t nnz,bool col_sort_bool, bool isWeighted): nrows_(nrows), ncols_(ncols), nnz_(nnz), isColSorted_(col_sort_bool), isWeighted_(isWeighted) 
    {
		rowPtr_.resize(nrows_+1); 
        colIds_.resize(nnz); 
        nzVals_.resize(nnz);
    }  // added by abhishek

	template <typename CIT>
	CSR(COO<RIT, CIT, VT> & cooMat);
	
	template <typename AddOp>
	void MergeDuplicateSort(AddOp binop);
	void PrintInfo();

	const pvector<CPT>* get_colPtr(); // added by abhishek
	const pvector<RIT>* get_rowIds(); 
	const pvector<VT>* get_nzVals();
    
    const CPT get_colPtr(size_t idx);

	CSR<RIT, VT, CPT>(CSR<RIT, VT, CPT> &&other): nrows_(other.nrows_),ncols_(other.ncols_),nnz_(other.nnz_),isWeighted_(other.isWeighted_),isColSorted_(other.isColSorted_)   // added by abhishek
	{
		rowPtr_.resize(nnz_); colIds_.resize(ncols_+1); nzVals_.resize(nnz_);
		colIds_ = std::move(other.colIds_);
		rowPtr_ = std::move(other.rowPtr_);
		nzVals_ = std::move(other.nzVals_);
	}

	CSR<RIT, VT, CPT>& operator= (CSR<RIT, VT, CPT> && other){ // added by abhishek
		nrows_ = other.nrows_;
		ncols_ = other.ncols_;
		nnz_ = other.nnz_;
		isWeighted_ = other.isWeighted_;
		isColSorted_ = other.isColSorted_;
		rowPtr_.resize(nnz_); colIds_.resize(ncols_+1); nzVals_.resize(nnz_);
		colIds_ = std::move(other.colIds_);
		rowPtr_ = std::move(other.rowPtr_);
		nzVals_ = std::move(other.nzVals_);
		return *this;
	}
    
    bool operator== (const CSR<RIT, VT, CPT> & other);


	size_t get_ncols(); // added by abhishek
	size_t get_nrows();
	size_t get_nnz() ;

	void nz_rows_pvector(pvector<RIT>* row_pointer) { rowPtr_ = std::move(*(row_pointer));} // added by abhishek
	void cols_pvector(pvector<CPT>* column_pointer) {colIds_ = std::move(*(column_pointer));}
	void nz_vals_pvector(pvector<VT>* value_pointer) {nzVals_ = std::move(*(value_pointer));}

	void count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon); // added by abhishek
	void sort_inside_column();

	void print_all(); // added by abhishek
    
	
	void ewiseApply(VT scalar);
	//void ewiseApply1(VT scalar);


	// template <typename T>
	// void dimApply(std::vector<T> mul_vector);

	template <typename T>
	void dimApply(pvector<T> &mul_vector);

	//template <typename T>
	void row_reduce();

	template <typename T>
	pvector<T> row_reduce_1();

	
	void matAddition(CSR &b);

	void matAddition_1(CSR &b);

	void matAddition_2(CSR &b);

	void matAddition_3(CSR &b);


    void column_split(std::vector< CSR<RIT, VT, CPT>* > &vec, int nsplit);
private:
	size_t nrows_;
	size_t ncols_;
	size_t nnz_;
 
	pvector<CPT> colIds_;
	pvector<RIT> rowPtr_;
	pvector<VT> nzVals_;
	bool isWeighted_;
	bool isColSorted_;
};





template <typename RIT, typename VT, typename CPT>
const pvector<CPT>* CSR<RIT, VT, CPT>::get_colPtr()
{
	return &colIds_;
}

template <typename RIT, typename VT, typename CPT>
const CPT CSR<RIT, VT, CPT>::get_colPtr(size_t idx)
{
    return colIds_[idx];
}

template <typename RIT, typename VT, typename CPT>
const pvector<RIT>*  CSR<RIT, VT, CPT>::get_rowIds()
{
	return &rowPtr_;
}

template <typename RIT, typename VT, typename CPT>
const pvector<VT>* CSR<RIT, VT, CPT>::get_nzVals()
{
	return &nzVals_;
} 



template <typename RIT, typename VT, typename CPT>
size_t CSR<RIT, VT, CPT>:: get_ncols()
{
	return ncols_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSR<RIT, VT, CPT>:: get_nrows()
{
	return nrows_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSR<RIT, VT, CPT>:: get_nnz()
{
	return nnz_;
}

template <typename RIT, typename VT, typename CPT>
void CSR<RIT, VT, CPT>::print_all()
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
	//std::cout<<std::endl;
	//std::cout<<std::endl<<"row_correspondents"<<std::endl;
	for(size_t i = 0; i < rowPtr_.size(); i++){
		std::cout<<rowPtr_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	//std::cout<<std::endl;
	//std::cout<<std::endl<<"nz_values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
	//std::cout<<std::endl;

}



//TODO: need parallel code
// template <typename RIT, typename VT, typename CPT>
// bool CSR<RIT, VT, CPT>::operator==(const CSR<RIT, VT, CPT> & rhs)
// {
//     if(nnz_ != rhs.nnz_ || nrows_  != rhs.nrows_ || ncols_ != rhs.ncols_) return false;
//     bool same = std::equal(colIds_.begin(), colIds_.begin()+ncols_+1, rhs.colIds_.begin());
//     same = same && std::equal(rowPtr_.begin(), rowPtr_.begin()+nnz_, rhs.rowPtr_.begin());
//     ErrorTolerantEqual<VT> epsilonequal(EPSILON);
//     same = same && std::equal(nzVals_.begin(), nzVals_.begin()+nnz_, rhs.nzVals_.begin(), epsilonequal );
//     return same;
// }

template <typename RIT, typename VT, typename CPT>
void CSR<RIT, VT, CPT>::PrintInfo()
{
	std::cout << "CSR matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl;
}
// Construct an CSR object from COO
// This will be a widely used function
// Optimize this as much as possible
template <typename RIT, typename VT, typename CPT>
template <typename CIT>
CSR<RIT, VT, CPT>::CSR(COO<RIT, CIT, VT> & cooMat)
{
	Timer t;
	t.Start();
	nrows_ = cooMat.nrows();
	ncols_ = cooMat.ncols();
	nnz_ = cooMat.nnz();
	isWeighted_ = cooMat.isWeighted();
	cooMat.BinByCol(colIds_, rowPtr_, nzVals_);
	MergeDuplicateSort(std::plus<VT>());
	isColSorted_ = true;
	t.Stop();
	//PrintTime("CSR Creation Time", t.Seconds());
}


template <typename RIT, typename VT, typename CPT>
void CSR<RIT, VT, CPT>::ewiseApply(VT scalar)
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


template <typename RIT, typename VT, typename CPT>
void CSR<RIT, VT, CPT>::row_reduce()
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
	std::cout<<"rowid size: "<<rowPtr_.size()<<std::endl;
	for(size_t i = 0; i < rowPtr_.size()-1; i++)
	{
		std::cout<<result_vector[i]<<std::endl;
	}
	
}

//Multiply each non zero value with a vector
template <typename RIT, typename VT, typename CPT>
template<typename T>
void CSR<RIT, VT, CPT>::dimApply(pvector<T> &mul_vector)
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







template <typename RIT, typename VT, typename CPT>
template <typename AddOp>
void CSR<RIT, VT, CPT>::MergeDuplicateSort(AddOp binop)
{
	pvector<RIT> sqNnzPerCol(ncols_);
#pragma omp parallel
	{
		pvector<std::pair<RIT, VT>> tosort;
#pragma omp for
        for(size_t i=0; i<ncols_; i++)
        {
            size_t nnzCol = colIds_[i+1]-colIds_[i];
            sqNnzPerCol[i] = 0;
            
            if(nnzCol>0)
            {
                if(tosort.size() < nnzCol) tosort.resize(nnzCol);
                
                for(size_t j=0, k=colIds_[i]; j<nnzCol; ++j, ++k)
                {
                    tosort[j] = std::make_pair(rowPtr_[k], nzVals_[k]);
                }
                
                //TODO: replace with radix or another integer sorting
                sort(tosort.begin(), tosort.begin()+nnzCol);
                
                size_t k = colIds_[i];
                rowPtr_[k] = tosort[0].first;
                nzVals_[k] = tosort[0].second;
                
                // k points to last updated entry
                for(size_t j=1; j<nnzCol; ++j)
                {
                    if(tosort[j].first != rowPtr_[k])
                    {
                        rowPtr_[++k] = tosort[j].first;
                        nzVals_[k] = tosort[j].second;
                    }
                    else
                    {
                        nzVals_[k] = binop(tosort[j].second, nzVals_[k]);
                    }
                }
                sqNnzPerCol[i] = k-colIds_[i]+1;
          
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
		size_t srcStart = colIds_[i];
		size_t srcEnd = colIds_[i] + sqNnzPerCol[i];
		size_t destStart = sqColPtr[i];
		std::copy(rowPtr_.begin()+srcStart, rowPtr_.begin()+srcEnd, sqRowIds.begin()+destStart);
		std::copy(nzVals_.begin()+srcStart, nzVals_.begin()+srcEnd, sqNzVals.begin()+destStart);
	}
	
	// now replace (just pointer swap)
	colIds_.swap(sqColPtr);
	rowPtr_.swap(sqRowIds);
	nzVals_.swap(sqNzVals);
	
}


// template <typename RIT, typename VT, typename CPT> 
// void CSR<RIT, VT, CPT>::count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon)
// {


// 	size_t num_of_elements = all_elements.size();

// 	pvector<std::pair<RIT, VT> > temp_array(num_of_elements);
// 	RIT count[10] = {0};
// 	size_t index_for_count;

// 	for(size_t i = 0; i < num_of_elements; i++){
// 		index_for_count = ((all_elements[i].first)/expon)%10;
// 		count[index_for_count]++;
// 	}

// 	for(int i = 1; i < 10; i++){
// 		count[i] += count[i-1];
// 	}

// 	for(size_t i = num_of_elements-1; i > 0; i--){
// 		index_for_count = ((all_elements[i].first)/expon)%10;
// 		temp_array[count[index_for_count] -1] = all_elements[i];
// 		count[index_for_count]--;
// 	}


// 	index_for_count = ((all_elements[0].first)/expon)%10;
// 	temp_array[count[index_for_count] -1] = all_elements[0];

// 	all_elements = std::move(temp_array);

// 	return;
// }




// // here rowIds vector is to be sorted piece by piece given that there are no row repititions in a single column or any zero element in nzVals
// template <typename RIT, typename VT, typename CPT>
// void CSR<RIT, VT, CPT>::sort_inside_column()
// {
// 	if(!isColSorted_)
// 	{
// 		#pragma omp parallel for
// 				for(size_t i=0; i<ncols_; i++)
// 				{
// 					size_t nnzCol = colIds_[i+1]-colIds_[i];
					
// 					pvector<std::pair<RIT, VT>> tosort(nnzCol);
// 					RIT max_row = 0;
// 					if(nnzCol>0)
// 					{
// 						//if(tosort.size() < nnzCol) tosort.resize(nnzCol);
						
// 						for(size_t j=0, k=colIds_[i]; j<nnzCol; ++j, ++k)
// 						{
// 							tosort[j] = std::make_pair(rowPtr_[k], nzVals_[k]);
// 							max_row = std::max(max_row, rowPtr_[k]);
// 						}

// 						//sort(tosort.begin(), tosort.end());
// 						for(size_t expon = 1; max_row/expon > 0; expon *= 10){
// 							count_sort(tosort, expon);
// 						}
						
// 						size_t k = colIds_[i];
// 						rowPtr_[k] = tosort[0].first;
// 						nzVals_[k] = tosort[0].second;
						
// 						// k points to last updated entry
// 						for(size_t j=1; j<nnzCol; ++j)
// 						{
// 								rowPtr_[++k] = tosort[j].first;
// 								nzVals_[k] = tosort[j].second;
// 						}
// 					}
// 				}
			

// 			isColSorted_ = true;
// 	}
// 	else{
// 		return;
// 	}
// }

// template <typename RIT, typename VT, typename CPT>
// void CSR<RIT, VT, CPT>::column_split(std::vector< CSR<RIT, VT, CPT>* > &vec, int nsplits)
// {   
//     vec.resize(nsplits);
//     int ncolsPerSplit = ncols_ / nsplits;
// #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         int nthreads = omp_get_num_threads();
// #pragma omp for
//         for(int s = 0; s < nsplits; s++){
//             CPT colStart = s * ncolsPerSplit;
//             CPT colEnd = (s < nsplits-1) ? (s + 1) * ncolsPerSplit: ncols_;
//             pvector<CPT> colPtr(colEnd - colStart + 1);
//             for(int i = 0; colStart+i < colEnd + 1; i++){
//                 colPtr[i] = colIds_[colStart + i] - colIds_[colStart];
//             }
//             pvector<RIT> rowIds(rowPtr_.begin() + colIds_[colStart], rowPtr_.begin() + colIds_[colEnd]);
//             pvector<VT> nzVals(nzVals_.begin() + colIds_[colStart], nzVals_.begin() + colIds_[colEnd]);
//             vec[s] = new CSR<RIT, VT, CPT>(nrows_, colEnd - colStart, colIds_[colEnd] - colIds_[colStart], false, true);
//             vec[s]->cols_pvector(&colPtr);
//             vec[s]->nz_rows_pvector(&rowIds);
//             vec[s]->nz_vals_pvector(&nzVals);
//         }
//     }
// }


#endif
