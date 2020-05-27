#ifndef CSC_ADDER_H
#define CSC_ADDER_H

#include "CSC.h" // need to check the relative paths for this section
#include "GAP/pvector.h"
#include "GAP/timer.h"

#include <vector> // needed while taking input
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iterator>
#include <assert.h>
#include <omp.h>

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
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

		}

		for(auto iter = umap.begin(); iter != umap.end(); iter++){
			if(iter->second != 0)
			{
				array_of_column_pointers[i].push_back(std::make_pair(iter->second, iter->first));
				count_for_nnz++;
			}
		}
	} 
	// parallel programming ended

	pvector<VT>* value_vector_for_csc = new pvector<VT>(count_for_nnz);
	pvector<RIT>* row_vector_for_csc = new pvector<RIT>(count_for_nnz);
	pvector<CPT>* prefix_sum_index_matrix = new pvector<CPT>(num_of_columns+1);

	(*prefix_sum_index_matrix)[0] = 0;
	for(CIT i = 1; i < num_of_columns+1; i++){
		(*prefix_sum_index_matrix)[i] = (*prefix_sum_index_matrix)[i-1] + array_of_column_pointers[i-1].size();
	}

#pragma omp parallel for
	for(CIT i = 0; i < num_of_columns; i++){
		size_t total_elements = array_of_column_pointers[i].size();
		for(size_t j = 0; j < total_elements; j++){
			(*row_vector_for_csc)[(j + (*prefix_sum_index_matrix)[i])] = ( (array_of_column_pointers[i]))[j].second;
			(*value_vector_for_csc)[(j + (*prefix_sum_index_matrix)[i])] = ( (array_of_column_pointers[i]))[j].first;
		}
	}// parallel programming ended

	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, count_for_nnz, false, true);
	result_matrix_csc.nz_rows_pvector(row_vector_for_csc);
	result_matrix_csc.cols_pvector(prefix_sum_index_matrix);
	result_matrix_csc.nz_vals_pvector(value_vector_for_csc);

	result_matrix_csc.sort_inside_column();

	clock.Stop();
	PrintTime("CSC Creation Time", clock.Seconds());

	return result_matrix_csc;

}




/////////////////////// adding vector of matrices in less time(more space as I used array of columns size of pointers) is done by here  /////////////////////////






template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM> // NM is number_of_matrices and CPT should include nnz range for the sum ie., should be suffice to be the CPT for sum

CSC<RIT, VT, CPT> add_vec_of_matrices_ls(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)          // _ls(at the end of function name) refers to less space, maybe more time(I used locks in omp prallel for)
{	 


	assert(vec_of_matrices.size() != 0);

	CIT num_of_columns = vec_of_matrices[0]->get_ncols();
	RIT num_of_rows = vec_of_matrices[0]->get_nrows();

#pragma omp parallel for
	for(NM i = 0; i < vec_of_matrices.size(); i++){
		assert(num_of_columns == vec_of_matrices[i]->get_ncols());
		assert(num_of_rows == vec_of_matrices[i]->get_nrows());
	}

	pvector<VT>* value_vector = new pvector<VT>();
	pvector<RIT>* row_vector = new pvector<RIT>();
	pvector<CIT>* column_vector = new pvector<CIT>();

	pvector<RIT>* nz_per_column = new pvector<RIT>(num_of_columns, 0); 

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
		}

		 omp_set_lock(&writelock);
		//#pragma openmp critical
		for(auto iter = umap.begin(); iter != umap.end(); iter++){
			if(iter->second != 0)
			{
				value_vector->push_back(iter->second);
				row_vector->push_back(iter->first);
				column_vector->push_back(i);
				count_for_nnz++;
				(*nz_per_column)[i] = (*nz_per_column)[i]+1;
			}
		}
		 omp_unset_lock(&writelock);

	} 
	omp_destroy_lock(&writelock);
	// parallel programming ended

//-------------comment from here 

	pvector<CPT>* prefix_sum_index_matrix = new pvector<CPT> (num_of_columns+1);
	pvector<CPT>* column_vector_for_csc = new pvector<CPT> (num_of_columns+1);
	(*prefix_sum_index_matrix)[0] = 0;
	(*column_vector_for_csc)[0] = 0;

	for(CIT i = 1; i < num_of_columns+1; i++){
		(*prefix_sum_index_matrix)[i] = (*prefix_sum_index_matrix)[i-1] + (*nz_per_column)[i-1];
		(*column_vector_for_csc)[i] = (*prefix_sum_index_matrix)[i];
	}


	pvector<VT>* value_vector_for_csc = new pvector<VT>(count_for_nnz);
	pvector<RIT>* row_vector_for_csc = new pvector<RIT>(count_for_nnz);

	omp_init_lock(&writelock);

#pragma omp parallel for
	for(size_t i = 0; i < count_for_nnz; i++){
		omp_set_lock(&writelock);
		CPT position = (*prefix_sum_index_matrix)[(*column_vector)[i] ];
		(*prefix_sum_index_matrix)[(*column_vector)[i] ]++;
		omp_unset_lock(&writelock);

		(*value_vector_for_csc)[position] = (*value_vector)[i];
		(*row_vector_for_csc)[position] = (*row_vector)[i];
		
	}
	omp_destroy_lock(&writelock);


	Timer clock;
	clock.Start();

	CSC<RIT, VT, CPT> result_matrix_csc(num_of_rows, num_of_columns, count_for_nnz, false, true);
	result_matrix_csc.nz_rows_pvector(row_vector_for_csc);
	result_matrix_csc.cols_pvector(column_vector_for_csc);
	result_matrix_csc.nz_vals_pvector(value_vector_for_csc);

	result_matrix_csc.sort_inside_column();

	clock.Stop();
	PrintTime("CSC Creation Time", clock.Seconds());

//------------to here for experimenting with time, note to uncomment below part while commenting above part.and also make sure to include COO.h for this process

	// COO<RIT, CIT, VT> result_matrix(num_of_rows, num_of_columns, count_for_nnz, true);

	// result_matrix.nz_rows_pvector(row_vector);
	// result_matrix.nz_cols_pvector(column_vector);
	// result_matrix.nz_vals_pvector(value_vector);

	// CSC<RIT, VT, CPT> result_matrix_csc(result_matrix);  // mergeduplicatesort in (COO to CSC) in CSC.h can be optimised for this case as there won't be any duplicates here i.e., can avoid radix sort kinda sequential addition there
	
	return result_matrix_csc;

}

#endif