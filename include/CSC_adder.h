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


//..........................................................................//



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