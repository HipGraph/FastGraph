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
    int x = atoi(argv[1]); // scale of random matrix, indicates number of rows
    int y = atoi(argv[2]); // scale of random matrix, indicates number of columns
    int d = atoi(argv[3]); // average degree of random matrix
	bool weighted = true;

	int k = atoi(argv[4]); // number of matrices
    int type = atoi(argv[5]); // Type of matrix
    int t = atoi(argv[6]); // number of threads

	std::vector< CSC<uint32_t, uint32_t, uint32_t>* > vec;
    std::vector< CSC<uint32_t, uint32_t, uint32_t>* > vec_temp;

    uint64_t total_nnz_in = 0;
    uint64_t total_nnz_out = 0;

    for(int i = 0; i < k; i++){
        COO<uint32_t, uint32_t, uint32_t> coo;
        if(type == 0){
            coo.GenER(x, y, d, weighted, i);   // Generate a weighted ER matrix with 2^x rows, 2^y columns and d nonzeros per column using random seed i
        }
        else{
            // For RMAT matrix need to be square. So x need to be equal to y.
            if (x != y){
                x = std::min(x,y);
            }
            coo.GenRMAT(x, d, weighted, i);   // Generate a weighted RMAT matrix with 2^x rows, 2^x columns and d nonzeros per column using random seed i
        }

        vec.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
        //vec_temp.push_back(new CSC<uint32_t, uint32_t, uint32_t>(coo));
        //vec[i]->print_all();
        total_nnz_in += vec[vec.size()-1]->get_nnz();
    }
    
    Timer clock;


    //std::vector<int> threads{1, 6, 12, 24, 48};
    //std::vector<int> threads{1, 16, 48};
    //std::vector<int> threads{48, 1, 12};
    std::vector<int> threads{48};

    int iterations = 1;

    for(int i = 0; i < threads.size(); i++){
        //omp_set_num_threads(threads[i]);
        //mkl_set_num_threads(threads[i]);

        omp_set_num_threads(t);
        mkl_set_num_threads(t);

        CSC<uint32_t, uint32_t, uint32_t> OutPairwiseLinear;
        CSC<uint32_t, uint32_t, uint32_t> SpAdd_out;
        pvector<uint32_t> nnzCPerCol;

        std::vector< CSC<uint32_t, uint32_t, uint32_t>* > tree(vec.begin(), vec.end());
        //std::vector< CSC<uint32_t, uint32_t, uint32_t>* > tree(vec_temp.begin(), vec_temp.end());
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
                    pvector<uint32_t> nnzCPerCol = symbolicSpAddRegularStatic<uint32_t,uint32_t,uint32_t,uint32_t>(tree[idxf], tree[idxs]);
                    tree[j] = new CSC<uint32_t, uint32_t, uint32_t>(SpAddRegularStatic<uint32_t,uint32_t,uint32_t,uint32_t>(tree[idxf], tree[idxs], nnzCPerCol));
                    clock.Stop();
                    tree_time += clock.Seconds();
                    delete temp1;
                    delete temp2;
                }
                else{
                    tree[j] = tree[idxf];
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
        std::cout << d << "," ;
        std::cout << k << "," ;
        std::cout << t << ",";
        std::cout << "SpAddPairwiseTreeStatic" << ","; 
        std::cout << tree_time << ",";
        std::cout << total_nnz_in << ",";
        std::cout << tree[0]->get_nnz() << std::endl;
    }

	return 0;

}
