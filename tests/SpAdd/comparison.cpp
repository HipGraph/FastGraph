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

int main(int argc, char* argv[]){

    double t0, t1, t2, t3;

	std::vector< CSC<uint32_t, float, uint32_t>* > vec;
    //std::vector< CSC<uint32_t, float, uint32_t>* > vec_temp;

    // below is method to use random matrices from COO.h
    uint32_t total_nnz_in = 0;
    uint32_t total_nnz_out = 0;
    for(int i = 0; i < 64; i++){
        std::string filename("/N/u2/t/taufique/Data/r0_s");
        filename = filename + std::to_string(i);
        COO<uint32_t, uint32_t, float> coo;
        t0 = omp_get_wtime();
        coo.ReadMM(filename);
        t1 = omp_get_wtime();
        printf("Time taken to read %s: %lf\n", filename.c_str(), t1-t0);
        vec.push_back(new CSC<uint32_t, float, uint32_t>(coo));
        //vec_temp.push_back(new CSC<uint32_t, float, uint32_t>(coo));
        total_nnz_in += vec[i]->get_nnz();
    }
    
    Timer clock;

    //std::vector<int> threads{1, 6, 12, 24, 48};
    //std::vector<int> threads{1, 16, 48};
    //std::vector<int> threads{48, 1, 12};
    std::vector<int> threads{48};

    int iterations = 1;

    for(int i = 0; i < threads.size(); i++){
        omp_set_num_threads(threads[i]);

        ////std::vector< CSC<uint32_t, float, uint32_t>* > tree(vec.begin(), vec.end());
        //std::vector< CSC<uint32_t, float, uint32_t>* > tree(vec_temp.begin(), vec_temp.end());
        //CSC<uint32_t, float, uint32_t> * temp1;
        //CSC<uint32_t, float, uint32_t> * temp2;
        //int nIntermediate = tree.size();
        //int level = 0;
        //double tree_time = 0;
        //while(nIntermediate > 1){
            //int j = 0;
            //int idxf = j * 2 + 0;
            //int idxs = idxf;
            //if(idxs + 1 < nIntermediate) idxs++;
            //while(idxs < nIntermediate){
                //if(idxf < idxs){
                    //temp1 = tree[idxf];
                    //temp2 = tree[idxs];
                    //clock.Start();
                    //pvector<uint32_t> nnzCPerCol = symbolicSpAddRegularDynamic<uint32_t,uint32_t,float,uint32_t>(tree[idxf], tree[idxs]);
                    //tree[j] = new CSC<uint32_t, float, uint32_t>(SpAddRegularDynamic<uint32_t,uint32_t,float,uint32_t>(tree[idxf], tree[idxs], nnzCPerCol));
                    //clock.Stop();
                    //tree_time += clock.Seconds();
                    //delete temp1;
                    //delete temp2;
                //}
                //else{
                    //tree[j] = tree[idxf];
                //}
                //j++;
                //idxf = j * 2 + 0;
                //idxs = idxf;
                //if(idxs + 1 < nIntermediate) idxs++;
            //}
            //nIntermediate = j;
            //level++;
        //}
        //std::cout << threads[i] << ",";
        //std::cout << "SpAddPairwiseTreeDynamic" << ","; 
        //std::cout << tree_time << std::endl;
        
        CSC<uint32_t, float, uint32_t> SpAdd_out;
        for(int it = 0; it < 1; it++){
            clock.Start(); 
            SpAdd_out = SpMultiAddHashSlidingDynamic<uint32_t,uint32_t, float, uint32_t> (vec, 32 * 1024, 32 * 1024, true);
            //SpAdd_out = SpMultiAddHashSlidingDynamic<uint32_t,uint32_t, float,uint32_t> (vec, 512, 512, true);
            clock.Stop();
            std::cout << threads[i] << ",";
            std::cout << "SpMultiAddHashSlidingDynamic" << ","; 
            std::cout << clock.Seconds() << std::endl;
            //std::cout << SpAddHybrid_out.get_nnz() << std::endl;
            //SpAddHybrid_out.print_all();
        }
        total_nnz_out = SpAdd_out.get_nnz();

        std::cout << "total_nnz_in: " << total_nnz_in << std::endl;
        std::cout << "total_nnz_out: " << total_nnz_out << std::endl;
        
    }

	return 0;

}
