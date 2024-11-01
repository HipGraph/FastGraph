
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>
// #include "../COO.cpp"
// #include "../CSC.cpp"

#include "../common/COO.h"
#include "../common/CSC.h"
#include "../common/NIST/mmio.h"



const double DAMPING_FACTOR = 0.85;
const double EPSILON1 = 1e-9; // Convergence threshold
const int MAX_ITER = 100;    // Maximum number of iterations

// This PageRank uses  only CSC matrix
std::vector<double> PageRank(CSC<uint32_t, double, size_t>& graph, int n) {
    std::vector<double> rank(n, 1.0 / n);  // Initialize ranks uniformly
    std::vector<double> new_rank(n, 0.0);  // Holds the rank for the next iteration
    std::vector<double> outdegree(n, 0.0); // Store the outdegree of each node

    // Get references to the CSC data
    const pvector<size_t>& colPtr = *(graph.get_colPtr());
    const pvector<uint32_t>& rowIds = *(graph.get_rowIds());
    const pvector<double>& nzVals = *(graph.get_nzVals());

    // Calculate outdegree for each node (number of outgoing links)
    for (size_t col = 0; col < n; col++) {
        for (size_t idx = colPtr[col]; idx < colPtr[col + 1]; idx++) {
            uint32_t row = rowIds[idx];
            outdegree[row] += nzVals[idx];
        }
    }

    // Identify dangling nodes (nodes with no outgoing edges)
    std::vector<uint32_t> dangling_nodes;
    for (size_t i = 0; i < n; i++) {
        if (outdegree[i] == 0.0) {
            dangling_nodes.push_back(i);
        }
    }

    // Iteratively update ranks until convergence or maximum iterations
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Copy the rank vector for this iteration
        const std::vector<double> rank_last = rank;

        // Reset new_rank and include the teleportation factor
        std::fill(new_rank.begin(), new_rank.end(), (1.0 - DAMPING_FACTOR) / n);

        // Sum of ranks of dangling nodes
        double dangling_sum = 0.0;
        for (uint32_t node : dangling_nodes) {
            dangling_sum += rank_last[node];
        }
        dangling_sum *= DAMPING_FACTOR / n;

        // Distribute ranks through incoming links
        for (size_t col = 0; col < n; col++) {
            size_t col_start = colPtr[col];
            size_t col_end = colPtr[col + 1];

            for (size_t idx = col_start; idx < col_end; idx++) {
                uint32_t row = rowIds[idx];
                if (outdegree[row] > 0) {
                    new_rank[col] += DAMPING_FACTOR * rank_last[row] * nzVals[idx] / outdegree[row];
                }
            }
        }

        // Add dangling node contributions
        for (size_t i = 0; i < n; i++) {
            new_rank[i] += dangling_sum;
        }

        // Check for convergence (L1 norm)
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            diff += std::fabs(new_rank[i] - rank[i]);
        }

        // Update the rank vector
        rank.swap(new_rank);

        // If ranks have converged, break the loop
        if (diff < EPSILON) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    return rank;
}

/* 
int main(int argc, char* argv[]) {
    // Set precision to max
    // std::cout << std::fixed << std::setprecision(15);
    // Check for the correct number of command-line arguments
    if (argc < 2) {
        std::cerr << "Kindly input or pass the MM file as command line argumetn." << std::endl;
        return 1;
    }
    
    // Create a COO matrix to read the Matrix Market (MM) file
    COO<uint32_t, uint32_t, double> coo;
    std::string filename = std::string(argv[1]);
    coo.ReadMM(filename);  // Read the MM file
    // std::cout << "File: " << filename << std::endl;

    coo.PrintInfo();  // Print COO matrix information
    coo.print_all();
    coo.make_stochastic();  // Convert to stochastic form


    // Convert the COO matrix to CSC format
    CSC<uint32_t, double, size_t> cscMatrix(coo);

    // Print the CSC matrix information
    // cscMatrix.PrintInfo();
    // std::cout << "\nBefore stochastic: \n";
    // cscMatrix.print_all();

    // Get the number of nodes in the graph {always it will be a square matrix}
    int n = coo.nrows();

    // Make the CSC matrix stochastic
    // cscMatrix.make_stochastic();  // another dumb try!

    // Print the CSC matrix information
    // std::cout << "\nAfter stochastic: \n";
    // cscMatrix.print_all();
    
    auto start = std::chrono::high_resolution_clock::now();

    // Calculate PageRank
    std::vector<double> ranks = PageRank(cscMatrix, n);

    // Print the PageRank results
    // std::cout << "PageRank Results:" << std::endl;
    // for (size_t i = 0; i < ranks.size(); i++) {
    //     std::cout << "Node " << i + 1 << ": " << ranks[i] << std::endl;
    // }


    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
     // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Print the sorted PageRank results
    std::vector<std::pair<int, double>> rank_pairs;
    for (size_t i = 0; i < ranks.size(); i++) {
        rank_pairs.push_back({i + 1, ranks[i]});  // Store node and its rank
    }
    std::sort(rank_pairs.begin(), rank_pairs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;  // Sort by rank value, descending
    });

    std::cout << "Sorted PageRank (C++):" << std::endl;
    for (const auto& [node, rank] : rank_pairs) {
        std::cout << "Node " << node << ": " << rank << std::endl;
    }

    // Output the elapsed time in seconds
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
 */
