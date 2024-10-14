#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "../COO.cpp"
#include "../CSC.cpp"
#include "../common/NIST/mmio.h"

const double DAMPING_FACTOR = 0.85;
const double EPSILON1 = 1e-6; // Convergence threshold
const int MAX_ITER = 100;    // Maximum number of iterations

// Function to compute PageRank
std::vector<double> PageRank(CSC<uint32_t, double>& graph, int n) {
    std::vector<double> rank(n, 1.0 / n);  // Initialize all ranks to 1/n as per the power iteration formula
    std::vector<double> new_rank(n, 0.0);  // Holds the rank for the next iteration
    std::vector<double> outdegree(n, 0.0); // Store the outdegree of each node

    // Calculate outdegree for each node (how many outgoing links each node has)
    for (size_t col = 0; col < n; col++) {
        outdegree[col] = graph.get_colPtr(col + 1) - graph.get_colPtr(col);
        // std::cout << outdegree[col] << "\n";
    }

    // Iteratively update ranks until convergence or maximum iterations
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Reset new_rank for each iteration
        std::fill(new_rank.begin(), new_rank.end(), 1.0/n);

        // Calculate new rank for each node based on incoming links
        for (size_t col = 0; col < n; col++) {
            for (size_t row = graph.get_colPtr(col); row < graph.get_colPtr(col + 1); row++) {
                uint32_t from_node = (*graph.get_rowIds())[row];
                if (outdegree[from_node] > 0) {
                    new_rank[col] += rank[from_node] / outdegree[from_node];
                }
            }
        }

        // Apply damping factor and random teleportation
        for (size_t i = 0; i < n; i++) {
            new_rank[i] = (DAMPING_FACTOR * new_rank[i]) + ((1.0 - DAMPING_FACTOR) / n);
        }

        // Check for convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; i++) {
            diff += std::fabs(new_rank[i] - rank[i]);
        }

        rank = new_rank;

        // If ranks have converged, break the loop
        if (diff < n * EPSILON1) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    return rank;
}

int main(int argc, char* argv[]) {
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

    coo.make_stochastic();  // Convert to stochastic form


    // Convert the COO matrix to CSC format
    CSC<uint32_t, double> cscMatrix(coo);

    // Print the CSC matrix information
    // cscMatrix.PrintInfo();
    // std::cout << "\nBefore stochastic: \n";
    cscMatrix.print_all();

    // Get the number of nodes in the graph {always it will be a square matrix}
    int n = coo.nrows();

    // Make the CSC matrix stochastic
    // cscMatrix.make_stochastic();  // another dumb try!

    // Print the CSC matrix information
    // std::cout << "\nAfter stochastic: \n";
    // cscMatrix.print_all();
    
    // Calculate PageRank
    std::vector<double> ranks = PageRank(cscMatrix, n);

    // Print the PageRank results
    // std::cout << "PageRank Results:" << std::endl;
    // for (size_t i = 0; i < ranks.size(); i++) {
    //     std::cout << "Node " << i + 1 << ": " << ranks[i] << std::endl;
    // }



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

    return 0;
}
