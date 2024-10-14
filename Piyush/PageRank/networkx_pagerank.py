import networkx as nx
import numpy as np
import sys

# Function to read the MM file and create a graph
def read_mm_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Skip comments starting with %%
    lines = [line for line in lines if not line.startswith('%')]
    
    nrows, ncols, nnz = map(int, lines[0].split())
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges from the remaining lines
    for line in lines[1:]:
        row, col, weight = map(float, line.split())
        # print(row, col, weight)
        G.add_edge(int(row), int(col), weight=weight)
    
    return G

def compute_pagerank(filename):
    G = read_mm_file(filename)
    # Get the number of nodes
    n = G.number_of_nodes()
    
    # Initialize the nstart vector to 1/n for each node
    # nstart = {node: 1.0/n for node in G.nodes}

    # Compute PageRank using NetworkX

    W = nx.stochastic_graph(G, weight='weight')
    print("\nstochastic_graph: ")
    for edge in W.edges(data=True):
        print(edge)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6) #, nstart=nstart)
    
    # Print the resultss
    # print("PageRank Results:")
    # for node, rank in pagerank.items():
    #     print(f"Node {node}: {rank}")

     # Sort by PageRank score
    sorted_ranks = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    
    print("Sorted PageRank (Python NetworkX):")
    for node, rank in sorted_ranks:
        print(f"Node {node}: {rank}")

if __name__ == "__main__": 
    filename = sys.argv[1]
    # filename = 'utm2a.mtx' 
    compute_pagerank(filename)
