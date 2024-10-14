from COO import *
import numpy as np
import scipy.io
from scipy.sparse import coo_matrix
import networkx as nx
import adapter 

# export OMP_NUM_THREADS=4
def run_test1():
    row  = np.array([0, 0, 1, 3, 1, 0, 0])
    col  = np.array([0, 2, 1, 3, 1, 0, 0])
    data = np.array([1, 1, 1, 1, 99, 1, 1], dtype=np.float32)
    A = coo_matrix((data, (row, col)), shape=(4, 4))

    c = adapter.scipy_sparse_matrix_to_splib_coo(A)
    # c.GenER(4,4, False, 1)
    c.PrintInfo()
    
    B = adapter.splib_coo_to_scipy_sparse_array(c)
    print(B)

def run_test2():
    c = COO_double()
    c.GenER(4,4, True, 1)
    c.PrintInfo()
    B = adapter.splib_coo_to_scipy_sparse_array(c)
    print(B)
    d = adapter.scipy_sparse_matrix_to_splib_coo(B)
    d.PrintInfo()
    print('ok')
    e = CSC_double(c)  # CSR
    e.column_reduce()

def run_test4():
    c = COO_double()
    c.GenER(11,11, True, 1)
    c.PrintInfo()
    B = adapter.splib_coo_to_scipy_sparse_array(c)
    print(B)
    d = adapter.scipy_sparse_matrix_to_splib_coo(B)
    d.PrintInfo()
    print('ok')
    print("\nType: ", type(c))
    e = CSR_double(c)  # CSR
    e.PrintInfo()
    # e.column_reduce()

def run_test5():
    # Step 1: Read the Matrix Market file (present in current directgory) into a scipy sparse COO matrix
    mm_file = 'input_2.mtx'
    print("Step 1: Reading Matrix Market file...")
    try:
        A = scipy.io.mmread(mm_file).tocoo()
        print(f"Matrix shape: {A.shape}, number of non-zeros: {A.nnz}")
    except Exception as e:
        print(f"Error reading MM file: {e}")
        return

    # Step 2: Create a NetworkX directed graph from the scipy sparse COO matrix
    print("\nStep 2: Creating NetworkX graph from the sparse matrix...")
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    print(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Step 3: Convert the NetworkX graph back to a scipy COO sparse matrix
    # This ensures that any internal ordering or processing by NetworkX is captured
    print("\nStep 3: Converting NetworkX graph back to scipy sparse COO matrix...")

    scipy_coo = nx.to_scipy_sparse_matrix(G, format='coo')
    print(f"Converted scipy COO matrix shape: {scipy_coo.shape}, number of non-zeros: {scipy_coo.nnz}")

    # Here end-user can do other operations with NetworkX graph

    # Step 4: Next, convert the scipy COO matrix to FastGraph's COO using adapter.py
    print("\nStep 4: Converting scipy COO matrix to FastGraph COO...")
    fastgraph_coo = adapter.scipy_sparse_matrix_to_splib_coo(scipy_coo)
    print("Conversion to FastGraph COO completed.")

    # Step 5: Print some information about the FastGraph COO for verification
    print("\nStep 5: FastGraph COO information (first 10 entries):")
    row_ptr = fastgraph_coo.get_row_ptr()
    col_ptr = fastgraph_coo.get_col_ptr()
    val_ptr = fastgraph_coo.get_val_ptr()
    print(f"Rows (first 10): {row_ptr[:10]}")
    print(f"Columns (first 10): {col_ptr[:10]}")
    print(f"Values (first 10): {val_ptr[:10]}")


    # Step 6: Converting to CSR and printing CSR info
    print("\nStep: Converting FastGraph COO to CSR and printing CSR info...")
    # print("\nType: ", type(fastgraph_coo))
    csr = CSR_double(fastgraph_coo)
    csr.PrintInfo()

# @profile(stream=mem_logs)
def run_test3():
    c=COO_double()
    c.PrintInfo()
    print((c.get_row_count(), c.get_col_count()))
    c.GenER(8,8,True, 1)
    c.PrintInfo()
    d=CSC_double(c)
    print(type(c))
    print(type(d))
    d.column_reduce()
    




# run_test1()
# run_test2()
# run_test3()
# run_test4()   # do not look at this test case
run_test5()
