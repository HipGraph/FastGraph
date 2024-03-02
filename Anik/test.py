from COO import *
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import adapter 

def run_test1():
    row  = np.array([0, 0, 1, 3, 1, 0, 0])
    col  = np.array([0, 2, 1, 3, 1, 0, 0])
    data = np.array([1, 1, 1, 1, 99, 1, 1], dtype=np.float32)
    A = coo_matrix((data, (row, col)), shape=(4, 4))

    c = adapter.scipy_sparse_matrix_to_splib_coo(A)
    
    c.PrintInfo()

    B = adapter.splib_coo_to_scipy_sparse_array(c)
    print(B)

def run_test2():
    c = COO_double()
    c.GenER(2,2,True,1)
    c.PrintInfo()
    B = adapter.splib_coo_to_scipy_sparse_array(c)
    print(B)
    d = adapter.scipy_sparse_matrix_to_splib_coo(B)
    d.PrintInfo()
    e = CSC_double(c)
    e.column_reduce()



# @profile(stream=mem_logs)
def run_test3():
    c=COO_double()
    c.PrintInfo()
    print((c.get_row_count(), c.get_col_count()))
    c.GenER(2,2,True,1)
    c.PrintInfo()
    d=CSC_double(c)
    print(type(c))
    print(type(d))
    d.column_reduce()
    



run_test2()
