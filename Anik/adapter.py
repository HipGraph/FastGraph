from COO import *
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
# from memory_profiler import profile

# mem_logs = open('mem_profile.log','a')


def splib_coo_to_scipy_sparse_array(splib_coo, copy=False):
    m1_row = np.array(splib_coo.get_row_ptr(), copy=copy)
    m1_col = np.array(splib_coo.get_col_ptr(), copy=copy)
    m1_val = np.array(splib_coo.get_val_ptr(), copy=copy)
    
    mat = coo_matrix((m1_val, (m1_row, m1_col)), shape=(splib_coo.get_row_count(), splib_coo.get_col_count()))
    
    return mat
    # G = nx.from_scipy_sparse_array(mat, create_using=nx.DiGraph)
    

def scipy_sparse_matrix_to_splib_coo(A):
    # only pass by reference for now. Ownership = NP. C++ will not call destructor for this
    # A = nx.to_scipy_sparse_array(G, format='coo')
    # (r, c) = A.shape
    # nnz = A.nnz

    row = A.row
    col = A.col
    val = A.data

    ret = COO_double()
    ret.update_row_pvector(row, False)   # transfer ownership = False
    ret.update_col_pvector(col, False)
    ret.update_val_pvector(val, False)

    return ret