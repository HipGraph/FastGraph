from COO import *
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from memory_profiler import profile

mem_logs = open('mem_profile.log','a')

# def splib_to

@profile(stream=mem_logs)
def run_test():
    c=COO_double()
    c.PrintInfo()
    print((c.get_row_count(), c.get_col_count()))

    # c.GenER(2,2,True,1)
    c.PrintInfo()

    c.update_row_pvector(np.array([1, 2, 3, 4, 99]))
    c.PrintInfo()
    # x = np.asarray(memoryview(c.get_row_ptr()))
    # x = np.ctypeslib.as_array(c.get_row_ptr())
    # x = np.array(c.get_row_ptr(), copy=False)


    m1_row = np.array(c.get_row_ptr(), copy=False)
    print(m1_row)
    m1_col = np.array(c.get_col_ptr(), copy=False)
    m1_val = np.array(c.get_val_ptr(), copy=False)
    # print(m1_row)
    # print(m1_col)
    # print(m1_val)

    # print((c.get_row_count(), c.get_col_count()))
    mat = coo_matrix((m1_val, (m1_row, m1_col)), shape=(c.get_row_count(), c.get_col_count()))


    G = nx.from_scipy_sparse_array(mat, create_using=nx.DiGraph)
    print(G)


    # print(len(x))
    # x = np.frombuffer(c.get_row_ptr(), dtype=np.int32, count=4)
    # del c

    # print(x)


    d=CSC_double(c)
    #d=CSC_double()
    print(type(c))
    # print(type(d))
    print(type(d))
    d.column_reduce()
    #d(c)
    #CSC_double(c)
    #d(c)
    #d=CSC_double(c)
    #d=CSC_double(c<int32_t,double,int32_t>)


    #d=CSC(c)

run_test()

def coo_to_scipy_sparse_array(splib_coo, copy=False, return_networkx=True):
    m1_row = np.array(splib_coo.get_row_ptr(), copy=copy)
    m1_col = np.array(splib_coo.get_col_ptr(), copy=copy)
    m1_val = np.array(splib_coo.get_val_ptr(), copy=copy)
    # print(m1_row)
    # print(m1_col)
    # print(m1_val)

    # print((c.get_row_count(), c.get_col_count()))
    mat = coo_matrix((m1_val, (m1_row, m1_col)), shape=(c.get_row_count(), c.get_col_count()))
    
    if not return_networkx:
        return mat
    G = nx.from_scipy_sparse_array(mat, create_using=nx.DiGraph)
    # print(G)
    return G