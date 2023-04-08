from .csplib import COO_int
from .csplib import CSC_double
#from .csplib import CSR_double


def test():
	c=COO_int()
	c.GenER(2,2,True,1)

	c.PrintInfo()
	d=CSC_double(c)
	#print("type c",type(c))
	#print("type d",type(d))
	#d.column_reduce()
	d.ewiseApply(2)

	#e=CSR_double(c)
	#print("type e",type(e))
	#e.ewiseApply(2)
	#e.row_reduce()
	