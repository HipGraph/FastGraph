from .csplib import COO_int
from .csplib import CSC_double
from .csplib import CSR_double


def test():
	c=COO_int()
	c.GenER(3,2,True,1)

	c.PrintInfo()
	#print("cooprintall")
	#c.print_all()
	d=CSC_double(c)
	#print("type d",type(d))
	#print("cscprintall")
	#d.csc_print_all()
	#print("type c",type(c))
	#print("type d",type(d))
	#d.column_reduce()
	#d.ewiseApply(2)

	e=CSR_double(c)
	e.PrintInfo()
	#print("type e",type(e))
	#e.ewiseApply(2)
	#e.row_reduce()
	#dimapplyvector=[1,2,3,4]
	#e.dimApply(dimapplyvector)
	