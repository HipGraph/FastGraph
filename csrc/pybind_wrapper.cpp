#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "common/COO.h"
#include "common/CSC.h"
#include "common/CSR.h"


namespace py = pybind11;

template <typename RIT, typename CIT, typename VT=double>
void define_coo(py::module &m, std::string classname)
{
	py::class_<COO<RIT,CIT,VT>>(m, classname.c_str())
		.def(py::init<>())
		.def("GenER", static_cast<void (COO<RIT,CIT,VT>::*)(int,int,bool,int64_t)>(&COO<RIT,CIT,VT>::GenER), "GenER")
		.def("GenER", static_cast<void (COO<RIT,CIT,VT>::*)(int,int,int,bool,int64_t)>(&COO<RIT,CIT,VT>::GenER), "GenER")
		.def("PrintInfo",&COO<RIT,CIT,VT>::PrintInfo);
	
	// py::class_<CSC<RIT,VT,CPT>>(m, classname.c_str())
	// 	.def(py::init<COO<RIT,CIT,VT>());
}


template <typename RIT, typename CIT, typename VT=double,typename CPT=size_t>
void define_csc(py::module &m, std::string classname)
{
	py::class_<CSC<RIT,VT,CPT>>(m, classname.c_str())
		.def(py::init<COO<RIT,CIT,VT>&>())
		.def("column_reduce",&CSC<RIT,VT,CPT>::column_reduce)
		.def("ewiseApply",&CSC<RIT,VT,CPT>::ewiseApply);
		
}


template <typename CIT, typename RIT, typename VT=double,typename RPT=size_t>
void define_csr(py::module &m,std::string classname)
{
	py::class_<CSR<CIT,VT,RPT>>(m, classname.c_str())
		.def(py::init<COO<RIT,CIT,VT>&>())
		// .def("row_reduce",&CSR<RIT,VT,CPT>::row_reduce)
		.def("ewiseApply",&CSR<CIT,VT,RPT>::ewiseApply)
		.def("row_reduce",&CSR<CIT,VT,RPT>::row_reduce);
}

PYBIND11_MODULE(csplib, m) {
	define_coo<int32_t,int32_t,int32_t>(m, "COO_int");
	define_csc<int32_t, int32_t, int32_t>(m, "CSC_double");
	define_csr<int32_t, int32_t, int32_t>(m, "CSR_double");
}

// PYBIND11_MODULE(CSC, m) {
	
// }
