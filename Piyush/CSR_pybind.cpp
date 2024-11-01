
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "common/COO.h"
#include "common/CSR.h"

// #include "COO.cpp"
// #include "CSR.cpp"

namespace py = pybind11;

template <typename RIT, typename CIT, typename VT=double,typename CPT=size_t>
void define_csr(py::module &m, std::string classname)
{
	py::class_<CSR<RIT,VT,CPT>>(m, classname.c_str())
		.def(py::init<COO<RIT,CIT,VT>&>())
		.def("PrintInfo",&CSR<RIT,VT,CPT>::PrintInfo)
		.def(py::init<>());
		
}

PYBIND11_MODULE(CSR, m) {
	//define_coo<int,int,double>(m, "COO_int");
	define_csr<int32_t, double, int32_t>(m, "CSR_double");
}

// PYBIND11_MODULE(CSC, m) {
	
// }