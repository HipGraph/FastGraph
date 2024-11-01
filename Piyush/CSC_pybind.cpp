
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "common/COO.h"
#include "common/CSC.h"

// #include "COO.cpp"
// #include "CSC.cpp"

namespace py = pybind11;

template <typename RIT, typename CIT, typename VT=double,typename CPT=size_t>
void define_csc(py::module &m, std::string classname)
{
	py::class_<CSC<RIT,VT,CPT>>(m, classname.c_str())
		.def(py::init<COO<RIT,CIT,VT>&>())
		.def(py::init<>());
		
}

PYBIND11_MODULE(CSC, m) {
	//define_coo<int,int,double>(m, "COO_int");
	define_csc<int32_t, double, int32_t>(m, "CSC_double");
}

// PYBIND11_MODULE(CSC, m) {
	
// }