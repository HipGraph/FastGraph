#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <cstdlib>

#include "COO.cpp"
#include "CSC.cpp"

namespace py = pybind11;

template <typename RIT, typename CIT, typename VT=double>
void define_coo(py::module &m, std::string classname)
{
	py::class_<COO<RIT,CIT,VT>>(m, classname.c_str())
		.def(py::init<>())
		.def("GenER", static_cast<void (COO<RIT,CIT,VT>::*)(int,int,bool,int64_t)>(&COO<RIT,CIT,VT>::GenER), "GenER")
		.def("GenER", static_cast<void (COO<RIT,CIT,VT>::*)(int,int,int,bool,int64_t)>(&COO<RIT,CIT,VT>::GenER), "GenER")
		.def("get_row_count", &COO<RIT,CIT,VT>::nrows)
		.def("get_col_count", &COO<RIT,CIT,VT>::ncols)
		.def("get_nnz_count", &COO<RIT,CIT,VT>::nnz)
		.def("get_row_ptr", [](COO<RIT,CIT,VT> &M) {
			return py::array_t<RIT> (
				{M.get_row_vector().size()},
				{sizeof(RIT)},
				M.get_row_vector().data(),
				py::capsule(M.get_row_vector().data(), [](void *p){ /* Numpy will manage this memory */ })
			);
		})
		.def("get_col_ptr", [](COO<RIT,CIT,VT> &M) {
			return py::array_t<CIT> (
				{M.get_col_vector().size()},
				{sizeof(CIT)},
				M.get_col_vector().data(),
				py::capsule(M.get_col_vector().data(), [](void *p){ /* Numpy will manage this memory */ })
			);
		})
		.def("get_val_ptr", [](COO<RIT,CIT,VT> &M) {
			return py::array_t<VT> (
				{M.get_val_vector().size()},
				{sizeof(VT)},
				M.get_val_vector().data(),
				py::capsule(M.get_val_vector().data(), [](void *p){ /* Numpy will manage this memory */ })
			);
		})
		.def("update_row_pvector", [](COO<RIT,CIT,VT> &M, py::array_t<RIT> np_array, bool transfer_ownership) {
			py::buffer_info buf = np_array.request();
			RIT *ptr = static_cast<RIT *>(buf.ptr);
			size_t sz = buf.size;
			M.update_row_pvector(ptr, sz, transfer_ownership);
		})
		.def("update_col_pvector", [](COO<RIT,CIT,VT> &M, py::array_t<CIT> np_array, bool transfer_ownership) {
			py::buffer_info buf = np_array.request();
			CIT *ptr = static_cast<CIT *>(buf.ptr);
			size_t sz = buf.size;
			M.update_col_pvector(ptr, sz, transfer_ownership);
		})
		.def("update_val_pvector", [](COO<RIT,CIT,VT> &M, py::array_t<VT> np_array, bool transfer_ownership) {
			py::buffer_info buf = np_array.request();
			VT *ptr = static_cast<VT *>(buf.ptr);
			size_t sz = buf.size;
			M.update_val_pvector(ptr, sz, transfer_ownership);
		})
		.def("PrintInfo",&COO<RIT,CIT,VT>::PrintInfo);
	
	// py::class_<CSC<RIT,VT,CPT>>(m, classname.c_str())
	// 	.def(py::init<COO<RIT,CIT,VT>());
}


template <typename RIT, typename CIT, typename VT=double,typename CPT=size_t>
void define_csc(py::module &m, std::string classname)
{
	py::class_<CSC<RIT,VT,CPT>>(m, classname.c_str())
		.def(py::init<COO<RIT,CIT,VT>&>())
		.def("column_reduce",&CSC<RIT,VT,CPT>::column_reduce);
		
}

PYBIND11_MODULE(COO, m) {
	define_coo<int32_t,int32_t,int32_t>(m, "COO_int");
	define_coo<int32_t, int32_t, double>(m, "COO_double");
	define_csc<int32_t, int32_t, double>(m, "CSC_double");
}

// PYBIND11_MODULE(CSC, m) {
	
// }