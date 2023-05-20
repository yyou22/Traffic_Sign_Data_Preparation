#include "cpca.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cpca_cpp, m) {
  m.doc() = "cPCA wrapped with pybind11";
  py::class_<CPCA>(m, "CPCA")
      .def(py::init<Eigen::Index const, bool const>())
      .def("initialize", &CPCA::initialize)
      .def("fit", &CPCA::fit)
      .def("transform", &CPCA::transform)
      // .def("fit_transform", &CPCA::fitTransform)
      .def("best_alpha", &CPCA::bestAlpha)
      .def("update_components", &CPCA::updateComponents)
      .def("best_alpha", &CPCA::bestAlpha)
      .def("logspace", &CPCA::logspace)
      .def("get_components", &CPCA::getComponents)
      .def("get_component", &CPCA::getComponent)
      .def("get_eigenvalues", &CPCA::getEigenvalues)
      .def("get_eigenvalue", &CPCA::getEigenvalue)
      .def("get_total_pos_eigenvalue", &CPCA::getTotalPosEigenvalue)
      .def("get_loadings", &CPCA::getLoadings)
      .def("get_loading", &CPCA::getLoading)
      .def("get_current_fg", &CPCA::getCurrentFg)
      .def("get_best_alpha", &CPCA::getBestAlpha)
      .def("get_reports", &CPCA::getReports);
}
