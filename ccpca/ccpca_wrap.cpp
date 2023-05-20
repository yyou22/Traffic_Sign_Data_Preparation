#include "ccpca.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(ccpca_cpp, m) {
  m.doc() = "ccPCA wrapped with pybind11";
  py::class_<CCPCA>(m, "CCPCA")
      .def(py::init<Eigen::Index const, bool const>())
      // fitTransform has some bug probably related to using pybind11
      // .def("fit_transform", &CCPCA::fitTransform)
      .def("fit", &CCPCA::fit)
      .def("transform", &CCPCA::transform)
      .def("best_alpha", &CCPCA::bestAlpha)
      .def("get_feat_contribs", &CCPCA::getFeatContribs)
      .def("get_scaled_feat_contribs", &CCPCA::getScaledFeatContribs)
      .def("get_components", &CCPCA::getComponents)
      .def("get_component", &CCPCA::getComponent)
      .def("get_eigenvalues", &CCPCA::getEigenvalues)
      .def("get_eigenvalue", &CCPCA::getEigenvalue)
      .def("get_total_pos_eigenvalue", &CCPCA::getTotalPosEigenvalue)
      .def("get_loadings", &CCPCA::getLoadings)
      .def("get_loading", &CCPCA::getLoading)
      .def("get_first_component", &CCPCA::getFirstComponent)
      .def("get_best_alpha", &CCPCA::getBestAlpha)
      .def("get_reports", &CCPCA::getReports);
}
