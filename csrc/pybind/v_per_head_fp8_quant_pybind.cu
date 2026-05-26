#include <torch/extension.h>

namespace aiter {
std::tuple<at::Tensor, at::Tensor> v_per_head_fp8_quant(at::Tensor& v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v_per_head_fp8_quant",
          &aiter::v_per_head_fp8_quant,
          py::arg("v"));
}
