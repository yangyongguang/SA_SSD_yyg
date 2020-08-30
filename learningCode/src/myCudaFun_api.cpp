#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "myCudaFun_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_warpper", &add_wrapper_fast, "add_warpper_fast");
}