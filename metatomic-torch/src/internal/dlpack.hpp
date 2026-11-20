#ifndef METATOMIC_TORCH_INTERNAL_DLPACK_HPP
#define METATOMIC_TORCH_INTERNAL_DLPACK_HPP

#include <torch/types.h>

#include <metatomic.hpp>

namespace metatomic_torch {
namespace details {

/// Create a DLPack tensor sharing its data with `tensor`. Meta tensors are
/// exported with `kDLExtDev` as the device and no data.
metatomic::DLPackTensor tensor_to_dlpack(const torch::Tensor& tensor);

/// Create a `torch::Tensor` sharing its data with `tensor`, taking ownership
/// of the DLPack tensor.
torch::Tensor dlpack_to_tensor(metatomic::DLPackTensor tensor);

}
}

#endif
