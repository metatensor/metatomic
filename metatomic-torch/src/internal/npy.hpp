#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace metatomic_torch {

// Write a CPU, contiguous tensor to NPY bytes.
// Supports: float64 (<f8), int64 (<i8), int32 (<i4), bool (|b1).
// Throws std::runtime_error on unsupported dtype/order or other errors.
std::vector<uint8_t> npy_write(const torch::Tensor& t);

// Read an NPY buffer into a CPU tensor.
// Accepts versions 1.0/2.0, ASCII header, C-order only.
// Supports: <f8, <i8, <i4, |b1.
torch::Tensor npy_read(const uint8_t* data, size_t size);

// Convenience overloads
inline torch::Tensor npy_read(const std::vector<uint8_t>& buf) {
  return npy_read(buf.data(), buf.size());
}

} // namespace metatomic_torch
