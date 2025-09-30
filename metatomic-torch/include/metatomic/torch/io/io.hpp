#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

#include "metatomic/torch/system.hpp"

namespace metatomic_torch {

// ===== File-based =====
void   save(const std::string& path, const System& system);
System load_system(const std::string& path);

// ===== In-memory =====
torch::Tensor save_buffer(const System& system);
System               load_system_buffer(const uint8_t* data, size_t size);
inline System        load_system_buffer(const std::vector<uint8_t>& data) {
  return load_system_buffer(data.data(), data.size());
}
inline System        load_system_buffer(const torch::Tensor& data) {
  // enforce CPU, contiguous, uint8, 1D
  auto t = data.contiguous().to(torch::kCPU);
  if (t.scalar_type() != torch::kUInt8) {
      throw std::runtime_error("System pickle: expected torch.uint8 buffer");
  }
  if (t.dim() != 1) {
      throw std::runtime_error("System pickle: expected 1D torch.uint8 buffer");
  }
  const uint8_t* ptr = t.data_ptr<uint8_t>();
  const size_t n = static_cast<size_t>(t.numel());
  return load_system_buffer(ptr, n);
}

} // namespace metatomic_torch
