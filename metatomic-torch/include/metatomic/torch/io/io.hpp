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
std::vector<uint8_t> save_buffer(const System& system);
System               load_system_buffer(const uint8_t* data, size_t size);
inline System        load_system_buffer(const std::vector<uint8_t>& buf) {
  return load_system_buffer(buf.data(), buf.size());
}

} // namespace metatomic_torch
