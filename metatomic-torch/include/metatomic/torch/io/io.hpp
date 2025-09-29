#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace metatomic_torch {

// From system.hpp
class NeighborListOptionsHolder;
using NeighborListOptions = torch::intrusive_ptr<NeighborListOptionsHolder>;

class SystemHolder;
using System = torch::intrusive_ptr<SystemHolder>;

// ===== File-based =====
void   save_system_file(const std::string& mta_path, const System& system);
System load_system_file(const std::string& mta_path);
bool   is_system_mta_file(const std::string& mta_path);

// ===== In-memory (BytesIO-like) =====
std::vector<uint8_t> save_system_memory(const System& system);
System               load_system_memory(const uint8_t* data, size_t size);
inline System        load_system_memory(const std::vector<uint8_t>& buf) {
  return load_system_memory(buf.data(), buf.size());
}
bool   is_system_mta_memory(const uint8_t* data, size_t size);
inline bool is_system_mta_memory(const std::vector<uint8_t>& buf) {
  return is_system_mta_memory(buf.data(), buf.size());
}

} // namespace metatomic_torch
