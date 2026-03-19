#ifndef METATOMIC_TORCH_MISC_HPP
#define METATOMIC_TORCH_MISC_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <torch/torch.h>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/system.hpp"

#include <torch/types.h>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

/// Get the runtime version of metatensor-torch as a string
METATOMIC_TORCH_EXPORT std::string version();

/// Select the best device according to the list of `model_devices` from a
/// model, the user-provided `desired_device` and what's available on the
/// current machine.
///
/// If `desired_device` is provided, it is checked against the `model_devices`
/// and the machine availability. If it contains a device index (e.g. "cuda:1"),
/// the base device type ("cuda") is used for these checks.
///
/// This function returns a c10::DeviceType (torch::DeviceType). It does NOT
/// decide a device index — callers that need a full torch::Device should
/// construct one from the returned DeviceType (and choose an index explicitly).
/// Or let it default away to zero via Device(DeviceType)
METATOMIC_TORCH_EXPORT torch::DeviceType pick_device(
	std::vector<std::string> model_devices,
	torch::optional<std::string> desired_device = torch::nullopt
);

/// Pick the output for the given `requested_output` from the availabilities of the
/// model's `outputs`, according to the optional `desired_variant`.
METATOMIC_TORCH_EXPORT std::string pick_output(
	std::string requested_output,
	torch::Dict<std::string, ModelOutput> outputs,
	torch::optional<std::string> desired_variant = torch::nullopt
);

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
  const auto n = static_cast<size_t>(t.numel());
  return load_system_buffer(ptr, n);
}

namespace details {

/// Validate that the given `name` is valid for a model output/input
///
/// The function returns a tuple with:
/// - a boolean indicating whether this is a known output/input
/// - the name of the base output/input (empty if custom)
/// - the name of the variant (empty if none)
std::tuple<bool, std::string, std::string> validate_name_and_check_variant(
    const std::string& name
);
}

}

#endif
