#ifndef METATOMIC_TORCH_MISC_HPP
#define METATOMIC_TORCH_MISC_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <torch/torch.h>

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

/// Get the multiplicative conversion factor to use to convert from
/// `from_unit` to `to_unit`. Both units are parsed as expressions (e.g.
/// `kJ / mol / A^2`, `(eV * u)^(1/2)`) and their dimensions must match.
///
/// @verbatim embed:rst:leading-slashes
///
/// .. seealso::
///
///     The general documentation for :ref:`units`, with the expression
///     syntax and list of supported base units.
///
/// @endverbatim
METATOMIC_TORCH_EXPORT double unit_conversion_factor(
    const std::string& from_unit,
    const std::string& to_unit
);

/// Save a `System` to a file at the given `path`.
METATOMIC_TORCH_EXPORT void save(const std::string& path, const System& system);

/// Load a `System` from a file at the given `path`.
METATOMIC_TORCH_EXPORT System load(const std::string& path);

/// Save a `System` to an in-memory buffer.
METATOMIC_TORCH_EXPORT torch::Tensor save_buffer(const System& system);

/// Load a `System` from an in-memory buffer, represented as a 1D torch.uint8
/// tensor on CPU.
METATOMIC_TORCH_EXPORT System load_buffer(torch::Tensor buffer);

} // namespace metatomic_torch

#endif
