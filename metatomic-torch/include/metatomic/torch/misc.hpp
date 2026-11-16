#ifndef METATOMIC_TORCH_MISC_HPP
#define METATOMIC_TORCH_MISC_HPP

#include <cstddef>
#include <string>
#include <torch/torch.h>

#include "metatomic/torch/system.hpp"

#include <torch/types.h>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

/// Get the runtime version of metatensor-torch as a string
METATOMIC_TORCH_EXPORT std::string version();

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
