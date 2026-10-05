#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <metatomic.hpp>

/// Plugin options
struct LennardJonesOptions {
    LennardJonesOptions(const std::map<std::string, std::string>& options);

    /// Length parameter of the 12-6 potential.
    double sigma;
    /// Depth of the potential well.
    double epsilon;
    /// Pair cutoff. The potential is shifted to zero at this distance.
    double cutoff;
    /// Atomic type reported in the model capabilities (not used to filter pairs).
    int32_t atomic_type;
    /// Unit of positions, cell, `sigma`, and `cutoff`.
    std::string length_unit;
    /// Unit of `epsilon` and of the returned energies.
    std::string energy_unit;
};

/// Shifted Lennard-Jones pair model used to test engines integration
class LennardJones final : public metatomic::BaseModel {
public:
    explicit LennardJones(LennardJonesOptions options);

    metatomic::ModelCapabilities capabilities() const override;

    metatomic::ModelMetadata metadata() const override;

    std::vector<metatomic::PairListOptions> requested_pair_lists() const override {
        return {pair_options_};
    }


    std::vector<metatomic::Quantity> requested_inputs() const override {
        /// No extra per-system data
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) override;

private:
    metatensor::TensorMap compute_energy_per_atom(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const metatomic::Quantity& request
    ) const;

    metatensor::TensorMap compute_energy_per_system(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const metatomic::Quantity& request
    ) const;

    LennardJonesOptions options_;
    metatomic::PairListOptions pair_options_;
};
