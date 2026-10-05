#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <metatomic.hpp>

#include "./lennard_jones.hpp"

//============================================================================//
//                          LennardJones options                              //
//============================================================================//

/// Locale-independent parse of a decimal floating-point string (`.` separator).
static double parse_double(const std::string& value, const std::string& name) {
    // std::stod follows LC_NUMERIC; "1.0" then fails on a comma-decimal locale.
    std::istringstream in(value);
    in.imbue(std::locale::classic());
    double result = 0.0;
    in >> std::noskipws >> result;
    if (!in || in.get() != std::char_traits<char>::eof()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be a number");
    }
    return result;
}

/// Parse a base-10 integer and reject anything that does not fit in `int32_t`.
static int32_t parse_int32(const std::string& value, const std::string& name) {
    size_t parsed = 0;
    int64_t result = 0;
    try {
        result = std::stoll(value, &parsed);
    } catch (const std::exception &) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be an integer");
    }

    if (parsed != value.size()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be an integer");
    }

    if (result < std::numeric_limits<int32_t>::min() || result > std::numeric_limits<int32_t>::max()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' is out of range");
    }

    return static_cast<int32_t>(result);
}

LennardJonesOptions::LennardJonesOptions(const std::map<std::string, std::string>& options) {
    const std::set<std::string> REQUIRED_KEYS = {
        "sigma", "epsilon", "cutoff", "atomic_type", "length_unit", "energy_unit"
    };

    size_t required_found = 0;
    for (const auto& item : options) {
        if (REQUIRED_KEYS.find(item.first) != REQUIRED_KEYS.end()) {
            required_found++;
        } else {
            throw metatomic::Error("unknown Lennard-Jones option: '" + item.first + "'");
        }
    }

    if (required_found != REQUIRED_KEYS.size()) {
        throw metatomic::Error(
            "missing required Lennard-Jones options: 'sigma', 'epsilon', "
            "'cutoff', and 'atomic_type' must all be provided"
        );
    }

    this->sigma = parse_double(options.at("sigma"), "sigma");
    if (!std::isfinite(this->sigma) || this->sigma <= 0.0) {
        throw metatomic::Error( "Lennard-Jones option 'sigma' must be finite and positive");
    }

    this->epsilon = parse_double(options.at("epsilon"), "epsilon");
    if (!std::isfinite(this->epsilon) || this->epsilon <= 0.0) {
        throw metatomic::Error( "Lennard-Jones option 'epsilon' must be finite and positive");
    }

    this->cutoff = parse_double(options.at("cutoff"), "cutoff");
    if (!std::isfinite(this->cutoff) || this->cutoff <= 0.0) {
        throw metatomic::Error( "Lennard-Jones option 'cutoff' must be finite and positive");
    }

    this->atomic_type = parse_int32(options.at("atomic_type"), "atomic_type");
    this->length_unit = options.at("length_unit");
    this->energy_unit = options.at("energy_unit");
}

//============================================================================//
//                          Base model functions                              //
//============================================================================//

LennardJones::LennardJones(LennardJonesOptions options):
    options_(std::move(options)),
    pair_options_(metatomic::PairListOptions::builder()
        .cutoff(options_.cutoff)
        .full_list(false)
        .strict(false)
        .build())
{}


metatomic::ModelCapabilities LennardJones::capabilities() const {
    // This model can compute either system or per-atom energies, with
    // positions gradients for the system energy.
    auto energy = metatomic::Quantity::builder()
        .name("energy")
        .unit(options_.energy_unit)
        .sample_kind(metatomic::SampleKind::System)
        .add_gradient(metatomic::Gradients::Positions)
        .build();

    auto atomic_energy = metatomic::Quantity::builder()
        .name("energy")
        .unit(options_.energy_unit)
        .sample_kind(metatomic::SampleKind::Atom)
        .build();

    return metatomic::ModelCapabilities::builder()
        .atomic_types({options_.atomic_type})
        .interaction_range(options_.cutoff)
        .length_unit(options_.length_unit)
        .supported_devices({metatomic::ModelCapabilities::Device::CPU})
        .dtype(metatomic::ModelCapabilities::DType::Float64)
        .add_output(energy)
        .add_output(atomic_energy)
        .build();
}

metatomic::ModelMetadata LennardJones::metadata() const {
    return metatomic::ModelMetadata::builder()
        .name("Lennard-Jones test model")
        .add_author("metatomic authors")
        .description("Shifted Lennard-Jones pair potential for engine tests")
        .build();
}

std::vector<metatensor::TensorMap> LennardJones::execute_inner(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const std::vector<metatomic::Quantity>& requested_outputs
) {
    if (requested_outputs.empty()) {
        return {};
    }

    auto results = std::vector<metatensor::TensorMap>();
    for (const auto& output : requested_outputs) {
        if (output.name() == "energy") {
            if (output.sample_kind() == metatomic::SampleKind::Atom) {
                auto energy = this->compute_energy_per_atom(systems, selected_atoms, output);
                results.emplace_back(std::move(energy));
            } else if (output.sample_kind() == metatomic::SampleKind::System) {
                auto energy = this->compute_energy_per_system(systems, selected_atoms, output);
                results.emplace_back(std::move(energy));
            } else {
                throw metatomic::Error(
                    "Lennard-Jones energy must use system or atom samples");
            }
        } else {
            throw metatomic::Error("unsupported output requested: '" + output.name() + "'");
        }
    }
    return results;
}

//============================================================================//
//                     Actual outputs implementations                         //
//============================================================================//

metatensor::TensorMap LennardJones::compute_energy_per_atom(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const metatomic::Quantity& request
) const {
    assert(request.name() == "energy");
    assert(request.sample_kind() == metatomic::SampleKind::Atom);

    if (!request.gradients().empty()) {
        throw metatomic::Error("gradients are not supported for per-atom energy");
    }

    // create the samples for this output
    auto energy_samples = metatensor::Labels({"_"});
    if (selected_atoms.has_value()) {
        energy_samples = selected_atoms.value();
    } else {
        auto samples_values = std::vector<int32_t>();
        for (size_t system_i = 0; system_i < systems.size(); system_i++) {
            for (size_t atom_i = 0; atom_i < systems[system_i].size(); atom_i++) {
                samples_values.push_back(static_cast<int32_t>(system_i));
                samples_values.push_back(static_cast<int32_t>(atom_i));
            }
        }

        auto shape = std::vector<uintptr_t>{samples_values.size() / 2, 2};
        auto samples_array = metatensor::DataArrayBase::to_mts_array(
            std::make_unique<metatensor::SimpleDataArray<int32_t>>(
                shape, std::move(samples_values)
            )
        );
        energy_samples = metatensor::Labels(
            {"system", "atom"},
            samples_array,
            metatensor::assume_unique{}
        );
    }

    // allocate the output values and gradients
    auto energy_values = std::vector<double>();
    energy_values.resize(energy_samples.count(), 0.0);


    // compute the energy for each system
    for (size_t system_i = 0; system_i < systems.size(); system_i++) {
        const auto& system = systems[system_i];

        auto pairs = system.pairs(pair_options_);
        auto displacements = pairs.values<double>();
        auto pair_samples = pairs.samples().values();
        auto n_pairs = displacements.shape()[0];

        auto cutoff_2 = options_.cutoff * options_.cutoff;
        auto sigma_2 = options_.sigma * options_.sigma;
        auto sigma_cutoff_2 = sigma_2 / cutoff_2;
        auto sigma_cutoff_6 = sigma_cutoff_2 * sigma_cutoff_2 * sigma_cutoff_2;
        auto shift = 4.0 * options_.epsilon * (sigma_cutoff_6 * sigma_cutoff_6 - sigma_cutoff_6);

        for (size_t pair = 0; pair < n_pairs; pair++) {
            auto first = pair_samples(pair, 0);
            auto second = pair_samples(pair, 1);

            auto first_position = energy_samples.position({static_cast<int32_t>(system_i), first});
            auto second_position = energy_samples.position({static_cast<int32_t>(system_i), second});

            if (!first_position && !second_position) {
                // Neither endpoint is selected, so skip this pair.
                continue;
            }

            auto dx = displacements(pair, 0, 0);
            auto dy = displacements(pair, 1, 0);
            auto dz = displacements(pair, 2, 0);
            auto distance_2 = dx * dx + dy * dy + dz * dz;

            if (distance_2 >= cutoff_2) {
                continue;
            }

            auto sigma_r_2 = sigma_2 / distance_2;
            auto sigma_r_6 = sigma_r_2 * sigma_r_2 * sigma_r_2;
            auto sigma_r_12 = sigma_r_6 * sigma_r_6;
            auto pair_energy = 4.0 * options_.epsilon * (sigma_r_12 - sigma_r_6) - shift;

            if (first_position) {
                energy_values[*first_position] += 0.5 * pair_energy;
            }

            if (second_position) {
                energy_values[*second_position] += 0.5 * pair_energy;
            }
        }
    }

    // create the output tensor map
    auto shape = std::vector<uintptr_t>{energy_values.size(), 1};
    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            shape, std::move(energy_values)
        ),
        energy_samples,
        {},
        metatensor::Labels({"energy"}, {{0}})
    );
    auto keys = metatensor::Labels({"_"}, {{0}});
    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.emplace_back(std::move(block));
    return metatensor::TensorMap(keys, std::move(blocks));
}

metatensor::TensorMap LennardJones::compute_energy_per_system(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const metatomic::Quantity& request
) const {
    assert(request.name() == "energy");
    assert(request.sample_kind() == metatomic::SampleKind::System);

    bool do_positions_gradients = false;
    for (auto gradient: request.gradients()) {
        if (gradient == metatomic::Gradients::Positions) {
            do_positions_gradients = true;
        } else {
            throw metatomic::Error("only positions gradients are supported");
        }
    }

    // create the samples for this output
    auto sample_values = std::vector<int32_t>();
    sample_values.reserve(systems.size());
    for (size_t system_i = 0; system_i < systems.size(); system_i++) {
        sample_values.push_back(static_cast<int32_t>(system_i));
    }

    auto shape = std::vector<uintptr_t>{sample_values.size(), 1};
    auto samples_array = metatensor::DataArrayBase::to_mts_array(
        std::make_unique<metatensor::SimpleDataArray<int32_t>>(
            shape, std::move(sample_values)
        )
    );
    auto energy_samples = metatensor::Labels(
        {"system"},
        std::move(samples_array),
        metatensor::assume_unique{}
    );

    // allocate the output values and gradients
    auto energy_values = std::vector<double>();
    energy_values.resize(energy_samples.count(), 0.0);

    size_t total_atoms = 0;
    for (const auto& system : systems) {
        total_atoms += system.size();
    }

    auto positions_gradients = std::vector<double>();
    if (do_positions_gradients) {
        positions_gradients.resize(3 * total_atoms, 0.0);
    }

    // compute the energy and gradients for each system
    size_t system_offset = 0;
    for (size_t system_i = 0; system_i < systems.size(); system_i++) {
        const auto& system = systems[system_i];

        auto pairs = system.pairs(pair_options_);
        auto displacements = pairs.values<double>();
        auto pair_samples = pairs.samples().values();
        auto n_pairs = displacements.shape()[0];

        auto cutoff_2 = options_.cutoff * options_.cutoff;
        auto sigma_2 = options_.sigma * options_.sigma;
        auto sigma_cutoff_2 = sigma_2 / cutoff_2;
        auto sigma_cutoff_6 = sigma_cutoff_2 * sigma_cutoff_2 * sigma_cutoff_2;
        auto sigma_cutoff_12 = sigma_cutoff_6 * sigma_cutoff_6;
        auto shift = 4.0 * options_.epsilon * (sigma_cutoff_12 - sigma_cutoff_6);

        for (size_t pair = 0; pair < n_pairs; pair++) {
            auto first = pair_samples(pair, 0);
            auto second = pair_samples(pair, 1);

            auto scale = 0.0;
            if (selected_atoms.has_value()) {
                auto first_position = selected_atoms->position({static_cast<int32_t>(system_i), first});
                auto second_position = selected_atoms->position({static_cast<int32_t>(system_i), second});

                if (!first_position && !second_position) {
                    // Neither endpoint is selected, so skip this pair.
                    continue;
                }

                if (first_position) {
                    scale += 0.5;
                }
                if (second_position) {
                    scale += 0.5;
                }
            } else {
                scale = 1.0;
            }

            auto dx = displacements(pair, 0, 0);
            auto dy = displacements(pair, 1, 0);
            auto dz = displacements(pair, 2, 0);
            auto distance_2 = dx * dx + dy * dy + dz * dz;

            if (distance_2 >= cutoff_2) {
                continue;
            }

            auto sigma_r_2 = sigma_2 / distance_2;
            auto sigma_r_6 = sigma_r_2 * sigma_r_2 * sigma_r_2;
            auto sigma_r_12 = sigma_r_6 * sigma_r_6;
            auto pair_energy = 4.0 * options_.epsilon * (sigma_r_12 - sigma_r_6) - shift;

            energy_values[system_i] += scale * pair_energy;

            if (do_positions_gradients) {
                // Analytical forces from the shifted LJ potential
                auto energy_derivative = 12.0 * options_.epsilon / distance_2 * (sigma_r_6 - 2.0 * sigma_r_12);
                const double displacement[3] = {dx, dy, dz};
                for (size_t xyz = 0; xyz < 3; xyz++) {
                    auto gradient = -2.0 * scale * energy_derivative * displacement[xyz];

                    auto first_offset = system_offset + static_cast<size_t>(first);
                    auto second_offset = system_offset + static_cast<size_t>(second);

                    positions_gradients[3 * first_offset + xyz] += gradient;
                    positions_gradients[3 * second_offset + xyz] -= gradient;
                }
            }
        }

        system_offset += system.size();
    }

    // create the output tensor map
    shape = std::vector<uintptr_t>{energy_values.size(), 1};
    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            shape, std::move(energy_values)
        ),
        energy_samples,
        {},
        metatensor::Labels({"energy"}, {{0}})
    );

    if (do_positions_gradients) {
        auto sample_values = std::vector<int32_t>();
        sample_values.reserve(3 * total_atoms);
        for (size_t system_i = 0; system_i < systems.size(); system_i++) {
            for (size_t atom_i = 0; atom_i < systems[system_i].size(); atom_i++) {
                sample_values.push_back(static_cast<int32_t>(system_i));
                sample_values.push_back(static_cast<int32_t>(system_i));
                sample_values.push_back(static_cast<int32_t>(atom_i));
            }
        }

        auto shape = std::vector<uintptr_t>{sample_values.size() / 3, 3};
        auto samples_array = metatensor::DataArrayBase::to_mts_array(
            std::make_unique<metatensor::SimpleDataArray<int32_t>>(
                shape, std::move(sample_values)
            )
        );

        auto gradients_samples = metatensor::Labels(
            {"sample", "system", "atom"},
            std::move(samples_array),
            metatensor::assume_unique{}
        );

        shape = std::vector<uintptr_t>{positions_gradients.size() / 3, 3, 1};
        auto gradient_block = metatensor::TensorBlock(
            std::make_unique<metatensor::SimpleDataArray<double>>(
                shape, std::move(positions_gradients)
            ),
            gradients_samples,
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            metatensor::Labels({"energy"}, {{0}})
        );
        block.add_gradient("positions", std::move(gradient_block));
    }

    auto keys = metatensor::Labels({"_"}, {{0}});
    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.emplace_back(std::move(block));
    return metatensor::TensorMap(keys, std::move(blocks));
}

//============================================================================//
//                             Plugin registration                            //
//============================================================================//

/// Plugin entry point.
std::unique_ptr<metatomic::BaseModel> load_model(
    const std::string& load_from,
    const std::map<std::string, std::string>& options
) {
    if (load_from != "metatomic-lj-model") {
        // not the model this plugin can load
        return nullptr;
    }

    auto lj_options = LennardJonesOptions(options);
    return std::make_unique<LennardJones>(std::move(lj_options));
}

MTA_REGISTER_CXX_PLUGIN("metatomic-lj-plugin", load_model);
