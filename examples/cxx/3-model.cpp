// .. _cxx-tutorial-model:
//
// Defining a custom model
// =======================
//
// This tutorial implements the same shifted Lennard-Jones model as the
// :ref:`C tutorial <c-tutorial-model>`, using the C++ API. We inherit from
// ``metatomic::BaseModel`` and return C++ objects from its virtual methods.
// The API handles serialization, error propagation, and object lifetimes.
//
// Our model computes the total energy of a single system and its gradient
// with respect to atomic positions. For each pair separated by a distance
// :math:`r` smaller than the cutoff :math:`r_c`, the energy is
//
// .. math::
//
//     E(r) = 4 \epsilon \left[
//         \left(\frac{\sigma}{r}\right)^{12}
//         - \left(\frac{\sigma}{r}\right)^6
//     \right] - E_{\mathrm{shift}},
//
// where :math:`E_{\mathrm{shift}}` is the unshifted potential at :math:`r_c`.
// This makes the energy zero at the cutoff, but does not smooth the forces.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <metatomic.hpp>

// %%
// Describing the model
// --------------------
//
// The model stores its parameters and the options identifying the pair list
// it needs. A half list contains each pair once, so we can sum pair energies
// without a factor of two. The engine is responsible for computing this list
// and attaching it to the system before execution.

class LennardJones final: public metatomic::BaseModel {
public:
    LennardJones(double sigma, double epsilon, double cutoff):
        sigma_(sigma), epsilon_(epsilon), cutoff_(cutoff),
        pair_options_(metatomic::PairListOptions::builder()
            .cutoff(cutoff)
            .full_list(false)
            .strict(true)
            .build())
    {
        if (!std::isfinite(sigma) || sigma <= 0 ||
            !std::isfinite(epsilon) || epsilon <= 0) {
            throw metatomic::Error("sigma and epsilon must be finite and positive");
        }
        auto x6 = std::pow(sigma_ / cutoff_, 6);
        shift_ = 4 * epsilon_ * (x6 * x6 - x6);
    }

// %%
// ``metadata()`` describes the model for users. ``capabilities()`` tells
// simulation engines which outputs, atomic types, units, devices, and data
// types it supports. Here, type 1 represents hydrogen, lengths are in
// Angstrom, and energies are in eV. Advertising a ``positions`` gradient
// allows callers to request forces.

    metatomic::ModelMetadata metadata() const override {
        return metatomic::ModelMetadata::builder()
            .name("Lennard-Jones model for tutorials")
            .add_author("metatomic authors")
            .build();
    }

    metatomic::ModelCapabilities capabilities() const override {
        auto energy = metatomic::Quantity::builder()
            .name("energy")
            .unit("eV")
            .sample_kind(metatomic::SampleKind::System)
            .add_gradient(metatomic::Gradients::Positions)
            .build();

        return metatomic::ModelCapabilities::builder()
            .atomic_types({1})
            .interaction_range(cutoff_)
            .length_unit("Angstrom")
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float64)
            .add_output(energy)
            .build();
    }

// %%
// ``requested_pair_lists()`` returns the pair-list options for the requested
// outputs. Our model needs only one list and no additional per-system inputs,
// so ``requested_inputs()`` returns an empty vector.

    std::vector<metatomic::PairListOptions> requested_pair_lists(
        const std::vector<metatomic::Quantity>& requested_outputs
    ) const override {
        if (requested_outputs.empty()) {
            return {};
        }
        return {pair_options_};
    }

    std::vector<metatomic::Quantity> requested_inputs(
        const std::vector<metatomic::Quantity>& /*requested_outputs*/
    ) const override {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) override;

private:
    double sigma_, epsilon_, cutoff_, shift_;
    metatomic::PairListOptions pair_options_;
};

// %%
// Computing energy and gradients
// ------------------------------
//
// ``execute_inner()`` performs the actual calculation. As in the C tutorial,
// we support one system and no atom selection. Unsupported requests raise
// ``metatomic::Error``. Engines should call ``metatomic::execute_model()``
// instead of this method directly: it handles unit conversion and optional
// consistency checks around the calculation.

std::vector<metatensor::TensorMap> LennardJones::execute_inner(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const std::vector<metatomic::Quantity>& requested_outputs
) {
    if (selected_atoms.has_value()) {
        throw metatomic::Error("this model does not support selected_atoms");
    }
    if (requested_outputs.empty()) {
        return {};
    }
    if (systems.size() != 1) {
        throw metatomic::Error("this model only supports a single system");
    }
    if (requested_outputs.size() != 1 || requested_outputs[0].name() != "energy" ||
        requested_outputs[0].sample_kind() != metatomic::SampleKind::System) {
        throw metatomic::Error("this model only supports total energy");
    }
    for (auto gradient: requested_outputs[0].gradients()) {
        if (gradient != metatomic::Gradients::Positions) {
            throw metatomic::Error("this model only supports positions gradients");
        }
    }
    bool with_gradients = !requested_outputs[0].gradients().empty();

// %%
// We retrieve the pair list using the same options we requested earlier.
// Its values have shape ``(n_pairs, 3, 1)`` and contain displacement vectors
// from the first atom to the second, including periodic cell shifts. Its
// samples identify the two atoms and the three cell shifts in each pair.
// ``values<double>()`` gives a typed view of the data without copying it.

    const auto& system = systems[0];
    auto pairs = system.pairs(pair_options_);
    auto displacements = pairs.values<double>();
    auto samples = pairs.samples().values();

    double energy = 0;
    auto gradients = std::vector<double>(3 * system.size(), 0.0);
    for (size_t pair = 0; pair < pairs.samples().count(); pair++) {
        auto dx = displacements(pair, 0, 0);
        auto dy = displacements(pair, 1, 0);
        auto dz = displacements(pair, 2, 0);
        auto r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 0 || r2 >= cutoff_ * cutoff_) {
            continue;
        }

        auto x2 = sigma_ * sigma_ / r2;
        auto x6 = x2 * x2 * x2;
        auto x12 = x6 * x6;
        energy += 4 * epsilon_ * (x12 - x6) - shift_;

        if (with_gradients) {
            // dE/d(r^2), with r^2 = |position_second - position_first|^2
            auto dedr2 = 12 * epsilon_ / r2 * (x6 - 2 * x12);
            auto first = static_cast<size_t>(samples(pair, 0));
            auto second = static_cast<size_t>(samples(pair, 1));
            const double displacement[] = {dx, dy, dz};
            for (size_t xyz = 0; xyz < 3; xyz++) {
                auto gradient = -2 * dedr2 * displacement[xyz];
                gradients[3 * first + xyz] += gradient;
                gradients[3 * second + xyz] -= gradient;
            }
        }
    }

// %%
// Packaging the output
// --------------------
//
// The energy is stored in a single-block ``TensorMap`` with one sample (the
// system), one property (energy), and no component axes. ``SimpleDataArray``
// owns the numerical data; moving it into a block transfers ownership.

    auto properties = metatensor::Labels({"energy"}, {{0}});
    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{1, 1}, std::vector<double>{energy}
        ),
        metatensor::Labels({"system"}, {{0}}), {}, properties
    );

// %%
// When requested, the position gradient has shape ``(n_atoms, 3, 1)``. Each
// sample identifies its parent energy sample, system, and atom. The ``xyz``
// component labels identify the three spatial directions. These values are
// energy derivatives: forces are their negatives.

    if (with_gradients) {
        auto sample_values = std::vector<int32_t>();
        for (size_t atom = 0; atom < system.size(); atom++) {
            sample_values.insert(sample_values.end(), {0, 0, static_cast<int32_t>(atom)});
        }
        auto gradient_samples = metatensor::Labels(
            {"sample", "system", "atom"}, sample_values.data(), system.size()
        );
        auto gradient_block = metatensor::TensorBlock(
            std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{system.size(), 3, 1}, std::move(gradients)
            ),
            gradient_samples,
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})}, properties
        );
        block.add_gradient("positions", std::move(gradient_block));
    }

    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.push_back(std::move(block));
    auto outputs = std::vector<metatensor::TensorMap>();
    outputs.emplace_back(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
    return outputs;
}

// %%
// Running the model
// -----------------
//
// We will use two atoms separated by 1.3 Angstrom along *z*. To keep the
// example self-contained, the helper below creates this non-periodic system
// and attaches its single pair. In a simulation, the engine would construct
// the system and compute the pair lists requested by the model.

static metatomic::System make_two_atom_system(
    double distance, const metatomic::PairListOptions& options
);

// %%
// .. raw:: html
//
//   <details><summary>Creating the two-atom system</summary>

// Convert a SimpleDataArray to an owning DLPack tensor for System's constructor.
// The DLPack tensor keeps the array's data alive after this helper returns.
template <typename T>
static metatomic::DLPackTensor tensor(metatensor::SimpleDataArray<T> array) {
    return metatomic::DLPackTensor(array.as_dlpack(
        {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
    ));
}

static metatomic::System make_two_atom_system(
    double distance, const metatomic::PairListOptions& options
) {
    auto system = metatomic::System(
        "Angstrom",
        tensor(metatensor::SimpleDataArray<int32_t>({2}, {1, 1})),
        tensor(metatensor::SimpleDataArray<double>({2, 3}, {0, 0, 0, 0, 0, distance})),
        tensor(metatensor::SimpleDataArray<double>({3, 3}, 0.0)),
        tensor(metatensor::SimpleDataArray<bool>({3}, false))
    );

    auto pairs = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{1, 3, 1}, std::vector<double>{0, 0, distance}
        ),
        metatensor::Labels(
            {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            {{0, 1, 0, 0, 0}}
        ),
        {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
        metatensor::Labels({"distance"}, {{0}})
    );
    system.add_pairs(options, std::move(pairs));
    return system;
}

// %%
// .. raw:: html
//
//   </details>
//
// We construct the model with :math:`\sigma = 1` Angstrom,
// :math:`\epsilon = 1` eV, and :math:`r_c = 3` Angstrom. A ``Quantity``
// describes the requested energy and gradient. ``std::nullopt`` selects all
// atoms, and ``check_consistency=true`` enables checks of the inputs and
// outputs. The model remains owned by the caller throughout execution.

int main() {
    auto model = LennardJones(/*sigma=*/1.0, /*epsilon=*/1.0, /*cutoff=*/3.0);
    auto energy_request = metatomic::Quantity::builder()
        .name("energy")
        .unit("eV")
        .sample_kind(metatomic::SampleKind::System)
        .add_gradient(metatomic::Gradients::Positions)
        .build();

    auto pair_options = model.requested_pair_lists({energy_request});
    auto systems = std::vector<metatomic::System>();
    systems.push_back(make_two_atom_system(/*distance=*/1.3, pair_options[0]));
    auto outputs = metatomic::execute_model(
        model, systems, std::nullopt, {energy_request}, /*check_consistency=*/true
    );

// %%
// Finally, we extract the energy and negate the position gradients to obtain
// forces. The energy is approximately -0.651537 eV. The atoms attract, with
// forces of approximately +2.239980 and -2.239980 eV/Angstrom along *z*.

    auto block = outputs[0].block_by_id(0);
    auto energy = block.values<double>();
    auto gradient_block = block.gradient("positions");
    auto gradients = gradient_block.values<double>();

    assert(std::abs(energy(0, 0) - (-0.651537)) < 1e-6);
    assert(std::abs(-gradients(0, 2, 0) - 2.239980) < 1e-6);
    assert(std::abs(-gradients(1, 2, 0) - (-2.239980)) < 1e-6);
    for (size_t atom = 0; atom < 2; atom++) {
        assert(gradients(atom, 0, 0) == 0.0);
        assert(gradients(atom, 1, 0) == 0.0);
    }
    std::cout << "Energy: " << energy(0, 0) << " eV\n";
    std::cout << "Forces along z: " << -gradients(0, 2, 0) << ", "
              << -gradients(1, 2, 0) << " eV/Angstrom\n";
}
