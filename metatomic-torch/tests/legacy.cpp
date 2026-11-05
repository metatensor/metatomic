#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

#include <metatensor.hpp>
#include <metatomic.hpp>
#include <metatomic/torch.hpp>
using namespace metatomic_torch;

#include <catch.hpp>
using namespace Catch::Matchers;

// ========================================================================== //
// Build TorchScript `AtomisticModel` in C++. `Module::define` only knows about
// builtin types by default, this resolver makes the metatensor and metatomic
// classes available under their short name (`Labels`, `System`, ...).

namespace {
class ClassesResolver final: public torch::jit::Resolver {
public:
    std::shared_ptr<torch::jit::SugaredValue> resolveValue(
        const std::string& name,
        torch::jit::GraphFunction& function,
        const torch::jit::SourceRange& location
    ) override {
        if (auto type = find(name)) {
            return std::make_shared<torch::jit::ClassValue>(type);
        }
        return torch::jit::nativeResolver()->resolveValue(name, function, location);
    }

    torch::jit::TypePtr resolveType(const std::string& name, const torch::jit::SourceRange& location) override {
        if (auto type = find(name)) {
            return type;
        }
        return torch::jit::nativeResolver()->resolveType(name, location);
    }

private:
    static c10::ClassTypePtr find(const std::string& name) {
        for (const auto* ns: {"metatensor", "metatomic"}) {
            auto type = torch::getCustomClass(std::string("__torch__.torch.classes.") + ns + "." + name);
            if (type) {
                return type;
            }
        }
        return nullptr;
    }
};
}

static void define(torch::jit::Module& module, const std::string& source) {
    module.define(source, std::make_shared<ClassesResolver>());
}

template <typename T>
static c10::TypePtr class_type() {
    return c10::getCustomClassType<c10::intrusive_ptr<T>>();
}

/// The forward function of all the inner modules
static const char* FORWARD_SIGNATURE = R"(
def forward(
    self,
    systems: List[System],
    outputs: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels],
) -> Dict[str, TensorMap]:
    self.calls += 1
)";

/// E_i = k/4 * sum_j |r_ij|^2, over the pairs i-j in a half neighbor list.
/// This can compute per-atom and per-system energies, and supports
/// `selected_atoms`. Gradients come from autograd.
static torch::jit::Module harmonic_pairs(double k, double cutoff) {
    auto module = torch::jit::Module("__torch__.HarmonicPairs");
    module.register_attribute("calls", c10::IntType::get(), int64_t{0});
    module.register_attribute("k", c10::FloatType::get(), k);
    module.register_attribute(
        "options",
        class_type<PairListOptionsHolder>(),
        torch::make_intrusive<PairListOptionsHolder>(cutoff, /*full_list*/ false, /*strict*/ true)
    );

    define(module, std::string(FORWARD_SIGNATURE) + R"(
    samples: List[Tensor] = []
    energies: List[Tensor] = []
    for i, system in enumerate(systems):
        pairs = system.pairs(self.options)
        half = 0.25 * self.k * (pairs.values ** 2).sum(dim=[1, 2])
        energy = half.new_zeros([len(system)])
        energy = energy.index_add(0, pairs.samples.column("first_atom"), half)
        energy = energy.index_add(0, pairs.samples.column("second_atom"), half)

        atoms = torch.arange(len(system))
        if selected_atoms is not None:
            selected = selected_atoms.values
            atoms = selected[selected[:, 0] == i][:, 1].to(atoms.dtype)

        samples.append(torch.stack([torch.full_like(atoms, i), atoms], dim=1))
        energies.append(energy[atoms])

    zero = torch.tensor([[0]])
    if outputs["energy"].sample_kind == "atom":
        values = torch.cat(energies).reshape(-1, 1)
        labels = Labels(["system", "atom"], torch.cat(samples))
    else:
        values = torch.stack([e.sum() for e in energies]).reshape(-1, 1)
        labels = Labels(["system"], torch.arange(len(systems)).reshape(-1, 1))

    block = TensorBlock(values, labels, [], Labels(["energy"], zero))
    return {"energy": TensorMap(Labels(["_"], zero), [block])}
)");

    return module;
}

/// E = sum_i m_i |r_i|^2, with explicit positions gradients, using the
/// masses given by the engine as custom data. `offset` is added to the
/// explicit gradients, to check they are used as-is.
static torch::jit::Module harmonic(double offset) {
    auto module = torch::jit::Module("__torch__.Harmonic");
    module.register_attribute("calls", c10::IntType::get(), int64_t{0});
    module.register_attribute("offset", c10::FloatType::get(), offset);

    define(module, std::string(FORWARD_SIGNATURE) + R"(
    masses: List[Tensor] = []
    energies: List[Tensor] = []
    for system in systems:
        mass = system.custom_data("mass").block_by_id(0).values
        masses.append(mass)
        energies.append((mass * system.positions ** 2).sum())

    zero = torch.tensor([[0]])
    block = TensorBlock(
        torch.stack(energies).reshape(-1, 1),
        Labels(["system"], torch.arange(len(systems)).reshape(-1, 1)),
        [],
        Labels(["energy"], zero),
    )

    if "positions" in outputs["energy"].explicit_gradients:
        samples: List[Tensor] = []
        values: List[Tensor] = []
        for i, system in enumerate(systems):
            atoms = torch.arange(len(system))
            system_i = torch.full_like(atoms, i)
            samples.append(torch.stack([system_i, system_i, atoms], dim=1))
            values.append(2.0 * masses[i] * system.positions.detach() + self.offset)

        gradient = TensorBlock(
            torch.cat(values).reshape(-1, 3, 1),
            Labels(["sample", "system", "atom"], torch.cat(samples)),
            [Labels(["xyz"], torch.tensor([[0], [1], [2]]))],
            block.properties,
        )
        block.add_gradient("positions", gradient)

    return {"energy": TensorMap(Labels(["_"], zero), [block])}
)");

    return module;
}

/// Wrap `module` in a TorchScript module with the same API as
/// `metatomic.torch.AtomisticModel`
static torch::jit::Module atomistic_model(
    torch::jit::Module module,
    ModelCapabilities capabilities,
    std::vector<PairListOptions> pair_lists,
    c10::Dict<std::string, ModelOutput> inputs
) {
    auto model = torch::jit::Module("__torch__.AtomisticModel");
    model.register_module("module", std::move(module));

    auto names = c10::List<std::string>();
    for (const auto& item: capabilities->outputs()) {
        names.push_back(item.key());
    }
    model.register_attribute("_model_capabilities_outputs_names", c10::ListType::ofStrings(), names);

    auto metadata = torch::make_intrusive<ModelMetadataHolder>(
        "test model",
        "a model for the LegacyModel tests",
        std::vector<std::string>{"metatomic developers"},
        torch::Dict<std::string, std::vector<std::string>>(),
        torch::Dict<std::string, std::string>()
    );
    model.register_attribute("_capabilities", class_type<ModelCapabilitiesHolder>(), capabilities);
    model.register_attribute("_metadata", class_type<ModelMetadataHolder>(), metadata);

    auto pair_lists_list = c10::List<PairListOptions>();
    for (auto& options: pair_lists) {
        pair_lists_list.push_back(std::move(options));
    }
    model.register_attribute(
        "_pair_lists", c10::ListType::create(class_type<PairListOptionsHolder>()), pair_lists_list
    );
    model.register_attribute(
        "_inputs", c10::DictType::create(c10::StringType::get(), class_type<ModelOutputHolder>()), inputs
    );

    define(model, R"(
def capabilities(self) -> ModelCapabilities:
    return self._capabilities

def metadata(self) -> ModelMetadata:
    return self._metadata

def requested_neighbor_lists(self) -> List[NeighborListOptions]:
    return self._pair_lists

def requested_inputs(self, use_new_names: bool = False) -> Dict[str, ModelOutput]:
    return self._inputs
)");

    return model;
}

static ModelCapabilities capabilities(
    const std::string& output_unit,
    const std::string& sample_kind,
    std::vector<std::string> explicit_gradients,
    const std::string& length_unit,
    double interaction_range
) {
    auto outputs = torch::Dict<std::string, ModelOutput>();
    outputs.insert("energy", torch::make_intrusive<ModelOutputHolder>(
        /*quantity*/ "", output_unit, sample_kind, std::move(explicit_gradients), /*description*/ ""
    ));

    return torch::make_intrusive<ModelCapabilitiesHolder>(
        outputs,
        std::vector<int64_t>{1, 8},
        interaction_range,
        length_unit,
        std::vector<std::string>{"cpu", "mps"},
        "float64"
    );
}

static const double CUTOFF = 3.5;
static const double K = 2.0;

static torch::jit::Module harmonic_pairs_model() {
    return atomistic_model(
        harmonic_pairs(K, CUTOFF),
        capabilities("eV", "atom", {}, "angstrom", CUTOFF),
        {torch::make_intrusive<PairListOptionsHolder>(CUTOFF, /*full_list*/ false, /*strict*/ true)},
        c10::Dict<std::string, ModelOutput>()
    );
}

static torch::jit::Module harmonic_model(double offset, const std::string& length_unit = "angstrom") {
    auto inputs = c10::Dict<std::string, ModelOutput>();
    inputs.insert("mass", torch::make_intrusive<ModelOutputHolder>(
        /*quantity*/ "", "u", "atom", std::vector<std::string>{}, /*description*/ ""
    ));

    return atomistic_model(
        harmonic(offset),
        capabilities("eV", "system", {"positions"}, length_unit, 0.0),
        {},
        inputs
    );
}

// ========================================================================== //
// Systems with data stored in metatensor-core arrays, as an engine would
// create them

template <typename T>
static metatomic::DLPackTensor dlpack(std::vector<uintptr_t> shape, std::vector<T> data) {
    auto array = metatensor::DataArrayBase::to_mts_array(
        std::make_unique<metatensor::SimpleDataArray<T>>(std::move(shape), std::move(data))
    );
    return metatomic::DLPackTensor(array.as_dlpack(
        {kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
    ));
}

static metatomic::DLPackTensor no_pbc() {
    // `SimpleDataArray<bool>` does not work with `std::vector<bool>`, use
    // `uint8_t` and change the dtype instead
    auto array = metatensor::DataArrayBase::to_mts_array(
        std::make_unique<metatensor::SimpleDataArray<uint8_t>>(
            std::vector<uintptr_t>{3}, std::vector<uint8_t>{0, 0, 0}
        )
    );
    auto* tensor = array.as_dlpack({kDLCPU, 0}, nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION});
    tensor->dl_tensor.dtype.code = kDLBool;
    return metatomic::DLPackTensor(tensor);
}

using Positions = std::vector<std::array<double, 3>>;

/// Positions of a small distorted cluster, in Angstrom
static Positions cluster(double scale = 1.0) {
    auto positions = Positions();
    for (size_t i = 0; i < 6; i++) {
        auto x = static_cast<double>(i);
        positions.push_back({
            scale * (1.1 * x + 0.1 * std::sin(x)),
            scale * (0.7 * std::cos(x)),
            scale * (0.4 * x * (x - 3.0) / 3.0),
        });
    }
    return positions;
}

/// All pairs i < j closer than `cutoff`, with their vectors r_j - r_i
struct Pairs {
    std::vector<std::array<size_t, 2>> atoms;
    std::vector<std::array<double, 3>> vectors;
};

static Pairs find_pairs(const Positions& positions, double cutoff) {
    auto result = Pairs();
    for (size_t i = 0; i < positions.size(); i++) {
        for (size_t j = i + 1; j < positions.size(); j++) {
            auto vector = std::array<double, 3>{};
            auto distance2 = 0.0;
            for (size_t d = 0; d < 3; d++) {
                vector[d] = positions[j][d] - positions[i][d];
                distance2 += vector[d] * vector[d];
            }
            if (distance2 < cutoff * cutoff) {
                result.atoms.push_back({i, j});
                result.vectors.push_back(vector);
            }
        }
    }
    return result;
}

/// Create a system in Angstrom, with the given pairs lists and masses
static metatomic::System system(
    const Positions& positions,
    const std::vector<metatomic::PairListOptions>& pair_lists = {},
    const std::vector<double>& masses = {}
) {
    auto n_atoms = positions.size();
    auto types = std::vector<int32_t>(n_atoms, 1);
    types[0] = 8;
    auto flat_positions = std::vector<double>();
    for (const auto& position: positions) {
        flat_positions.insert(flat_positions.end(), position.begin(), position.end());
    }

    auto result = metatomic::System(
        "angstrom",
        dlpack<int32_t>({n_atoms}, std::move(types)),
        dlpack<double>({n_atoms, 3}, std::move(flat_positions)),
        dlpack<double>({3, 3}, std::vector<double>(9, 0.0)),
        no_pbc()
    );

    for (const auto& options: pair_lists) {
        // the cutoff is in the model units, which are Angstrom in these tests
        auto pairs = find_pairs(positions, options.cutoff());
        auto samples = std::vector<int32_t>();
        auto vectors = std::vector<double>();
        for (size_t p = 0; p < pairs.atoms.size(); p++) {
            samples.insert(samples.end(), {
                static_cast<int32_t>(pairs.atoms[p][0]), static_cast<int32_t>(pairs.atoms[p][1]), 0, 0, 0
            });
            vectors.insert(vectors.end(), pairs.vectors[p].begin(), pairs.vectors[p].end());
        }

        auto n_pairs = pairs.atoms.size();
        result.add_pairs(options, metatensor::TensorBlock(
            std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{n_pairs, 3, 1}, std::move(vectors)
            ),
            metatensor::Labels(
                {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
                samples.data(), n_pairs
            ),
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            metatensor::Labels({"distance"}, {{0}})
        ));
    }

    if (!masses.empty()) {
        auto samples = std::vector<int32_t>();
        for (size_t i = 0; i < n_atoms; i++) {
            samples.insert(samples.end(), {0, static_cast<int32_t>(i)});
        }
        auto block = metatensor::TensorBlock(
            std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{n_atoms, 1}, masses
            ),
            metatensor::Labels({"system", "atom"}, samples.data(), n_atoms),
            {},
            metatensor::Labels({"mass"}, {{0}})
        );

        auto blocks = std::vector<metatensor::TensorBlock>();
        blocks.push_back(std::move(block));
        result.add_custom_data("mass", metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks)));
    }

    return result;
}

/// Energies, positions and strain gradients of `HarmonicPairs`
struct Reference {
    std::vector<double> energies;
    std::vector<std::array<double, 3>> positions_gradient;
    std::array<std::array<double, 3>, 3> strain_gradient;
};

static Reference harmonic_pairs_reference(const Positions& positions) {
    auto pairs = find_pairs(positions, CUTOFF);
    auto result = Reference{
        std::vector<double>(positions.size(), 0.0),
        std::vector<std::array<double, 3>>(positions.size(), {0.0, 0.0, 0.0}),
        {},
    };

    for (size_t p = 0; p < pairs.atoms.size(); p++) {
        auto [i, j] = pairs.atoms[p];
        const auto& vector = pairs.vectors[p];
        auto half = 0.25 * K * (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
        result.energies[i] += half;
        result.energies[j] += half;
        for (size_t a = 0; a < 3; a++) {
            result.positions_gradient[i][a] -= K * vector[a];
            result.positions_gradient[j][a] += K * vector[a];
            for (size_t b = 0; b < 3; b++) {
                result.strain_gradient[a][b] += K * vector[a] * vector[b];
            }
        }
    }

    return result;
}

static double sum(const std::vector<double>& values) {
    auto result = 0.0;
    for (auto value: values) {
        result += value;
    }
    return result;
}

static metatomic::Quantity energy(
    const std::string& sample_kind = "system",
    std::vector<metatomic::Gradients> gradients = {},
    const std::string& unit = "eV"
) {
    return metatomic::Quantity::builder()
        .name("energy")
        .unit(unit)
        .sample_kind(sample_kind == "atom" ? metatomic::SampleKind::Atom : metatomic::SampleKind::System)
        .gradients(std::move(gradients))
        .build();
}

/// Number of times the inner module was called
static int64_t calls(const torch::jit::Module& inner) {
    return inner.attr("calls").toInt();
}

static std::vector<std::string> names(const metatensor::Labels& labels) {
    auto result = std::vector<std::string>();
    for (const auto* name: labels.names()) {
        result.emplace_back(name);
    }
    return result;
}

/// The first value in `block`
static double first_value(metatensor::TensorBlock& block) {
    auto values = block.values();
    return values(0, 0);
}

static double first_value(metatensor::TensorBlock&& block) {
    return first_value(block);
}

static const auto TOLERANCE = 1e-12;

// ========================================================================== //

TEST_CASE("LegacyModel") {
    using metatomic::Gradients;

    SECTION("capabilities and requests") {
        auto model = LegacyModel(harmonic_pairs_model());

        CHECK(model.metadata().name() == "test model");

        auto capabilities = model.capabilities();
        CHECK(capabilities.length_unit() == "angstrom");
        CHECK(capabilities.interaction_range() == CUTOFF);
        CHECK((capabilities.supported_devices() == std::vector<metatomic::ModelCapabilities::Device>{
            metatomic::ModelCapabilities::Device::CPU, metatomic::ModelCapabilities::Device::Metal
        }));
        CHECK((capabilities.dtype() == metatomic::ModelCapabilities::DType::Float64));
        CHECK(capabilities.atomic_types() == std::vector<int32_t>{1, 8});

        // per-atom outputs are also declared per-system, with autograd gradients
        const auto& outputs = capabilities.outputs();
        REQUIRE(outputs.size() == 2);
        CHECK(outputs[0].name() == "energy");
        CHECK((outputs[0].sample_kind() == metatomic::SampleKind::Atom));
        CHECK(outputs[0].gradients().empty());
        CHECK(outputs[1].name() == "energy");
        CHECK((outputs[1].sample_kind() == metatomic::SampleKind::System));
        CHECK((outputs[1].gradients() == std::vector<Gradients>{Gradients::Positions, Gradients::Strain}));

        auto pair_lists = model.requested_pair_lists({energy()});
        REQUIRE(pair_lists.size() == 1);
        CHECK(pair_lists[0].cutoff() == CUTOFF);
        CHECK_FALSE(pair_lists[0].full_list());
        CHECK(pair_lists[0].strict());
        CHECK(model.requested_inputs({energy()}).empty());

        auto harmonic = LegacyModel(harmonic_model(0.0));
        auto inputs = harmonic.requested_inputs({energy()});
        REQUIRE(inputs.size() == 1);
        CHECK(inputs[0].name() == "mass");
        CHECK(inputs[0].unit() == "u");
        CHECK((inputs[0].sample_kind() == metatomic::SampleKind::Atom));

        // explicit gradients are kept, and strain is added for autograd
        auto harmonic_outputs = harmonic.capabilities().outputs();
        REQUIRE(harmonic_outputs.size() == 1);
        CHECK((harmonic_outputs[0].gradients() == std::vector<Gradients>{Gradients::Positions, Gradients::Strain}));
    }

    SECTION("energy and autograd gradients") {
        auto atomistic = harmonic_pairs_model();
        auto model = LegacyModel(atomistic);
        auto pair_lists = model.requested_pair_lists({});

        auto positions = std::vector<Positions>{cluster(), cluster(1.05)};
        auto systems = std::vector<metatomic::System>();
        for (const auto& p: positions) {
            systems.push_back(system(p, pair_lists));
        }

        auto outputs = metatomic::execute_model(
            model, systems, std::nullopt,
            {energy("system", {Gradients::Positions, Gradients::Strain})},
            /*check_consistency*/ true
        );
        REQUIRE(outputs.size() == 1);
        auto block = outputs[0].block_by_id(0);
        auto values = block.values();

        auto positions_gradient = block.gradient("positions");
        const auto positions_samples = positions_gradient.samples().values_cpu();
        auto positions_values = positions_gradient.values();
        auto strain_values = block.gradient("strain").values();

        auto row = size_t{0};
        for (size_t system_i = 0; system_i < positions.size(); system_i++) {
            auto reference = harmonic_pairs_reference(positions[system_i]);
            CHECK_THAT(values(system_i, 0), WithinAbs(sum(reference.energies), TOLERANCE));

            for (size_t atom = 0; atom < positions[system_i].size(); atom++) {
                CHECK(positions_samples(row, 0) == static_cast<int32_t>(system_i));
                CHECK(positions_samples(row, 1) == static_cast<int32_t>(system_i));
                CHECK(positions_samples(row, 2) == static_cast<int32_t>(atom));
                for (size_t a = 0; a < 3; a++) {
                    CHECK_THAT(
                        positions_values(row, a, 0),
                        WithinAbs(reference.positions_gradient[atom][a], TOLERANCE)
                    );
                }
                row += 1;
            }

            for (size_t a = 0; a < 3; a++) {
                for (size_t b = 0; b < 3; b++) {
                    CHECK_THAT(
                        strain_values(system_i, a, b, 0),
                        WithinAbs(reference.strain_gradient[a][b], TOLERANCE)
                    );
                }
            }
        }
        CHECK(row == positions_samples.shape()[0]);
    }

    SECTION("per-atom and per-system outputs") {
        auto atomistic = harmonic_pairs_model();
        auto inner = atomistic.attr("module").toModule();
        auto model = LegacyModel(atomistic);

        auto positions = cluster();
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(positions, model.requested_pair_lists({})));
        auto reference = harmonic_pairs_reference(positions);

        // the same name with different sample kinds needs two calls
        auto outputs = metatomic::execute_model(
            model, systems, std::nullopt, {energy("atom"), energy("system")}, /*check_consistency*/ true
        );
        CHECK(calls(inner) == 2);
        REQUIRE(outputs.size() == 2);

        auto per_atom = outputs[0].block_by_id(0);
        CHECK(names(per_atom.samples()) == std::vector<std::string>{"system", "atom"});
        auto per_atom_values = per_atom.values();
        REQUIRE(per_atom_values.shape()[0] == positions.size());
        for (size_t atom = 0; atom < positions.size(); atom++) {
            CHECK_THAT(per_atom_values(atom, 0), WithinAbs(reference.energies[atom], TOLERANCE));
        }

        auto per_system = outputs[1].block_by_id(0);
        CHECK(names(per_system.samples()) == std::vector<std::string>{"system"});
        CHECK_THAT(first_value(per_system), WithinAbs(sum(reference.energies), TOLERANCE));

        // a single request is a single call
        metatomic::execute_model(model, systems, std::nullopt, {energy("atom")}, true);
        CHECK(calls(inner) == 3);
    }

    SECTION("selected atoms") {
        auto model = LegacyModel(harmonic_pairs_model());
        auto positions = cluster();
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(positions, model.requested_pair_lists({})));
        auto reference = harmonic_pairs_reference(positions);

        auto selected = metatensor::Labels({"system", "atom"}, {{0, 1}, {0, 4}});
        auto outputs = metatomic::execute_model(model, systems, selected, {energy("atom")}, true);
        auto block = outputs[0].block_by_id(0);
        CHECK(block.samples() == selected);
        auto values = block.values();
        CHECK_THAT(values(0, 0), WithinAbs(reference.energies[1], TOLERANCE));
        CHECK_THAT(values(1, 0), WithinAbs(reference.energies[4], TOLERANCE));

        outputs = metatomic::execute_model(model, systems, selected, {energy("system")}, true);
        CHECK_THAT(
            first_value(outputs[0].block_by_id(0)),
            WithinAbs(reference.energies[1] + reference.energies[4], TOLERANCE)
        );
    }

    SECTION("explicit gradients and inputs") {
        auto model = LegacyModel(harmonic_model(/*offset*/ 1.0));
        auto positions = cluster();
        auto masses = std::vector<double>{16.0, 1.0, 1.0, 2.0, 1.0, 3.0};
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(positions, {}, masses));

        auto outputs = metatomic::execute_model(
            model, systems, std::nullopt,
            {energy("system", {Gradients::Positions, Gradients::Strain})},
            /*check_consistency*/ true
        );
        auto block = outputs[0].block_by_id(0);

        auto expected_energy = 0.0;
        auto expected_strain = std::array<std::array<double, 3>, 3>{};
        for (size_t i = 0; i < positions.size(); i++) {
            for (size_t a = 0; a < 3; a++) {
                expected_energy += masses[i] * positions[i][a] * positions[i][a];
                for (size_t b = 0; b < 3; b++) {
                    expected_strain[a][b] += 2.0 * masses[i] * positions[i][a] * positions[i][b];
                }
            }
        }
        CHECK_THAT(first_value(block), WithinAbs(expected_energy, 1e-10));

        // the explicit gradient is used as-is, including the offset
        auto positions_values = block.gradient("positions").values();
        for (size_t i = 0; i < positions.size(); i++) {
            for (size_t a = 0; a < 3; a++) {
                CHECK_THAT(
                    positions_values(i, a, 0),
                    WithinAbs(2.0 * masses[i] * positions[i][a] + 1.0, TOLERANCE)
                );
            }
        }

        // the strain gradient comes from autograd
        auto strain_values = block.gradient("strain").values();
        for (size_t a = 0; a < 3; a++) {
            for (size_t b = 0; b < 3; b++) {
                CHECK_THAT(strain_values(0, a, b, 0), WithinAbs(expected_strain[a][b], 1e-10));
            }
        }
    }

    SECTION("units") {
        // the model works in Bohr and eV, the engine in Angstrom and kcal/mol
        auto model = LegacyModel(harmonic_model(/*offset*/ 0.0, "Bohr"));
        auto positions = cluster();
        auto masses = std::vector<double>{16.0, 1.0, 1.0, 2.0, 1.0, 3.0};
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(positions, {}, masses));

        auto outputs = metatomic::execute_model(
            model, systems, std::nullopt,
            {energy("system", {Gradients::Positions, Gradients::Strain}, "kcal/mol")},
            /*check_consistency*/ true
        );
        auto block = outputs[0].block_by_id(0);

        auto length = unit_conversion_factor("angstrom", "bohr");
        auto energy_factor = unit_conversion_factor("eV", "kcal/mol");

        auto expected_energy = 0.0;
        for (size_t i = 0; i < positions.size(); i++) {
            for (size_t a = 0; a < 3; a++) {
                expected_energy += masses[i] * positions[i][a] * positions[i][a];
            }
        }
        expected_energy *= energy_factor * length * length;
        CHECK_THAT(first_value(block), WithinRel(expected_energy, 1e-12));

        // dE/dr in kcal/mol/A
        auto positions_values = block.gradient("positions").values();
        for (size_t i = 0; i < positions.size(); i++) {
            for (size_t a = 0; a < 3; a++) {
                auto expected = energy_factor * length * length * 2.0 * masses[i] * positions[i][a];
                CHECK_THAT(positions_values(i, a, 0), WithinAbs(expected, 1e-9));
            }
        }

        // the strain gradient is an energy
        auto strain_values = block.gradient("strain").values();
        auto trace = strain_values(0, 0, 0, 0) + strain_values(0, 1, 1, 0) + strain_values(0, 2, 2, 0);
        CHECK_THAT(trace, WithinRel(2.0 * expected_energy, 1e-12));
    }

    SECTION("errors") {
        // not an AtomisticModel
        auto not_a_model = torch::jit::Module("__torch__.NotAModel");
        CHECK_THROWS_WITH(LegacyModel(not_a_model),
            Contains("this TorchScript module is not an AtomisticModel: it does not have a 'capabilities' method")
        );

        // empty length unit
        auto no_length_unit = harmonic_model(0.0, "");
        CHECK_THROWS_WITH(LegacyModel(no_length_unit),
            Contains("this model does not declare a length unit, and can not be used with LegacyModel")
        );

        // gradients of per-atom outputs are not declared
        auto model = LegacyModel(harmonic_pairs_model());
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(cluster(), model.requested_pair_lists({})));
        CHECK_THROWS(metatomic::execute_model(
            model, systems, std::nullopt, {energy("atom", {Gradients::Positions})}, true
        ));
    }
}

#ifdef METATOMIC_TORCH_PLUGIN_PATH
TEST_CASE("metatomic-torch plugin") {
    using metatomic::Gradients;

    auto path = std::string("legacy-model-test.pt");
    auto extra_files = torch::jit::ExtraFilesMap{
        {"metatomic-version", metatomic_torch::version()},
        {"extensions", "[]"},
        {"extensions-deps", "[]"},
    };
    harmonic_pairs_model().save(path, extra_files);

    metatomic::load_plugin(METATOMIC_TORCH_PLUGIN_PATH);

    SECTION("load and run a model") {
        // the plugin is found without giving its name
        auto model = metatomic::load_model(path);
        CHECK(model.metadata().name() == "test model");

        model = metatomic::load_model(path, std::nullopt, "metatomic-torch");
        auto pair_lists = model.requested_pair_lists({energy()});
        REQUIRE(pair_lists.size() == 1);

        auto positions = cluster();
        auto systems = std::vector<metatomic::System>();
        systems.push_back(system(positions, pair_lists));
        auto reference = harmonic_pairs_reference(positions);

        auto outputs = metatomic::execute_model(
            model, systems, std::nullopt,
            {energy("system", {Gradients::Positions, Gradients::Strain})},
            /*check_consistency*/ true
        );
        auto block = outputs[0].block_by_id(0);
        CHECK_THAT(first_value(block), WithinAbs(sum(reference.energies), TOLERANCE));

        auto positions_values = block.gradient("positions").values();
        for (size_t atom = 0; atom < positions.size(); atom++) {
            for (size_t a = 0; a < 3; a++) {
                CHECK_THAT(
                    positions_values(atom, a, 0),
                    WithinAbs(reference.positions_gradient[atom][a], TOLERANCE)
                );
            }
        }
    }

    SECTION("errors") {
        auto not_a_model = std::string("not-a-model.pt");
        std::ofstream(not_a_model) << "this is not a TorchScript archive";
        CHECK_THROWS_WITH(
            metatomic::load_model(not_a_model, std::nullopt, "metatomic-torch"),
            Contains("plugin 'metatomic-torch' could not load the model")
        );

        CHECK_THROWS_WITH(
            metatomic::load_model(path, R"({"device": "cpu"})", "metatomic-torch"),
            Contains("unknown option for metatomic-torch models: 'device'")
        );
    }
}
#endif
