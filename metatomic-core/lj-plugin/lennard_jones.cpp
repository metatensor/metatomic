#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <metatomic.h>
#include <metatomic.hpp>
#include <nlohmann/json.hpp>


namespace {

struct LennardJonesOptions {
    double sigma = 1.0;
    double epsilon = 1.0;
    double cutoff = 3.0;
    std::vector<int32_t> atomic_types = {1};
    std::string length_unit = "Angstrom";
    std::string energy_unit = "eV";
};

double parse_double(const nlohmann::json& json, const std::string& name, double fallback) {
    if (!json.contains(name)) {
        return fallback;
    }

    const auto& value = json[name];
    if (value.is_number()) {
        return value.get<double>();
    }
    if (!value.is_string()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be a number");
    }

    // std::stod follows LC_NUMERIC; "1.0" then fails on a comma-decimal locale.
    std::istringstream in(value.get<std::string>());
    in.imbue(std::locale::classic());
    double result = 0.0;
    in >> std::noskipws >> result;
    if (!in || in.get() != std::char_traits<char>::eof()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be a number");
    }
    return result;
}

int32_t parse_int32(const std::string& value, const std::string& name) {
    size_t parsed = 0;
    int64_t result = 0;
    try {
        result = std::stoll(value, &parsed);
    } catch (const std::exception&) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be an integer");
    }
    if (parsed != value.size()) {
        throw metatomic::Error("Lennard-Jones option '" + name + "' must be an integer");
    }
    if (result < std::numeric_limits<int32_t>::min()
        || result > std::numeric_limits<int32_t>::max())
    {
        throw metatomic::Error("Lennard-Jones option '" + name + "' is out of range");
    }
    return static_cast<int32_t>(result);
}

std::vector<int32_t> parse_atomic_types(const nlohmann::json& json) {
    if (!json.contains("atomic_type")) {
        return {1};
    }

    const auto& value = json["atomic_type"];
    if (value.is_array()) {
        auto types = value.get<std::vector<int32_t>>();
        if (types.empty()) {
            throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
        }
        return types;
    }
    if (!value.is_string()) {
        throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
    }

    std::vector<int32_t> types;
    std::istringstream in(value.get<std::string>());
    std::string token;
    while (std::getline(in, token, ',')) {
        auto first = token.find_first_not_of(" \t");
        if (first == std::string::npos) {
            throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
        }
        auto last = token.find_last_not_of(" \t");
        types.push_back(parse_int32(token.substr(first, last - first + 1), "atomic_type"));
    }
    if (types.empty()) {
        throw metatomic::Error("Lennard-Jones option 'atomic_type' must be an integer");
    }
    return types;
}

LennardJonesOptions parse_options(const char* options_json) {
    auto json = nlohmann::json::parse(options_json == nullptr ? "{}" : options_json);
    if (!json.is_object()) {
        throw metatomic::Error("Lennard-Jones options must be a JSON object");
    }

    const std::vector<std::string> allowed = {
        "sigma", "epsilon", "cutoff", "atomic_type", "length_unit", "energy_unit"
    };
    for (const auto& item: json.items()) {
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end()) {
            throw metatomic::Error("unknown Lennard-Jones option: '" + item.key() + "'");
        }
    }

    LennardJonesOptions options;
    options.sigma = parse_double(json, "sigma", options.sigma);
    options.epsilon = parse_double(json, "epsilon", options.epsilon);
    options.cutoff = parse_double(json, "cutoff", options.cutoff);
    options.atomic_types = parse_atomic_types(json);
    if (json.contains("length_unit")) {
        options.length_unit = json["length_unit"].get<std::string>();
    }
    if (json.contains("energy_unit")) {
        options.energy_unit = json["energy_unit"].get<std::string>();
    }

    if (!std::isfinite(options.sigma) || options.sigma <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'sigma' must be finite and positive");
    }
    if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'epsilon' must be finite and positive");
    }
    if (!std::isfinite(options.cutoff) || options.cutoff <= 0.0) {
        throw metatomic::Error("Lennard-Jones option 'cutoff' must be finite and positive");
    }
    if (options.length_unit.empty()) {
        throw metatomic::Error("Lennard-Jones option 'length_unit' must not be empty");
    }
    if (options.energy_unit.empty()) {
        throw metatomic::Error("Lennard-Jones option 'energy_unit' must not be empty");
    }

    return options;
}

struct Calculation {
    std::vector<double> atomic_energies;
    std::vector<double> positions_gradient;
};

struct Selection {
    bool all = true;
    std::vector<std::vector<char>> atoms;
    std::vector<int32_t> systems;
};

bool atom_selected(const Selection& selection, size_t system, size_t atom) {
    return selection.all || selection.atoms[system][atom] != 0;
}

Selection parse_selection(
    const std::vector<metatomic::System>& systems,
    const metatensor::Labels* selected_atoms
) {
    Selection selection;
    if (selected_atoms == nullptr) {
        selection.systems.reserve(systems.size());
        for (size_t system = 0; system < systems.size(); system++) {
            selection.systems.push_back(static_cast<int32_t>(system));
        }
        return selection;
    }

    const auto names = selected_atoms->names();
    if (names.size() != 2
        || std::strcmp(names[0], "system") != 0
        || std::strcmp(names[1], "atom") != 0)
    {
        throw metatomic::Error(
            "Lennard-Jones selected_atoms must have names ['system', 'atom']"
        );
    }

    selection.all = false;
    selection.atoms.resize(systems.size());
    for (size_t system = 0; system < systems.size(); system++) {
        selection.atoms[system].assign(systems[system].size(), 0);
    }

    std::set<int32_t> unique_systems;
    const auto values = selected_atoms->values_cpu();
    if (values.shape().size() != 2 || values.shape()[1] != 2) {
        throw metatomic::Error("Lennard-Jones selected_atoms must have two columns");
    }
    for (size_t row = 0; row < values.shape()[0]; row++) {
        const auto system = values(row, 0);
        const auto atom = values(row, 1);
        if (system < 0 || static_cast<size_t>(system) >= systems.size()) {
            throw metatomic::Error("Lennard-Jones selected_atoms contains an invalid system index");
        }
        if (atom < 0 || static_cast<size_t>(atom) >= systems[static_cast<size_t>(system)].size()) {
            throw metatomic::Error("Lennard-Jones selected_atoms contains an invalid atom index");
        }
        selection.atoms[static_cast<size_t>(system)][static_cast<size_t>(atom)] = 1;
        unique_systems.insert(system);
    }
    selection.systems.assign(unique_systems.begin(), unique_systems.end());
    return selection;
}

double system_energy(const Calculation& calculation, const Selection& selection, size_t system) {
    double energy = 0.0;
    for (size_t atom = 0; atom < calculation.atomic_energies.size(); atom++) {
        if (atom_selected(selection, system, atom)) {
            energy += calculation.atomic_energies[atom];
        }
    }
    return energy;
}

metatensor::Labels labels_from_values(
    const std::vector<std::string>& names,
    const std::vector<int32_t>& values,
    size_t count
) {
    if (count == 0) {
        return metatensor::Labels(names);
    }
    return metatensor::Labels(names, values.data(), count);
}

metatensor::TensorMap tensor_map_from_block(metatensor::TensorBlock block) {
    // TensorBlock is move-only, so `{std::move(block)}` cannot construct the vector.
    std::vector<metatensor::TensorBlock> blocks;
    blocks.push_back(std::move(block));
    return metatensor::TensorMap(
        metatensor::Labels({"_"}, {{0}}), std::move(blocks)
    );
}

metatensor::TensorMap system_energy_output(
    const std::vector<Calculation>& calculations,
    const Selection& selection,
    bool include_positions_gradient
) {
    auto properties = metatensor::Labels({"energy"}, {{0}});
    std::vector<int32_t> samples;
    std::vector<double> energies;
    samples.reserve(selection.systems.size());
    energies.reserve(selection.systems.size());
    for (auto system: selection.systems) {
        samples.push_back(system);
        energies.push_back(system_energy(
            calculations[static_cast<size_t>(system)],
            selection,
            static_cast<size_t>(system)
        ));
    }

    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{selection.systems.size(), 1}, std::move(energies)
        ),
        labels_from_values({"system"}, samples, selection.systems.size()),
        {},
        properties
    );

    if (include_positions_gradient) {
        std::vector<int32_t> gradient_samples;
        std::vector<double> gradient_values;
        for (size_t sample = 0; sample < selection.systems.size(); sample++) {
            const auto system = static_cast<size_t>(selection.systems[sample]);
            const auto& gradient = calculations[system].positions_gradient;
            const auto atom_count = gradient.size() / 3;
            for (size_t atom = 0; atom < atom_count; atom++) {
                gradient_samples.insert(
                    gradient_samples.end(),
                    {static_cast<int32_t>(sample), selection.systems[sample], static_cast<int32_t>(atom)}
                );
                gradient_values.insert(
                    gradient_values.end(),
                    gradient.begin() + static_cast<std::ptrdiff_t>(3 * atom),
                    gradient.begin() + static_cast<std::ptrdiff_t>(3 * atom + 3)
                );
            }
        }
        const auto row_count = gradient_samples.size() / 3;
        auto gradient = metatensor::TensorBlock(
            std::make_unique<metatensor::SimpleDataArray<double>>(
                std::vector<uintptr_t>{row_count, 3, 1}, std::move(gradient_values)
            ),
            labels_from_values({"sample", "system", "atom"}, gradient_samples, row_count),
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            properties
        );
        block.add_gradient("positions", std::move(gradient));
    }

    return tensor_map_from_block(std::move(block));
}

metatensor::TensorMap atom_energy_output(
    const std::vector<Calculation>& calculations,
    const Selection& selection
) {
    auto properties = metatensor::Labels({"energy"}, {{0}});
    std::vector<int32_t> samples;
    std::vector<double> energies;
    for (auto system: selection.systems) {
        const auto sys = static_cast<size_t>(system);
        const auto& atomic = calculations[sys].atomic_energies;
        for (size_t atom = 0; atom < atomic.size(); atom++) {
            if (atom_selected(selection, sys, atom)) {
                samples.insert(samples.end(), {system, static_cast<int32_t>(atom)});
                energies.push_back(atomic[atom]);
            }
        }
    }
    const auto row_count = energies.size();
    auto block = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{row_count, 1}, std::move(energies)
        ),
        labels_from_values({"system", "atom"}, samples, row_count),
        {},
        properties
    );
    return tensor_map_from_block(std::move(block));
}

class LennardJones final: public metatomic::BaseModel {
public:
    explicit LennardJones(LennardJonesOptions options):
        options_(std::move(options)),
        pair_options_(metatomic::PairListOptions::builder()
            .cutoff(options_.cutoff)
            .full_list(false)
            .strict(false)
            .add_requestor("lj-plugin")
            .build())
    {}

    metatomic::ModelCapabilities capabilities() const final {
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

        std::vector<int64_t> atomic_types;
        atomic_types.reserve(options_.atomic_types.size());
        for (auto type: options_.atomic_types) {
            atomic_types.push_back(type);
        }

        return metatomic::ModelCapabilities::builder()
            .atomic_types(std::move(atomic_types))
            .interaction_range(options_.cutoff)
            .length_unit(options_.length_unit)
            .supported_devices({metatomic::ModelCapabilities::Device::CPU})
            .dtype(metatomic::ModelCapabilities::DType::Float64)
            .add_output(energy)
            .add_output(atomic_energy)
            .build();
    }

    metatomic::ModelMetadata metadata() const final {
        return metatomic::ModelMetadata::builder()
            .name("Lennard-Jones test model")
            .add_author("metatomic")
            .description("Shifted Lennard-Jones pair potential for engine tests")
            .add_reference("model", "https://github.com/metatensor/lj-test")
            .build();
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists() const final {
        return {pair_options_};
    }

    std::vector<metatomic::Quantity> requested_inputs() const final {
        return {};
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        const metatensor::Labels* selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs
    ) final {
        for (const auto& output: requested_outputs) {
            validate_output(output);
        }
        if (requested_outputs.empty()) {
            return {};
        }

        const auto selection = parse_selection(systems, selected_atoms);
        std::vector<Calculation> calculations;
        calculations.reserve(systems.size());
        for (size_t system = 0; system < systems.size(); system++) {
            const std::vector<char>* mask = selection.all ? nullptr : &selection.atoms[system];
            calculations.push_back(calculate(systems[system], mask));
        }

        std::vector<metatensor::TensorMap> outputs;
        outputs.reserve(requested_outputs.size());
        for (const auto& output: requested_outputs) {
            if (output.sample_kind() == metatomic::SampleKind::Atom) {
                outputs.push_back(atom_energy_output(calculations, selection));
                continue;
            }
            const auto& gradients = output.gradients();
            auto positions = std::find(
                gradients.begin(), gradients.end(), metatomic::Gradients::Positions
            );
            outputs.push_back(system_energy_output(
                calculations, selection, positions != gradients.end()
            ));
        }
        return outputs;
    }

private:
    void validate_output(const metatomic::Quantity& output) const {
        if (output.name() != "energy") {
            throw metatomic::Error("Lennard-Jones plugin only supports the 'energy' output");
        }
        if (output.sample_kind() != metatomic::SampleKind::System
            && output.sample_kind() != metatomic::SampleKind::Atom)
        {
            throw metatomic::Error("Lennard-Jones energy must use system or atom samples");
        }
        for (const auto gradient: output.gradients()) {
            if (gradient != metatomic::Gradients::Positions) {
                throw metatomic::Error(
                    "Lennard-Jones plugin only supports positions gradients"
                );
            }
        }
    }

    Calculation calculate(
        const metatomic::System& system,
        const std::vector<char>* selected
    ) const {
        auto pairs = system.pairs(pair_options_);
        auto displacements = pairs.values<double>();
        const auto pair_samples = pairs.samples().values_cpu();
        if (displacements.shape().size() != 3
            || displacements.shape()[1] != 3
            || displacements.shape()[2] != 1)
        {
            throw metatomic::Error("Lennard-Jones pair values must have shape (pairs, 3, 1)");
        }
        if (pair_samples.shape().size() != 2 || pair_samples.shape()[1] < 2) {
            throw metatomic::Error("Lennard-Jones pair samples must identify two atoms");
        }
        if (pair_samples.shape()[0] != displacements.shape()[0]) {
            throw metatomic::Error("Lennard-Jones pair samples and values have different sizes");
        }

        Calculation result;
        result.atomic_energies.assign(system.size(), 0.0);
        result.positions_gradient.resize(3 * system.size(), 0.0);
        const auto cutoff_2 = options_.cutoff * options_.cutoff;
        const auto sigma_2 = options_.sigma * options_.sigma;
        const auto cutoff_ratio_2 = sigma_2 / cutoff_2;
        const auto cutoff_ratio_6 = cutoff_ratio_2
            * cutoff_ratio_2 * cutoff_ratio_2;
        const auto shift = 4.0 * options_.epsilon
            * (cutoff_ratio_6 * cutoff_ratio_6 - cutoff_ratio_6);

        for (size_t pair = 0; pair < displacements.shape()[0]; pair++) {
            const auto dx = displacements(pair, 0, 0);
            const auto dy = displacements(pair, 1, 0);
            const auto dz = displacements(pair, 2, 0);
            const auto distance_2 = dx * dx + dy * dy + dz * dz;
            if (distance_2 <= 0.0) {
                throw metatomic::Error("Lennard-Jones pair distance must be positive");
            }
            if (distance_2 >= cutoff_2) {
                continue;
            }

            const auto ratio_2 = sigma_2 / distance_2;
            const auto ratio_6 = ratio_2 * ratio_2 * ratio_2;
            const auto ratio_12 = ratio_6 * ratio_6;
            const auto pair_energy = 4.0 * options_.epsilon
                * (ratio_12 - ratio_6) - shift;
            const auto half_energy = 0.5 * pair_energy;

            const auto first = pair_samples(pair, 0);
            const auto second = pair_samples(pair, 1);
            if (first < 0 || second < 0
                || static_cast<size_t>(first) >= system.size()
                || static_cast<size_t>(second) >= system.size())
            {
                throw metatomic::Error("Lennard-Jones pair contains an invalid atom index");
            }

            result.atomic_energies[static_cast<size_t>(first)] += half_energy;
            result.atomic_energies[static_cast<size_t>(second)] += half_energy;

            const auto first_on = selected == nullptr
                || (*selected)[static_cast<size_t>(first)] != 0;
            const auto second_on = selected == nullptr
                || (*selected)[static_cast<size_t>(second)] != 0;
            double scale = 0.0;
            if (first_on) {
                scale += 0.5;
            }
            if (second_on) {
                scale += 0.5;
            }
            if (scale == 0.0) {
                continue;
            }

            const auto energy_derivative = 12.0 * options_.epsilon / distance_2
                * (ratio_6 - 2.0 * ratio_12);
            const double displacement[3] = {dx, dy, dz};
            for (size_t xyz = 0; xyz < 3; xyz++) {
                const auto gradient = -2.0 * scale * energy_derivative * displacement[xyz];
                result.positions_gradient[3 * static_cast<size_t>(first) + xyz] += gradient;
                result.positions_gradient[3 * static_cast<size_t>(second) + xyz] -= gradient;
            }
        }

        return result;
    }

    LennardJonesOptions options_;
    metatomic::PairListOptions pair_options_;
};

mta_status_t load_model(
    const char* load_from,
    const char* options_json,
    mta_model_t* model
) {
    if (load_from == nullptr
        || (std::string(load_from) != "lennard-jones" && std::string(load_from) != "lj"))
    {
        return MTA_UNSUPPORTED_MODEL_ERROR;
    }

    return metatomic::details::catch_exceptions([&]() {
        if (model == nullptr) {
            throw metatomic::Error("model output pointer must not be null");
        }
        *model = metatomic::BaseModel::to_mta_model(
            std::make_unique<LennardJones>(parse_options(options_json))
        );
    });
}

} // namespace


MTA_REGISTER_PLUGIN(register_plugin, {
    mta_plugin_t plugin = {};
    plugin.abi_version = MTA_ABI_VERSION;
    plugin.name = "lj-plugin";
    plugin.load_model = load_model;
    return register_plugin(plugin);
});
