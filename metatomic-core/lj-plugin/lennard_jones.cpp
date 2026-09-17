#include <algorithm>
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

namespace {

/// Plugin options parsed from the string-to-string map passed to `load_model`.
struct LennardJonesOptions {
  /// Length parameter of the 12-6 potential.
  double sigma = 1.0;
  /// Depth of the potential well.
  double epsilon = 1.0;
  /// Pair cutoff. The potential is shifted to zero at this distance.
  double cutoff = 3.0;
  /// Atomic type reported in the model capabilities (not used to filter pairs).
  int32_t atomic_type = 1;
  /// Unit of positions, cell, `sigma`, and `cutoff`.
  std::string length_unit = "Angstrom";
  /// Unit of `epsilon` and of the returned energies.
  std::string energy_unit = "eV";
};

/// Locale-independent parse of a decimal floating-point string (`.` separator).
double parse_double(const std::string &value, const std::string &name) {
  // std::stod follows LC_NUMERIC; "1.0" then fails on a comma-decimal locale.
  std::istringstream in(value);
  in.imbue(std::locale::classic());
  double result = 0.0;
  in >> std::noskipws >> result;
  if (!in || in.get() != std::char_traits<char>::eof()) {
    throw metatomic::Error("Lennard-Jones option '" + name +
                           "' must be a number");
  }
  return result;
}

/// Parse a base-10 integer and reject anything that does not fit in `int32_t`.
int32_t parse_int32(const std::string &value, const std::string &name) {
  size_t parsed = 0;
  int64_t result = 0;
  try {
    result = std::stoll(value, &parsed);
  } catch (const std::exception &) {
    throw metatomic::Error("Lennard-Jones option '" + name +
                           "' must be an integer");
  }
  if (parsed != value.size()) {
    throw metatomic::Error("Lennard-Jones option '" + name +
                           "' must be an integer");
  }
  if (result < std::numeric_limits<int32_t>::min() ||
      result > std::numeric_limits<int32_t>::max()) {
    throw metatomic::Error("Lennard-Jones option '" + name +
                           "' is out of range");
  }
  return static_cast<int32_t>(result);
}

/// Read plugin options, keep the defaults for missing keys, and reject unknown ones.
LennardJonesOptions
parse_options(const std::map<std::string, std::string> &options) {
  const std::vector<std::string> allowed = {"sigma",       "epsilon",
                                            "cutoff",      "atomic_type",
                                            "length_unit", "energy_unit"};
  for (const auto &item : options) {
    if (std::find(allowed.begin(), allowed.end(), item.first) ==
        allowed.end()) {
      throw metatomic::Error("unknown Lennard-Jones option: '" + item.first +
                             "'");
    }
  }

  LennardJonesOptions parsed;
  if (const auto it = options.find("sigma"); it != options.end()) {
    parsed.sigma = parse_double(it->second, "sigma");
  }
  if (const auto it = options.find("epsilon"); it != options.end()) {
    parsed.epsilon = parse_double(it->second, "epsilon");
  }
  if (const auto it = options.find("cutoff"); it != options.end()) {
    parsed.cutoff = parse_double(it->second, "cutoff");
  }
  if (const auto it = options.find("atomic_type"); it != options.end()) {
    parsed.atomic_type = parse_int32(it->second, "atomic_type");
  }

  const auto length_unit = options.find("length_unit");
  if (length_unit != options.end()) {
    parsed.length_unit = length_unit->second;
  }
  const auto energy_unit = options.find("energy_unit");
  if (energy_unit != options.end()) {
    parsed.energy_unit = energy_unit->second;
  }

  if (!std::isfinite(parsed.sigma) || parsed.sigma <= 0.0) {
    throw metatomic::Error(
        "Lennard-Jones option 'sigma' must be finite and positive");
  }
  if (!std::isfinite(parsed.epsilon) || parsed.epsilon <= 0.0) {
    throw metatomic::Error(
        "Lennard-Jones option 'epsilon' must be finite and positive");
  }
  if (!std::isfinite(parsed.cutoff) || parsed.cutoff <= 0.0) {
    throw metatomic::Error(
        "Lennard-Jones option 'cutoff' must be finite and positive");
  }
  if (parsed.length_unit.empty()) {
    throw metatomic::Error(
        "Lennard-Jones option 'length_unit' must not be empty");
  }
  if (parsed.energy_unit.empty()) {
    throw metatomic::Error(
        "Lennard-Jones option 'energy_unit' must not be empty");
  }

  return parsed;
}

/// Which atoms contribute to returned energies / selection-aware gradients.
///
/// Uses `vector<char>` rather than `vector<bool>` to avoid the latter's proxy
/// reference semantics (no real `bool&`, packing surprises with pointers).
struct Selection {
  /// When true, every atom in every system is selected (`atoms` is unused).
  bool all = true;
  /// Per-system mask: non-zero means the atom is selected.
  std::vector<std::vector<char>> atoms;
  /// System indices that appear at least once in the selection (sorted).
  std::vector<int32_t> systems;
};

/// Build a `Selection` from `["system", "atom"]` labels.
///
/// An empty `selected_atoms` means every atom. Names and index bounds are
/// validated by metatomic when `check_consistency` is enabled; we still guard
/// bounds here so a direct `execute_inner` call cannot index out of range.
Selection
parse_selection(const std::vector<metatomic::System> &systems,
                const std::optional<metatensor::Labels> &selected_atoms) {
  Selection selection;
  if (!selected_atoms.has_value()) {
    selection.systems.reserve(systems.size());
    for (size_t system = 0; system < systems.size(); system++) {
      selection.systems.push_back(static_cast<int32_t>(system));
    }
    return selection;
  }

  selection.all = false;
  selection.atoms.resize(systems.size());
  for (size_t system = 0; system < systems.size(); system++) {
    selection.atoms[system].assign(systems[system].size(), 0);
  }

  std::set<int32_t> unique_systems;
  const auto values = selected_atoms->values_cpu();
  assert(values.shape().size() == 2 && values.shape()[1] == 2);
  for (size_t row = 0; row < values.shape()[0]; row++) {
    const auto system = values(row, 0);
    const auto atom = values(row, 1);
    assert(system >= 0 && static_cast<size_t>(system) < systems.size());
    assert(atom >= 0 && static_cast<size_t>(atom) <
                            systems[static_cast<size_t>(system)].size());
    selection.atoms[static_cast<size_t>(system)][static_cast<size_t>(atom)] = 1;
    unique_systems.insert(system);
  }
  selection.systems.assign(unique_systems.begin(), unique_systems.end());
  return selection;
}

metatensor::TensorMap tensor_map_from_block(metatensor::TensorBlock block) {
  // TensorBlock is move-only, so `{std::move(block)}` cannot construct the
  // vector.
  std::vector<metatensor::TensorBlock> blocks;
  blocks.push_back(std::move(block));
  return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}),
                               std::move(blocks));
}

/// System-level energy samples: one row per selected system.
///
/// Positions-gradient samples use `(sample, system, atom)` where `sample` is
/// the row index in this block (not the system id). Both endpoints of a
/// contributing pair appear, including an unselected neighbor whose position
/// still affects the selected energy.
metatensor::TensorMap system_energy_output(
    const std::vector<std::vector<double>> &atomic_energies,
    const std::vector<std::vector<double>> &positions_gradients,
    const Selection &selection, bool include_positions_gradient) {
  auto properties = metatensor::Labels({"energy"}, {{0}});
  std::vector<int32_t> samples;
  std::vector<double> energies;
  samples.reserve(selection.systems.size());
  energies.reserve(selection.systems.size());
  for (auto system : selection.systems) {
    samples.push_back(system);
    const auto &per_atom = atomic_energies[static_cast<size_t>(system)];
    double energy = 0.0;
    for (size_t atom = 0; atom < per_atom.size(); atom++) {
      if (selection.all ||
          selection.atoms[static_cast<size_t>(system)][atom] != 0) {
        energy += per_atom[atom];
      }
    }
    energies.push_back(energy);
  }

  auto system_samples =
      selection.systems.empty()
          ? metatensor::Labels({"system"})
          : metatensor::Labels({"system"}, samples.data(),
                               selection.systems.size());
  auto block = metatensor::TensorBlock(
      std::make_unique<metatensor::SimpleDataArray<double>>(
          std::vector<uintptr_t>{selection.systems.size(), 1},
          std::move(energies)),
      std::move(system_samples), {}, properties);

  if (include_positions_gradient) {
    std::vector<int32_t> gradient_samples;
    std::vector<double> gradient_values;
    for (size_t sample = 0; sample < selection.systems.size(); sample++) {
      const auto system = static_cast<size_t>(selection.systems[sample]);
      const auto &gradient = positions_gradients[system];
      const auto atom_count = gradient.size() / 3;
      for (size_t atom = 0; atom < atom_count; atom++) {
        gradient_samples.insert(gradient_samples.end(),
                                {static_cast<int32_t>(sample),
                                 selection.systems[sample],
                                 static_cast<int32_t>(atom)});
        gradient_values.insert(
            gradient_values.end(),
            gradient.begin() + static_cast<std::ptrdiff_t>(3 * atom),
            gradient.begin() + static_cast<std::ptrdiff_t>(3 * atom + 3));
      }
    }
    const auto row_count = gradient_samples.size() / 3;
    auto gradient_sample_labels =
        row_count == 0
            ? metatensor::Labels({"sample", "system", "atom"})
            : metatensor::Labels({"sample", "system", "atom"},
                                 gradient_samples.data(), row_count);
    auto gradient = metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(
            std::vector<uintptr_t>{row_count, 3, 1},
            std::move(gradient_values)),
        std::move(gradient_sample_labels),
        {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})}, properties);
    block.add_gradient("positions", std::move(gradient));
  }

  return tensor_map_from_block(std::move(block));
}

/// Per-atom energy samples for selected atoms only (no gradients).
metatensor::TensorMap
atom_energy_output(const std::vector<std::vector<double>> &atomic_energies,
                   const Selection &selection) {
  auto properties = metatensor::Labels({"energy"}, {{0}});
  std::vector<int32_t> samples;
  std::vector<double> energies;
  for (auto system : selection.systems) {
    const auto sys = static_cast<size_t>(system);
    const auto &atomic = atomic_energies[sys];
    for (size_t atom = 0; atom < atomic.size(); atom++) {
      if (selection.all || selection.atoms[sys][atom] != 0) {
        samples.insert(samples.end(), {system, static_cast<int32_t>(atom)});
        energies.push_back(atomic[atom]);
      }
    }
  }
  const auto row_count = energies.size();
  auto atom_samples =
      row_count == 0
          ? metatensor::Labels({"system", "atom"})
          : metatensor::Labels({"system", "atom"}, samples.data(), row_count);
  auto block = metatensor::TensorBlock(
      std::make_unique<metatensor::SimpleDataArray<double>>(
          std::vector<uintptr_t>{row_count, 1}, std::move(energies)),
      std::move(atom_samples), {}, properties);
  return tensor_map_from_block(std::move(block));
}

/// Shifted Lennard-Jones pair model used to exercise engines.
class LennardJones final : public metatomic::BaseModel {
public:
  /// Store the parsed options and the half neighbor list they imply.
  explicit LennardJones(LennardJonesOptions options)
      : options_(std::move(options)),
        pair_options_(metatomic::PairListOptions::builder()
                          .cutoff(options_.cutoff)
                          .full_list(false)
                          .strict(false)
                          .add_requestor("lj-plugin")
                          .build()) {}

  /// System energy with a positions gradient, plus a per-atom energy without one.
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

    return metatomic::ModelCapabilities::builder()
        .atomic_types({static_cast<int64_t>(options_.atomic_type)})
        .interaction_range(options_.cutoff)
        .length_unit(options_.length_unit)
        .supported_devices({metatomic::ModelCapabilities::Device::CPU})
        .dtype(metatomic::ModelCapabilities::DType::Float64)
        .add_output(energy)
        .add_output(atomic_energy)
        .build();
  }

  /// Name and description shown when an engine inspects the loaded model.
  metatomic::ModelMetadata metadata() const final {
    return metatomic::ModelMetadata::builder()
        .name("Lennard-Jones test model")
        .add_author("metatomic")
        .description("Shifted Lennard-Jones pair potential for engine tests")
        .build();
  }

  /// The half neighbor list `calculate` walks. A full list would double-count.
  std::vector<metatomic::PairListOptions> requested_pair_lists() const final {
    return {pair_options_};
  }

  /// No extra per-system data; pairs are requested separately.
  std::vector<metatomic::Quantity> requested_inputs() const final { return {}; }

  /// Compute each requested energy output for `systems`.
  std::vector<metatensor::TensorMap> execute_inner(
      const std::vector<metatomic::System> &systems,
      const std::optional<metatensor::Labels> &selected_atoms,
      const std::vector<metatomic::Quantity> &requested_outputs) final {
    for (const auto &output : requested_outputs) {
      validate_output(output);
    }
    if (requested_outputs.empty()) {
      return {};
    }

    const auto selection = parse_selection(systems, selected_atoms);
    std::vector<std::vector<double>> atomic_energies;
    std::vector<std::vector<double>> positions_gradients;
    atomic_energies.reserve(systems.size());
    positions_gradients.reserve(systems.size());
    for (size_t system = 0; system < systems.size(); system++) {
      const std::vector<char> *mask =
          selection.all ? nullptr : &selection.atoms[system];
      auto [energies, gradient] = calculate(systems[system], mask);
      atomic_energies.push_back(std::move(energies));
      positions_gradients.push_back(std::move(gradient));
    }

    std::vector<metatensor::TensorMap> outputs;
    outputs.reserve(requested_outputs.size());
    for (const auto &output : requested_outputs) {
      if (output.sample_kind() == metatomic::SampleKind::Atom) {
        outputs.push_back(atom_energy_output(atomic_energies, selection));
        continue;
      }
      const auto &gradients = output.gradients();
      auto positions = std::find(gradients.begin(), gradients.end(),
                                 metatomic::Gradients::Positions);
      outputs.push_back(system_energy_output(
          atomic_energies, positions_gradients, selection,
          positions != gradients.end()));
    }
    return outputs;
  }

private:
  /// Reject outputs this pair potential cannot produce.
  void validate_output(const metatomic::Quantity &output) const {
    if (output.name() != "energy") {
      throw metatomic::Error(
          "Lennard-Jones plugin only supports the 'energy' output");
    }
    if (output.sample_kind() != metatomic::SampleKind::System &&
        output.sample_kind() != metatomic::SampleKind::Atom) {
      throw metatomic::Error(
          "Lennard-Jones energy must use system or atom samples");
    }
    if (output.sample_kind() == metatomic::SampleKind::Atom &&
        !output.gradients().empty()) {
      throw metatomic::Error(
          "Lennard-Jones per-atom energy does not support gradients");
    }
    for (const auto gradient : output.gradients()) {
      if (gradient != metatomic::Gradients::Positions) {
        throw metatomic::Error(
            "Lennard-Jones plugin only supports positions gradients");
      }
    }
  }

  /// Shifted 12-6 Lennard-Jones over the half neighbor list.
  ///
  /// Each pair contributes `pair_energy` split equally onto both endpoints.
  /// The positions gradient is of the *selected* energy only: each selected
  /// endpoint contributes a weight of `1/2`, so `scale ∈ {0, 0.5, 1}`.
  /// Differentiating that selected energy still needs both endpoints'
  /// coordinates — including an unselected neighbor.
  ///
  /// Returns per-atom energies and the flattened `[atom * 3 + xyz]` positions
  /// gradient of the selected energy.
  std::pair<std::vector<double>, std::vector<double>>
  calculate(const metatomic::System &system,
            const std::vector<char> *selected) const {
    auto pairs = system.pairs(pair_options_);
    auto displacements = pairs.values<double>();
    const auto pair_samples = pairs.samples().values_cpu();
    // Layout is part of the pairs interface (validated when pairs are added).
    assert(displacements.shape().size() == 3 && displacements.shape()[1] == 3 &&
           displacements.shape()[2] == 1);
    assert(pair_samples.shape().size() == 2 && pair_samples.shape()[1] >= 2);
    assert(pair_samples.shape()[0] == displacements.shape()[0]);

    std::vector<double> atomic_energies(system.size(), 0.0);
    std::vector<double> positions_gradient(3 * system.size(), 0.0);
    const auto cutoff_2 = options_.cutoff * options_.cutoff;
    const auto sigma_2 = options_.sigma * options_.sigma;
    const auto cutoff_ratio_2 = sigma_2 / cutoff_2;
    const auto cutoff_ratio_6 =
        cutoff_ratio_2 * cutoff_ratio_2 * cutoff_ratio_2;
    const auto shift = 4.0 * options_.epsilon *
                       (cutoff_ratio_6 * cutoff_ratio_6 - cutoff_ratio_6);

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
      const auto pair_energy =
          4.0 * options_.epsilon * (ratio_12 - ratio_6) - shift;
      const auto half_energy = 0.5 * pair_energy;

      const auto first = pair_samples(pair, 0);
      const auto second = pair_samples(pair, 1);
      // Each endpoint receives half the pair energy, regardless of selection.
      atomic_energies[static_cast<size_t>(first)] += half_energy;
      atomic_energies[static_cast<size_t>(second)] += half_energy;

      // Each selected endpoint contributes half the pair energy.
      // Differentiate that selected energy with respect to both endpoints,
      // including an unselected neighbor whose position affects the energy.
      const auto first_on =
          selected == nullptr || (*selected)[static_cast<size_t>(first)] != 0;
      const auto second_on =
          selected == nullptr || (*selected)[static_cast<size_t>(second)] != 0;
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

      // Analytical forces from the shifted LJ potential
      const auto energy_derivative =
          12.0 * options_.epsilon / distance_2 * (ratio_6 - 2.0 * ratio_12);
      const double displacement[3] = {dx, dy, dz};
      for (size_t xyz = 0; xyz < 3; xyz++) {
        const auto gradient =
            -2.0 * scale * energy_derivative * displacement[xyz];
        positions_gradient[3 * static_cast<size_t>(first) + xyz] += gradient;
        // The negative sign is for the equal and opposite force on the
        // second atom in the pair.
        positions_gradient[3 * static_cast<size_t>(second) + xyz] -= gradient;
      }
    }

    return {std::move(atomic_energies), std::move(positions_gradient)};
  }

  /// Values from `load_model`, after defaults and checks.
  LennardJonesOptions options_;
  /// Half list within `options_.cutoff`, shared by every system in a call.
  metatomic::PairListOptions pair_options_;
};

} // namespace

/// Plugin entry point. Returns null when `load_from` is not this test model,
/// so another registered plugin can try.
std::unique_ptr<metatomic::BaseModel>
load_model(const std::string &load_from,
           const std::map<std::string, std::string> &options) {
  if (load_from != "mta-testing-lennard-jones") {
    return nullptr;
  }
  return std::make_unique<LennardJones>(parse_options(options));
}

MTA_REGISTER_CXX_PLUGIN("lj-plugin", load_model);
