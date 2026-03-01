#include <torch/torch.h>

#include "metatomic/torch/model.hpp"
#include "metatomic/torch/version.h"
#include "metatomic/torch/misc.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstring>

#include "internal/zip.hpp"
#include "internal/npy.hpp"

#include "metatomic/torch/system.hpp"
#include <metatensor/torch.hpp>


namespace metatomic_torch {

std::string version() {
    return METATOMIC_TORCH_VERSION;
}

using namespace std::string_literals;

static inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static inline bool available_device(const std::string &n) {
    if (n == "cpu") {
        return true;
    } else if (n == "cuda") {
        return torch::cuda::is_available();
    } else  if (n == "mps") {
        return torch::mps::is_available();
    } else if (n == "hip") {
#ifdef TORCH_ENABLE_HIP
        return torch::hip::is_available();
#else
        return false;
#endif
    } else if (
        n == "xla" || n == "ipu" || n == "xpu" || n == "ve" || n == "opencl" ||
        n == "opengl" || n == "vulkan" || n == "mkldnn" || n == "ideep" ||
        n == "mtia" || n == "meta" || n == "hpu"
    ) {
        // For many backends we can't reliably test availability at runtime;
        // assume existence (caller may still fail when using the device).
        return true;
    } else {
        return false;
    }
}

static inline torch::DeviceType map_to_devicetype(const std::string &n) {
    if (n == "cpu") {
        return torch::DeviceType::CPU;
    } else if (n == "cuda") {
        return torch::DeviceType::CUDA;
    } else if (n == "mps") {
        return torch::DeviceType::MPS;
    } else if (n == "hip") {
        return torch::DeviceType::HIP;
    } else if (n == "xla") {
        return torch::DeviceType::XLA;
    } else if (n == "ipu") {
        return torch::DeviceType::IPU;
    } else if (n == "xpu") {
        return torch::DeviceType::XPU;
    } else if (n == "ve") {
        return torch::DeviceType::VE;
    } else if (n == "opencl") {
        return torch::DeviceType::OPENCL;
    } else if (n == "opengl") {
        return torch::DeviceType::OPENGL;
    } else if (n == "vulkan") {
        return torch::DeviceType::Vulkan;
    } else if (n == "mkldnn") {
        return torch::DeviceType::MKLDNN;
    } else if (n == "ideep") {
        return torch::DeviceType::IDEEP;
    } else if (n == "mtia") {
        return torch::DeviceType::MTIA;
    } else if (n == "meta") {
        return torch::DeviceType::Meta;
    } else if (n == "hpu") {
        return torch::DeviceType::HPU;
    } else {
        C10_THROW_ERROR(ValueError, "failed to find a valid device type for '" + n + "'");
    }
}

static inline bool is_known_device(const std::string &n) {
    return (
        n == "cpu" || n == "cuda" || n == "mps" || n == "hip" || n == "xla" ||
        n == "ipu" || n == "xpu" || n == "ve" || n == "opencl" ||
        n == "opengl" || n == "vulkan" || n == "mkldnn" || n == "ideep" ||
        n == "mtia" || n == "meta" || n == "hpu"
    );
}

c10::DeviceType pick_device(
    std::vector<std::string> model_devices,
    torch::optional<std::string> desired_device
) {
    // build list of available (normalized) device names in order
    std::vector<std::string> available;
    available.reserve(model_devices.size());
    for (auto &d : model_devices) {
        std::string n = lower(d);
        if (!is_known_device(n)) {
            TORCH_WARN("'model_devices' contains an entry for unknown device '" + d + "' (" + n + "); ignoring.");
            continue;
        }
        if (!available_device(n)) {
            continue;
        }
        available.emplace_back(std::move(n));
    }

    if (available.empty()) {
        C10_THROW_ERROR(ValueError,
            "failed to find a valid device. None of the "
            "model-supported devices are available."
        );
    }

    // if no desired device requested, pick first available
    if (!desired_device.has_value() || desired_device->empty()) {
        return map_to_devicetype(available.front());
    }

    // normalize desired and check
    std::string wanted_str = lower(desired_device.value());
    torch::DeviceType wanted_type;
    try {
        wanted_type = torch::Device(wanted_str).type();
    } catch (const std::exception &) {
        C10_THROW_ERROR(ValueError, "invalid device string: " + desired_device.value());
    }

    for (auto &a : available) {
        if (map_to_devicetype(a) == wanted_type) {
            return wanted_type;
        }
    }

    C10_THROW_ERROR(ValueError,
        "failed to find requested device (" + desired_device.value() +
        "): it is either not supported by this model or not available on this machine"
    );
}

std::string pick_output(
    std::string requested_output,
    torch::Dict<std::string, ModelOutput> outputs,
    torch::optional<std::string> desired_variant
) {
    std::vector<std::string> matching_keys;
    bool has_exact = false;

    for (const auto& output: outputs) {
        const auto& key = output.key();

        // match either exact `requested_output` or `requested_output/<variant>`
        if (key == requested_output
            || (key.size() > requested_output.size()
                && key.compare(0, requested_output.size(), requested_output) == 0
                && key[requested_output.size()] == '/')) {
            matching_keys.emplace_back(key);

            if (key == requested_output) {
                has_exact = true;
            }
        }
    }

    if (matching_keys.empty()) {
        C10_THROW_ERROR(ValueError,
            "output '" + requested_output + "' not found in outputs"
        );
    }

    if (desired_variant != torch::nullopt) {
        const auto& output = requested_output + "/" + desired_variant.value();
        auto it = std::find(matching_keys.begin(), matching_keys.end(), output);
        if (it != matching_keys.end()) {
            return *it;
        }
        C10_THROW_ERROR(ValueError,
            "variant '" + desired_variant.value() + "' for output '" + requested_output +
            "' not found in outputs"
        );
    } else if (has_exact) {
        return requested_output;
    } else {
        std::ostringstream oss;
        oss << "output '" << requested_output << "' has no default variant and no `desired_variant` was given. Available variants are:";

        size_t maxlen = 0;
        for (const auto& key: matching_keys) {
            maxlen = std::max(key.size(), maxlen);
        }

        for (const auto& key: matching_keys) {
            auto description = outputs.at(key)->description;
            std::string padding(maxlen - key.size(), ' ');
            oss << "\n - '" << key << "'" << padding << ": " << description;
        }
        C10_THROW_ERROR(ValueError, oss.str());
    }
}


static bool ends_with(const std::string& s, const std::string& suff) {
    return s.size() >= suff.size() && s.compare(s.size() - suff.size(), suff.size(), suff) == 0;
}

static bool starts_with(const std::string& s, const std::string& pref) {
    return s.size() >= pref.size() && s.compare(0, pref.size(), pref) == 0;
}

static void require_mta_extension(const std::string& path) {
    if (!ends_with(path, ".mta")) {
        throw std::runtime_error("The provided path must have the `.mta` extension.");
    }
}

// ---------- write: System -> ZIP (file/memory)

static void write_system_to_zip(ZipWriter& zw, const System& system) {
    using metatensor_torch::TensorBlock;
    using metatensor_torch::TensorMap;

    // positions, cell, types, pbc -> .npy (stored/no compression)
    auto bytes = npy_write(system->positions().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("positions.npy", bytes);

    bytes = npy_write(system->cell().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("cell.npy", bytes);

    bytes = npy_write(system->types().contiguous().cpu());
    zw.add_file("types.npy", bytes);

    bytes = npy_write(system->pbc().contiguous().to(torch::kBool).cpu());
    zw.add_file("pbc.npy", bytes);

    // Neighbor lists
    auto all_neighbors = system->known_neighbor_lists();
    for (size_t i = 0; i < all_neighbors.size(); ++i) {
        const auto& options = all_neighbors[i];
        const std::string base = "pairs/" + std::to_string(i) + "/";

        // options.json (JSON string)
        std::string json = options->to_json();
        zw.add_file(
            base + "options.json",
            reinterpret_cast<const uint8_t*>(json.data()),
            json.size()
        );

        // data.mts (TensorBlock bytes)
        TensorBlock block = system->get_neighbor_list(options);
        torch::Tensor mts_buf = metatensor_torch::save_buffer(block).contiguous().cpu();
        zw.add_file(
            base + "data.mts",
            mts_buf.data_ptr(),
            static_cast<size_t>(mts_buf.numel())
        );
    }

    // Extra data
    auto keys = system->known_data();
    for (const auto& key: keys) {
        TensorMap tmap = system->get_data(key);
        torch::Tensor mts_buf = metatensor_torch::save_buffer(tmap).contiguous().cpu();
        zw.add_file(
            "data/" + key + ".mts",
            mts_buf.data_ptr(),
            static_cast<size_t>(mts_buf.numel())
        );
    }
}

void save(const std::string& path, const System& system) {
    require_mta_extension(path);
    ZipWriter zw(path);
    write_system_to_zip(zw, system);
    zw.finalize();
}

torch::Tensor save_buffer(const System& system) {
    ZipWriter zw(0);
    write_system_to_zip(zw, system);
    auto bytes = zw.finalize_to_vector();

    auto bytes_sp = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
    auto bytes_as_tensor = torch::from_blob(
        bytes_sp->data(),
        {static_cast<int64_t>(bytes_sp->size())},
        [bytes_sp](void*) mutable {
            bytes_sp.reset();
        },
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );

    return bytes_as_tensor;
}

static System read_system_from_zip(ZipReader& zr) {
    using metatensor_torch::TensorBlock;
    using metatensor_torch::TensorMap;

    // Validate required files exist
    if (!(zr.has("positions.npy") && zr.has("cell.npy") && zr.has("types.npy") && zr.has("pbc.npy"))) {
        throw std::runtime_error("File does not contain a valid System object (.npy core missing)");
    }

    // Load core arrays
    auto positions = npy_read(zr.read("positions.npy"));
    auto cell = npy_read(zr.read("cell.npy"));
    auto types = npy_read(zr.read("types.npy"));
    auto pbc = npy_read(zr.read("pbc.npy"));

    System system = torch::make_intrusive<SystemHolder>(types, positions, cell, pbc);

    // Neighbor lists: find all "pairs/<idx>/options.json"
    std::vector<std::string> names = zr.list();
    struct NeighborsInZip {
        size_t idx;
        std::string options_path;
        std::string data_path;
    };
    std::vector<NeighborsInZip> entries;

    for (const auto& name : names) {
        if (starts_with(name, "pairs/") && ends_with(name, "/options.json")) {
        const auto after_pairs = name.substr(6); // "<idx>/options.json"
        auto slash = after_pairs.find('/');
        if (slash == std::string::npos) {
            throw std::runtime_error("malformed neighbor list path");
        }
        auto idx_str = after_pairs.substr(0, slash);
        size_t idx = static_cast<size_t>(std::stoul(idx_str));
        entries.push_back({ idx, name, "pairs/" + std::to_string(idx) + "/data.mts" });
        }
    }

    for (const auto& entry: entries) {
        // options.json -> NeighborListOptions
        auto options_bytes = zr.read(entry.options_path);
        std::string json(reinterpret_cast<const char*>(options_bytes.data()), options_bytes.size());
        NeighborListOptions opts = NeighborListOptionsHolder::from_json(json);

        // data.mts -> TensorBlock (zero-copy into tensor with captured lifetime)
        auto data_bytes_sp = std::make_shared<std::vector<uint8_t>>(zr.read(entry.data_path));
        auto t = torch::from_blob(
            data_bytes_sp->data(),
            {static_cast<int64_t>(data_bytes_sp->size())},
            [data_bytes_sp](void*) mutable {
                data_bytes_sp.reset();
            },
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
        );

        TensorBlock block = metatensor_torch::load_block_buffer(t);
        system->add_neighbor_list(opts, block);
    }

    // Extra data under data/*.mts
    for (const auto& name : names) {
        if (starts_with(name, "data/") && ends_with(name, ".mts")) {
            auto stem = name.substr(5); // "<key>.mts"
            auto dot = stem.rfind('.');
            if (dot == std::string::npos) {
                throw std::runtime_error("malformed extra data path");
            }
            const auto key = stem.substr(0, dot);

            auto bytes_sp = std::make_shared<std::vector<uint8_t>>(std::move(zr.read(name)));
            auto t = torch::from_blob(
                bytes_sp->data(),
                {static_cast<int64_t>(bytes_sp->size())},
                [bytes_sp](void*) mutable {
                    bytes_sp.reset();
                },
                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
            );

            TensorMap tmap = metatensor_torch::load_buffer(t);
            system->add_data(key, tmap);
        }
    }

    return system;
}

System load_system(const std::string& path) {
    require_mta_extension(path);
    ZipReader zr(path);
    return read_system_from_zip(zr);
}

System load_system_buffer(const uint8_t* data, size_t size) {
    ZipReader zr(data, size);
    return read_system_from_zip(zr);
}


/// Known inputs and outputs
inline std::unordered_set<std::string> KNOWN_INPUTS_OUTPUTS = {
    "energy",
    "energy_ensemble",
    "energy_uncertainty",
    "features",
    "non_conservative_forces",
    "non_conservative_stress",
    "positions",
    "momenta",
    "velocities",
    "masses",
    "charges",
};

std::tuple<bool, std::string, std::string> details::validate_name_and_check_variant(
    const std::string& name
) {
    if (KNOWN_INPUTS_OUTPUTS.find(name) != KNOWN_INPUTS_OUTPUTS.end()) {
        // known output, nothing to do
        return {true, name, ""};
    }

    auto double_colon = name.rfind("::");
    if (double_colon != std::string::npos) {
        if (double_colon == 0 || double_colon == (name.length() - 2)) {
            C10_THROW_ERROR(ValueError,
                "Invalid name for model output: '" + name + "'. "
                "Non-standard names should look like '<domain>::<output>' "
                "with non-empty domain and output."
            );
        }

        auto custom_name = name.substr(0, double_colon);
        auto output_name = name.substr(double_colon + 2);

        auto slash = custom_name.find('/');
        if (slash != std::string::npos) {
            // "domain/variant::custom" is not allowed
            C10_THROW_ERROR(ValueError,
                "Invalid name for model output: '" + name + "'. "
                "Non-standard name with variant should look like "
                "'<domain>::<output>/<variant>'"
            );
        }

        slash = output_name.find('/');
        if (slash != std::string::npos) {
            if (slash == 0 || slash == (name.length() - 1)) {
            C10_THROW_ERROR(ValueError,
                    "Invalid name for model output: '" + name + "'. "
                    "Non-standard name with variant should look like "
                    "'<domain>::<output>/<variant>' with non-empty domain, "
                    "output and variant."
                );
            }
        }

        // this is a custom output, nothing more to check
        return {false, "", ""};
    }

    auto slash = name.find('/');
    if (slash != std::string::npos) {
        if (slash == 0 || slash == (name.length() - 1)) {
            C10_THROW_ERROR(ValueError,
                "Invalid name for model output: '" + name + "'. "
                "Variant names should look like '<output>/<variant>' "
                "with non-empty output and variant."
            );
        }

        auto base = name.substr(0, slash);
        auto double_colon = base.rfind("::");
        if (double_colon != std::string::npos) {
            // we don't do anything for custom outputs
            return {false, "", ""};
        }

        if (KNOWN_INPUTS_OUTPUTS.find(base) == KNOWN_INPUTS_OUTPUTS.end()) {
            C10_THROW_ERROR(ValueError,
                "Invalid name for model output with variant: '" + name + "'. "
                "'" + base + "' is not a known output."
            );
        }

        return {true, base, name};
    }

    C10_THROW_ERROR(ValueError,
        "Invalid name for model output: '" + name + "' is not a known output. "
        "Variant names should be of the form '<output>/<variant>'. "
        "Non-standard names should have the form '<domain>::<output>'."
    );
}

} // namespace metatomic_torch
