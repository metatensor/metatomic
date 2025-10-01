#include <torch/torch.h>

#include "metatomic/torch/version.h"
#include "metatomic/torch/misc.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
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

std::string pick_device(
    std::vector<std::string> model_devices,
    torch::optional<std::string> desired_device
) {
    auto available_devices = std::vector<std::string>();
    std::string selected_device = "cpu";

    for (const auto& device: model_devices) {
        if (device == "cpu") {
            available_devices.emplace_back("cpu");
        } else if (device == "cuda") {
            if (torch::cuda::is_available()) {
                available_devices.emplace_back("cuda");
            }
        } else if (device == "mps") {
            if (torch::mps::is_available()) {
                available_devices.emplace_back("mps");
            }
        } else {
            TORCH_WARN("'model_devices' contains an entry for unknown device (" + torch::str(device)
                + "). It will be ignored.");
        }
    }

    if (available_devices.empty()) {
        C10_THROW_ERROR(ValueError,
            "failed to find a valid device. None of the devices supported by the model ("
            + torch::str(model_devices) + ") where available (" + torch::str(available_devices) + ")."
        );
    }

    if (desired_device == torch::nullopt) {
        // no user request, pick the device the model prefers
        selected_device = available_devices[0];
    } else {
        bool found_desired_device = false;
        for (const auto& device: available_devices) {
            if (device == desired_device) {
                selected_device = device;
                found_desired_device = true;
                break;
            }
        }

        if (!found_desired_device) {
            C10_THROW_ERROR(ValueError,
                "failed to find requested device (" + torch::str(desired_device.value()) + 
                "): it is either not supported by this model or not available on this machine"
            );
        }
    }
    return selected_device;
}

// ---------- small helpers and validators

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

static void write_system_to_zip(io::ZipWriter& zw, const System& system) {
  using metatensor_torch::TensorBlock;
  using metatensor_torch::TensorMap;

  // positions, cell, types, pbc -> .npy (stored/no compression)
    auto bytes = io::npy_write(system->positions().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("positions.npy", bytes, io::ZIP_STORED);
    bytes = io::npy_write(system->cell().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("cell.npy", bytes, io::ZIP_STORED);
    bytes = io::npy_write(system->types().contiguous().cpu());
    zw.add_file("types.npy", bytes, io::ZIP_STORED);
    bytes = io::npy_write(system->pbc().contiguous().to(torch::kBool).cpu());
    zw.add_file("pbc.npy", bytes, io::ZIP_STORED);

  // Neighbor lists
  {
    auto nls = system->known_neighbor_lists(); // std::vector<NeighborListOptions>
    for (size_t i = 0; i < nls.size(); ++i) {
      const auto& opts = nls[i];
      const std::string base = "pairs/" + std::to_string(i) + "/";

      // options.json (JSON string)
      std::string json = opts->to_json();
      zw.add_file(base + "options.json",
                  reinterpret_cast<const uint8_t*>(json.data()), json.size(), io::ZIP_STORED);

      // data.mts (TensorBlock bytes)
      TensorBlock block = system->get_neighbor_list(opts);
      torch::Tensor mts_buf = metatensor_torch::save_buffer(block).contiguous().cpu();
      zw.add_file(base + "data.mts",
                  mts_buf.data_ptr(), static_cast<size_t>(mts_buf.numel()), io::ZIP_STORED);
    }
  }

  // Extra data (TensorMap)
  {
    auto keys = system->known_data(); // std::vector<std::string>
    for (const auto& key : keys) {
      TensorMap tmap = system->get_data(key);
      torch::Tensor mts_buf = metatensor_torch::save_buffer(tmap).contiguous().cpu();
      zw.add_file("data/" + key + ".mts",
                  mts_buf.data_ptr(), static_cast<size_t>(mts_buf.numel()), io::ZIP_STORED);
    }
  }
}

void save(const std::string& path, const System& system) {
  require_mta_extension(path);
  io::ZipWriter zw(path);
  write_system_to_zip(zw, system);
  zw.finalize();
}

torch::Tensor save_buffer(const System& system) {
  io::ZipWriter zw(0);
  write_system_to_zip(zw, system);
  auto bytes = zw.finalize_to_vector();

  auto bytes_sp = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
  auto bytes_as_tensor = torch::from_blob(
    bytes_sp->data(),
    { static_cast<long>(bytes_sp->size()) },
    [bytes_sp](void*) mutable { bytes_sp.reset(); },
    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
  );

  return bytes_as_tensor;
}

// ---------- read: ZIP -> System (file/memory)

static System read_system_from_zip(io::ZipReader& zr) {
  using metatensor_torch::TensorBlock;
  using metatensor_torch::TensorMap;

  // Validate required files exist
  if (!(zr.has("positions.npy") && zr.has("cell.npy") &&
        zr.has("types.npy") && zr.has("pbc.npy"))) {
    throw std::runtime_error("File does not contain a valid System object (.npy core missing)");
  }

  // Load core arrays
  auto pos   = io::npy_read(zr.read("positions.npy"));
  auto cell  = io::npy_read(zr.read("cell.npy"));
  auto types = io::npy_read(zr.read("types.npy"));
  auto pbc   = io::npy_read(zr.read("pbc.npy"));

  // Construct System (constructor: types, positions, cell, pbc)
  System system = torch::make_intrusive<SystemHolder>(types, pos, cell, pbc);

  // Neighbor lists: find all "pairs/<idx>/options.json"
  std::vector<std::string> names = zr.list();
  struct NLRec { size_t idx; std::string options_path; std::string data_path; };
  std::vector<NLRec> recs;

  for (const auto& name : names) {
    if (starts_with(name, "pairs/") && ends_with(name, "/options.json")) {
      const auto after_pairs = name.substr(6); // "<idx>/options.json"
      auto slash = after_pairs.find('/');
      if (slash == std::string::npos) {
        throw std::runtime_error("malformed neighbor list path");
      }
      auto idx_str = after_pairs.substr(0, slash);
      size_t idx = static_cast<size_t>(std::stoul(idx_str));
      recs.push_back({ idx, name, "pairs/" + std::to_string(idx) + "/data.mts" });
    }
  }

  for (const auto& r : recs) {
    // options.json -> NeighborListOptions
    auto options_bytes = zr.read(r.options_path);
    std::string json(reinterpret_cast<const char*>(options_bytes.data()), options_bytes.size());
    NeighborListOptions opts = NeighborListOptionsHolder::from_json(json);

    // data.mts -> TensorBlock (zero-copy into tensor with captured lifetime)
    auto data_bytes_sp = std::make_shared<std::vector<uint8_t>>(zr.read(r.data_path));
    auto t = torch::from_blob(
        data_bytes_sp->data(),
        { static_cast<long>(data_bytes_sp->size()) },
        [data_bytes_sp](void*) mutable { data_bytes_sp.reset(); },
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );

    TensorBlock block = metatensor_torch::load_block_buffer(t);
    system->add_neighbor_list(opts, block);
  }

  // Extra data under data/*.mts
  for (const auto& name : names) {
    if (starts_with(name, "data/") && ends_with(name, ".mts")) {
      const auto stem = name.substr(5); // "<key>.mts"
      const auto dot = stem.rfind('.');
      if (dot == std::string::npos) {
        throw std::runtime_error("malformed extra data path");
      }
      const auto key = stem.substr(0, dot);

      auto bytes_sp = std::make_shared<std::vector<uint8_t>>(std::move(zr.read(name)));
      auto t = torch::from_blob(
          bytes_sp->data(),
          { static_cast<long>(bytes_sp->size()) },
          [bytes_sp](void*) mutable { bytes_sp.reset(); },
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
  io::ZipReader zr(path);
  return read_system_from_zip(zr);
}

System load_system_buffer(const uint8_t* data, size_t size) {
  io::ZipReader zr(data, size);
  return read_system_from_zip(zr);
}

} // namespace metatomic_torch
