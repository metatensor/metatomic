#include "metatomic/torch/io/io.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "metatomic/torch/io/zip.hpp"
#include "metatomic/torch/io/npy.hpp"

#include "metatomic/torch/system.hpp"        // your header shown above
#include <metatensor/torch.hpp>              // TensorBlock/TensorMap

namespace metatomic_torch {

// ---------- small helpers

static inline void ensure(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

static inline bool ends_with(const std::string& s, const std::string& suff) {
  return s.size() >= suff.size() && s.compare(s.size() - suff.size(), suff.size(), suff) == 0;
}

static inline bool starts_with(const std::string& s, const std::string& pref) {
  return s.size() >= pref.size() && s.compare(0, pref.size(), pref) == 0;
}

static inline void require_mta_extension(const std::string& path) {
  ensure(ends_with(path, ".mta"), "The provided path must have the `.mta` extension.");
}

// ---------- validators

bool is_system_mta_file(const std::string& path) {
  try {
    io::ZipReader zr;
    zr.open_file(path);
    return zr.has("positions.npy") && zr.has("cell.npy") &&
           zr.has("types.npy") && zr.has("pbc.npy");
  } catch (...) {
    return false;
  }
}

bool is_system_mta_memory(const uint8_t* data, size_t size) {
  try {
    io::ZipReader zr;
    zr.open_memory(data, size, /*flags*/ 0);
    return zr.has("positions.npy") && zr.has("cell.npy") &&
           zr.has("types.npy") && zr.has("pbc.npy");
  } catch (...) {
    return false;
  }
}

// ---------- NeighborListOptions JSON glue (actual API)

static std::string neighbor_options_to_json(const NeighborListOptions& opts) {
  return opts->to_json();
}

static NeighborListOptions neighbor_options_from_json(const std::string& json) {
  return NeighborListOptionsHolder::from_json(json);
}

// ---------- write: System -> ZIP (file/memory)

static void write_system_to_zip(io::ZipWriter& zw, const System& system) {
  using metatensor_torch::TensorBlock;
  using metatensor_torch::TensorMap;

  // positions, cell, types, pbc -> .npy (stored/no compression)
  {
    auto bytes = io::npy_write(system->positions().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("positions.npy", bytes, 0);
  }
  {
    auto bytes = io::npy_write(system->cell().contiguous().to(torch::kFloat64).cpu());
    zw.add_file("cell.npy", bytes, 0);
  }
  {
    auto t = system->types().contiguous().cpu(); // keep i32 or i64
    auto bytes = io::npy_write(t);
    zw.add_file("types.npy", bytes, 0);
  }
  {
    auto bytes = io::npy_write(system->pbc().contiguous().to(torch::kBool).cpu());
    zw.add_file("pbc.npy", bytes, 0);
  }

  // Neighbor lists
  {
    auto nls = system->known_neighbor_lists(); // std::vector<NeighborListOptions>
    for (size_t i = 0; i < nls.size(); ++i) {
      const auto& opts = nls[i];
      const std::string base = "pairs/" + std::to_string(i) + "/";

      // options.json (JSON string)
      std::string json = neighbor_options_to_json(opts);
      zw.add_file(base + "options.json",
                  reinterpret_cast<const uint8_t*>(json.data()), json.size(), 0);

      // data.mts (TensorBlock bytes via member save_buffer)
      TensorBlock block = system->get_neighbor_list(opts);
      torch::Tensor mts_buf = metatensor_torch::save_buffer(block);
      mts_buf = mts_buf.contiguous().cpu();
      zw.add_file(base + "data.mts",
                  mts_buf.data_ptr(), static_cast<size_t>(mts_buf.numel()), 0);
    }
  }

  // Extra data (TensorMap)
  {
    auto keys = system->known_data(); // std::vector<std::string>
    for (const auto& key : keys) {
      TensorMap tmap = system->get_data(key);
      torch::Tensor mts_buf = metatensor_torch::save_buffer(tmap);
      mts_buf = mts_buf.contiguous().cpu();
      zw.add_file("data/" + key + ".mts",
                  mts_buf.data_ptr(), static_cast<size_t>(mts_buf.numel()), 0);
    }
  }
}

void save(const std::string& path, const System& system) {
  require_mta_extension(path);
  io::ZipWriter zw;
  zw.open_file(path, /*flags*/ 0);
  write_system_to_zip(zw, system);
  zw.finalize();
}

torch::Tensor save_buffer(const System& system) {
  io::ZipWriter zw;
  zw.open_memory(/*initial*/ 0, /*flags*/ 0);
  write_system_to_zip(zw, system);
  auto bytes = zw.finalize_to_vector();

  // create a tensor and copy bytes in
  auto bytes_as_tensor = torch::empty(
      { static_cast<long>(bytes.size()) },
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
  );
  if (!bytes.empty()) {
      std::memcpy(bytes_as_tensor.data_ptr<uint8_t>(), bytes.data(), bytes.size());
  }
  return bytes_as_tensor;
}

// ---------- read: ZIP -> System (file/memory)

static System read_system_from_zip(io::ZipReader& zr) {
  using metatensor_torch::TensorBlock;
  using metatensor_torch::TensorMap;

  // Validate required files exist
  ensure(zr.has("positions.npy") && zr.has("cell.npy") &&
         zr.has("types.npy") && zr.has("pbc.npy"),
         "File does not contain a valid System object (.npy core missing)");

  // Load core arrays
  auto pos   = io::npy_read(zr.read("positions.npy")).contiguous().to(torch::kFloat64).cpu();
  auto cell  = io::npy_read(zr.read("cell.npy")).contiguous().to(torch::kFloat64).cpu();
  auto types = io::npy_read(zr.read("types.npy")).contiguous().cpu();
  auto pbc   = io::npy_read(zr.read("pbc.npy")).contiguous().to(torch::kBool).cpu();

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
      ensure(slash != std::string::npos, "malformed neighbor list path");
      auto idx_str = after_pairs.substr(0, slash);
      size_t idx = static_cast<size_t>(std::stoul(idx_str));
      recs.push_back({ idx, name, "pairs/" + std::to_string(idx) + "/data.mts" });
    }
  }
  std::sort(recs.begin(), recs.end(), [](const NLRec& a, const NLRec& b){ return a.idx < b.idx; });

  for (const auto& r : recs) {
    // options.json -> NeighborListOptions
    auto options_bytes = zr.read(r.options_path);
    std::string json(reinterpret_cast<const char*>(options_bytes.data()), options_bytes.size());
    NeighborListOptions opts = neighbor_options_from_json(json);

    // data.mts -> TensorBlock
    auto data_bytes = zr.read(r.data_path);
    auto bytes_sp = std::make_shared<std::vector<uint8_t>>(data_bytes);
    auto t = torch::from_blob(
        bytes_sp->data(),
        { static_cast<long>(bytes_sp->size()) },
        // deleter: called when tensor storage is released
        [bytes_sp](void* /*unused*/) mutable {
            // bytes_sp will be destroyed here; vector frees its memory.
            bytes_sp.reset();
        },
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
      ensure(dot != std::string::npos, "malformed extra data path");
      const auto key = stem.substr(0, dot);

      auto bytes = zr.read(name);
      auto bytes_sp = std::make_shared<std::vector<uint8_t>>(bytes);
      auto t = torch::from_blob(const_cast<uint8_t*>(bytes.data()),
                                { static_cast<long>(bytes.size()) },
                                // deleter: called when tensor storage is released
                                [bytes_sp](void* /*unused*/) mutable {
                                    // bytes_sp will be destroyed here; vector frees its memory.
                                    bytes_sp.reset();
                                },
                                torch::TensorOptions().dtype(torch::kUInt8));

      TensorMap tmap = metatensor_torch::load_buffer(t);
      system->add_data(key, tmap);
    }
  }

  return system;
}

System load_system(const std::string& path) {
  require_mta_extension(path);
  io::ZipReader zr;
  zr.open_file(path);
  return read_system_from_zip(zr);
}

System load_system_buffer(const uint8_t* data, size_t size) {
  io::ZipReader zr;
  zr.open_memory(data, size, /*flags*/ 0);
  return read_system_from_zip(zr);
}

} // namespace metatomic_torch
