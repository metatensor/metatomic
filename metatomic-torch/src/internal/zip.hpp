#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace metatomic_torch {
namespace io {

// Explicit constant to request STORED (no compression) in miniz
inline constexpr unsigned ZIP_STORED = 0;

/**
 * RAII wrappers around miniz with:
 *  - default "stored" (no compression) level
 *  - file and in-memory (BytesIO-like) modes
 */
class ZipWriter {
public:
  ZipWriter();
  ~ZipWriter();

  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;
  ZipWriter(ZipWriter&& other) noexcept;
  ZipWriter& operator=(ZipWriter&& other) noexcept;

  // Open a zip archive that writes directly to a file on disk.
  // flags are miniz writer flags (0 is fine).
  void open_file(const std::string& path, unsigned int flags = 0);

  // Open a zip archive that writes to a heap buffer (in-memory).
  // initial_allocation_size is a hint; 0 is fine.
  // flags are miniz writer flags (0 is fine).
  void open_memory(size_t initial_allocation_size = 0, unsigned int flags = 0);

  // Add a file from memory into the archive at name_in_zip.
  // Default level = 0 -> "stored" (no compression), matching the Python behavior.
  void add_file(const std::string& name_in_zip, const void* data, size_t size, unsigned int level = 0);
  void add_file(const std::string& name_in_zip, const std::vector<uint8_t>& buf, unsigned int level = 0) {
    add_file(name_in_zip, buf.data(), buf.size(), level);
  }

  // For file targets: finalize and close (writes central directory).
  void finalize();

  // For memory targets: finalize and return the whole archive as bytes, then close.
  std::vector<uint8_t> finalize_to_vector();

  bool is_open() const noexcept { return opened_; }
  bool is_memory() const noexcept { return target_is_heap_; }

private:
  struct Impl;
  Impl* impl_;
  bool opened_ = false;
  bool target_is_heap_ = false;
  bool finalized_ = false;

  void close_no_throw();
};

class ZipReader {
public:
  ZipReader();
  ~ZipReader();

  ZipReader(const ZipReader&) = delete;
  ZipReader& operator=(const ZipReader&) = delete;
  ZipReader(ZipReader&& other) noexcept;
  ZipReader& operator=(ZipReader&& other) noexcept;

  // Open an archive from a file on disk.
  void open_file(const std::string& path);

  // Open an archive from a memory buffer (BytesIO-like).
  // The buffer must remain valid for the lifetime of the ZipReader.
  void open_memory(const void* data, size_t size, unsigned int flags = 0);

  bool has(const std::string& name_in_zip) const;
  std::vector<uint8_t> read(const std::string& name_in_zip) const;
  std::vector<std::string> list() const;

  bool is_open() const noexcept { return opened_; }
  bool is_memory() const noexcept { return opened_from_mem_; }

private:
  struct Impl;
  Impl* impl_;
  bool opened_ = false;
  bool opened_from_mem_ = false;

  void close_no_throw();
};

} // namespace io
} // namespace metatomic_torch
