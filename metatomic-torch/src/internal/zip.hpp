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
 * RAII wrapper around miniz for writing ZIP archives.
 * Can target a file or an in-memory buffer.
 */
class ZipWriter {
public:
  ZipWriter();
  ~ZipWriter();

  // Construct and open to a file
  explicit ZipWriter(const std::string& path, unsigned int flags = 0);

  // Construct and open to an in-memory buffer
  explicit ZipWriter(size_t initial_allocation_size, unsigned int flags = 0);

  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;
  ZipWriter(ZipWriter&& other) noexcept;
  ZipWriter& operator=(ZipWriter&& other) noexcept;

  // Add a file from memory into the archive at name_in_zip
  void add_file(const std::string& name_in_zip, const void* data, size_t size, unsigned int level = ZIP_STORED);
  void add_file(const std::string& name_in_zip, const std::vector<uint8_t>& buf, unsigned int level = ZIP_STORED) {
    add_file(name_in_zip, buf.data(), buf.size(), level);
  }

  // For file targets: finalize and close (writes central directory)
  void finalize();

  // For memory targets: finalize and return the whole archive as bytes, then close
  std::vector<uint8_t> finalize_to_vector();

  bool is_open() const noexcept { return opened_; }
  bool is_memory() const noexcept { return target_is_heap_; }

private:
  struct Impl;
  Impl* impl_ = nullptr;
  bool opened_ = false;
  bool target_is_heap_ = false;
  bool finalized_ = false;

  // Internal helpers used by constructors
  void open_file(const std::string& path, unsigned int flags);
  void open_memory(size_t initial_allocation_size, unsigned int flags);
  void close_no_throw();
};


/**
 * RAII wrapper around miniz for reading ZIP archives.
 * Can read from a file or an in-memory buffer.
 */
class ZipReader {
public:
  ZipReader();
  ~ZipReader();

  // Construct and open from a file
  explicit ZipReader(const std::string& path);

  // Construct and open from an in-memory buffer
  explicit ZipReader(const void* data, size_t size, unsigned int flags = 0);

  ZipReader(const ZipReader&) = delete;
  ZipReader& operator=(const ZipReader&) = delete;
  ZipReader(ZipReader&& other) noexcept;
  ZipReader& operator=(ZipReader&& other) noexcept;

  bool has(const std::string& name_in_zip) const;
  std::vector<uint8_t> read(const std::string& name_in_zip) const;
  std::vector<std::string> list() const;

  bool is_open() const noexcept { return opened_; }
  bool is_memory() const noexcept { return opened_from_mem_; }

private:
  struct Impl;
  Impl* impl_ = nullptr;
  bool opened_ = false;
  bool opened_from_mem_ = false;

  // Internal helpers used by constructors
  void open_file(const std::string& path);
  void open_memory(const void* data, size_t size, unsigned int flags);
  void close_no_throw();
};

} // namespace io
} // namespace metatomic_torch
