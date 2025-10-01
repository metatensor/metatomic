#include "zip.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <utility>

#include "miniz.h"

namespace metatomic_torch {
namespace io {

// ----------------- helpers

static std::runtime_error mz_error(const std::string& where, mz_zip_archive* zip = nullptr) {
  mz_zip_error err = zip ? mz_zip_get_last_error(zip) : MZ_ZIP_UNDEFINED_ERROR;
  const char* msg = mz_zip_get_error_string(err);
  std::string full = where + " failed: ";
  full += (msg ? msg : "unknown error");
  return std::runtime_error(full);
}

// ============================================================================
// ZipWriter
// ============================================================================

struct ZipWriter::Impl {
  mz_zip_archive zip{};
  std::string path;
};

ZipWriter::ZipWriter() : impl_(new Impl()) {
  mz_zip_zero_struct(&impl_->zip);
}

ZipWriter::ZipWriter(const std::string& path, unsigned int flags)
  : ZipWriter() {
  open_file(path, flags);
}

ZipWriter::ZipWriter(size_t initial_allocation_size, unsigned int flags)
  : ZipWriter() {
  open_memory(initial_allocation_size, flags);
}

ZipWriter::~ZipWriter() {
  close_no_throw();
  delete impl_;
  impl_ = nullptr;
}

ZipWriter::ZipWriter(ZipWriter&& other) noexcept
  : impl_(std::exchange(other.impl_, nullptr)),
    opened_(std::exchange(other.opened_, false)),
    target_is_heap_(std::exchange(other.target_is_heap_, false)),
    finalized_(std::exchange(other.finalized_, false)) {}

ZipWriter& ZipWriter::operator=(ZipWriter&& other) noexcept {
  if (this != &other) {
    close_no_throw();
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
    opened_ = std::exchange(other.opened_, false);
    target_is_heap_ = std::exchange(other.target_is_heap_, false);
    finalized_ = std::exchange(other.finalized_, false);
  }
  return *this;
}

void ZipWriter::open_file(const std::string& path, unsigned int flags) {
  if (opened_) throw std::runtime_error("ZipWriter: archive already open");
  mz_zip_zero_struct(&impl_->zip);
  impl_->path = path;

  if (!mz_zip_writer_init_file_v2(&impl_->zip, path.c_str(), 0 /*reserve*/, flags)) {
    throw mz_error("mz_zip_writer_init_file_v2", &impl_->zip);
  }

  opened_ = true;
  target_is_heap_ = false;
  finalized_ = false;
}

void ZipWriter::open_memory(size_t initial_allocation_size, unsigned int flags) {
  if (opened_) throw std::runtime_error("ZipWriter: archive already open");
  mz_zip_zero_struct(&impl_->zip);

  if (!mz_zip_writer_init_heap_v2(&impl_->zip, 0 /*reserve*/, initial_allocation_size, flags)) {
    throw mz_error("mz_zip_writer_init_heap_v2", &impl_->zip);
  }

  opened_ = true;
  target_is_heap_ = true;
  finalized_ = false;
}

void ZipWriter::add_file(const std::string& name_in_zip,
                         const void* data, size_t size,
                         unsigned int level) {
  if (!opened_) throw std::runtime_error("ZipWriter::add_file: archive not open");
  if (finalized_) throw std::runtime_error("ZipWriter::add_file: archive already finalized");
  if (!data && size != 0) throw std::invalid_argument("ZipWriter::add_file: null data with non-zero size");

  // Deterministic mtime: set to epoch (0)
  MZ_TIME_T ts = (MZ_TIME_T)0;

  if (!mz_zip_writer_add_mem_ex_v2(
          &impl_->zip,
          name_in_zip.c_str(),
          data,
          size,
          /*pComment*/ nullptr,
          /*comment_size*/ 0,
          /*level_and_flags*/ level,   // e.g., io::ZIP_STORED
          /*uncomp_size*/ 0,
          /*uncomp_crc32*/ 0,
          /*last_modified*/ &ts,
          /*user_extra_data_local*/ nullptr,
          /*user_extra_data_local_len*/ 0,
          /*user_extra_data_central*/ nullptr,
          /*user_extra_data_central_len*/ 0)) {
    throw mz_error("mz_zip_writer_add_mem_ex_v2", &impl_->zip);
  }
}

void ZipWriter::finalize() {
  if (!opened_) return;
  if (finalized_) return;

  if (target_is_heap_) {
    // We don't return bytes here; for heap-based archives, prefer finalize_to_vector().
    void* p = nullptr; size_t n = 0;
    if (!mz_zip_writer_finalize_heap_archive(&impl_->zip, &p, &n)) {
      throw mz_error("mz_zip_writer_finalize_heap_archive", &impl_->zip);
    }
    if (p) MZ_FREE(p);
  } else {
    if (!mz_zip_writer_finalize_archive(&impl_->zip)) {
      throw mz_error("mz_zip_writer_finalize_archive", &impl_->zip);
    }
  }

  if (!mz_zip_writer_end(&impl_->zip)) {
    throw mz_error("mz_zip_writer_end", &impl_->zip);
  }

  opened_ = false;
  finalized_ = true;
}

std::vector<uint8_t> ZipWriter::finalize_to_vector() {
  if (!opened_) return {};
  if (finalized_) return {};

  if (!target_is_heap_) {
    throw std::runtime_error("ZipWriter::finalize_to_vector: not a memory archive; use finalize()");
  }

  void* p = nullptr;
  size_t n = 0;
  if (!mz_zip_writer_finalize_heap_archive(&impl_->zip, &p, &n)) {
    throw mz_error("mz_zip_writer_finalize_heap_archive", &impl_->zip);
  }
  if (!mz_zip_writer_end(&impl_->zip)) {
    if (p) MZ_FREE(p);
    throw mz_error("mz_zip_writer_end", &impl_->zip);
  }

  opened_ = false;
  finalized_ = true;

  std::vector<uint8_t> out(n);
  if (n && p) std::memcpy(out.data(), p, n);
  if (p) MZ_FREE(p);
  return out;
}

void ZipWriter::close_no_throw() {
  if (opened_) {
    if (target_is_heap_) {
      void* p = nullptr; size_t n = 0;
      (void)mz_zip_writer_finalize_heap_archive(&impl_->zip, &p, &n);
      if (p) MZ_FREE(p);
    } else {
      (void)mz_zip_writer_finalize_archive(&impl_->zip);
    }
    (void)mz_zip_writer_end(&impl_->zip);
    opened_ = false;
  } else {
    if (mz_zip_get_mode(&impl_->zip) == MZ_ZIP_MODE_WRITING) {
      (void)mz_zip_writer_end(&impl_->zip);
    }
  }
}

// ============================================================================
// ZipReader
// ============================================================================

struct ZipReader::Impl {
  mz_zip_archive zip{};
  std::string path;
  const void* mem_ptr = nullptr; // not owned
  size_t mem_size = 0;
};

ZipReader::ZipReader() : impl_(new Impl()) {
  mz_zip_zero_struct(&impl_->zip);
}

ZipReader::ZipReader(const std::string& path)
  : ZipReader() {
  open_file(path);
}

ZipReader::ZipReader(const void* data, size_t size, unsigned int flags)
  : ZipReader() {
  open_memory(data, size, flags);
}

ZipReader::~ZipReader() {
  close_no_throw();
  delete impl_;
  impl_ = nullptr;
}

ZipReader::ZipReader(ZipReader&& other) noexcept
  : impl_(std::exchange(other.impl_, nullptr)),
    opened_(std::exchange(other.opened_, false)),
    opened_from_mem_(std::exchange(other.opened_from_mem_, false)) {}

ZipReader& ZipReader::operator=(ZipReader&& other) noexcept {
  if (this != &other) {
    close_no_throw();
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
    opened_ = std::exchange(other.opened_, false);
    opened_from_mem_ = std::exchange(other.opened_from_mem_, false);
  }
  return *this;
}

void ZipReader::open_file(const std::string& path) {
  if (opened_) throw std::runtime_error("ZipReader: archive already open");
  mz_zip_zero_struct(&impl_->zip);
  impl_->path = path;
  impl_->mem_ptr = nullptr; impl_->mem_size = 0;

  if (!mz_zip_reader_init_file(&impl_->zip, path.c_str(), 0)) {
    throw mz_error("mz_zip_reader_init_file", &impl_->zip);
  }

  opened_ = true;
  opened_from_mem_ = false;
}

void ZipReader::open_memory(const void* data, size_t size, unsigned int flags) {
  if (opened_) throw std::runtime_error("ZipReader: archive already open");
  if (!data && size) throw std::invalid_argument("ZipReader::open_memory: null data with non-zero size");

  mz_zip_zero_struct(&impl_->zip);
  impl_->path.clear();
  impl_->mem_ptr = data;
  impl_->mem_size = size;

  if (!mz_zip_reader_init_mem(&impl_->zip, data, size, flags)) {
    throw mz_error("mz_zip_reader_init_mem", &impl_->zip);
  }

  opened_ = true;
  opened_from_mem_ = true;
}

bool ZipReader::has(const std::string& name_in_zip) const {
  if (!opened_) throw std::runtime_error("ZipReader::has: archive not open");

  int idx = mz_zip_reader_locate_file(const_cast<mz_zip_archive*>(&impl_->zip),
                                      name_in_zip.c_str(),
                                      nullptr,
                                      0 /* exact, case-sensitive */);
  return idx >= 0;
}

std::vector<std::string> ZipReader::list() const {
  if (!opened_) throw std::runtime_error("ZipReader::list: archive not open");

  std::vector<std::string> out;
  const mz_uint total = mz_zip_reader_get_num_files(const_cast<mz_zip_archive*>(&impl_->zip));
  out.reserve(total);

  char buf[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
  for (mz_uint i = 0; i < total; ++i) {
    mz_uint n = mz_zip_reader_get_filename(const_cast<mz_zip_archive*>(&impl_->zip), i, buf, sizeof(buf));
    if (n == 0) continue;
    out.emplace_back(buf);
  }
  return out;
}

std::vector<uint8_t> ZipReader::read(const std::string& name_in_zip) const {
  if (!opened_) throw std::runtime_error("ZipReader::read: archive not open");

  size_t sz = 0;
  void* ptr = mz_zip_reader_extract_file_to_heap(const_cast<mz_zip_archive*>(&impl_->zip),
                                                 name_in_zip.c_str(),
                                                 &sz,
                                                 0 /* flags */);
  if (!ptr) {
    throw mz_error("mz_zip_reader_extract_file_to_heap", const_cast<mz_zip_archive*>(&impl_->zip));
  }

  std::vector<uint8_t> bytes(sz);
  if (sz) std::memcpy(bytes.data(), ptr, sz);
  MZ_FREE(ptr);
  return bytes;
}

void ZipReader::close_no_throw() {
  if (opened_) {
    (void)mz_zip_reader_end(&impl_->zip);
    opened_ = false;
    opened_from_mem_ = false;
  } else {
    if (mz_zip_get_mode(&impl_->zip) == MZ_ZIP_MODE_READING) {
      (void)mz_zip_reader_end(&impl_->zip);
    }
  }
}

} // namespace io
} // namespace metatomic_torch
