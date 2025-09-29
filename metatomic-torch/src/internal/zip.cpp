#include "zip.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <utility>

#include "miniz.h"

struct mta_zip_archive: public mz_zip_archive {};

using namespace metatomic_torch;

static std::runtime_error mz_error(const std::string& where, mz_zip_archive* zip = nullptr) {
    mz_zip_error err = zip != nullptr ? mz_zip_get_last_error(zip) : MZ_ZIP_UNDEFINED_ERROR;
    const char* msg = mz_zip_get_error_string(err);
    std::string full = where + " failed: ";
    full += (msg != nullptr ? msg : "unknown error");
    return std::runtime_error(full);
}

// ============================================================================
// ZipWriter
// ============================================================================

ZipWriter::ZipWriter(const std::string& path):
    zip_(std::make_unique<mta_zip_archive>())
{
    mz_zip_zero_struct(zip_.get());

    auto status = mz_zip_writer_init_file_v2(
        zip_.get(),
        path.c_str(),
        /*size_to_reserve_at_beginning=*/ 0,
        /*flags=*/ 0
    );
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_writer_init_file_v2", zip_.get());
    }

    in_memory_ = false;
    finalized_ = false;
}

ZipWriter::ZipWriter(size_t initial_allocation_size):
    zip_(std::make_unique<mta_zip_archive>())
{
    mz_zip_zero_struct(zip_.get());

    auto status = mz_zip_writer_init_heap_v2(
        zip_.get(),
        /*size_to_reserve_at_beginning=*/ 0,
        initial_allocation_size,
        /*flags=*/ 0
    );
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_writer_init_heap_v2", zip_.get());
    }

    in_memory_ = true;
    finalized_ = false;
}

ZipWriter::~ZipWriter() {
    close_no_throw();
}

ZipWriter::ZipWriter(ZipWriter&& other) noexcept: zip_(nullptr) {
    *this = std::move(other);
}

ZipWriter& ZipWriter::operator=(ZipWriter&& other) noexcept {
    close_no_throw();

    std::swap(zip_, other.zip_);
    std::swap(in_memory_, other.in_memory_);
    std::swap(finalized_, other.finalized_);

    return *this;
}

void ZipWriter::add_file(const std::string& name_in_zip, const void* data, size_t size) {
    if (finalized_) {
        throw std::runtime_error("ZipWriter::add_file: archive already finalized");
    }

    if (data == nullptr && size != 0) {
        throw std::invalid_argument("ZipWriter::add_file: null data with non-zero size");
    }

    // Deterministic mtime: set to epoch (0)
    auto ts = (MZ_TIME_T)0;

    auto status = mz_zip_writer_add_mem_ex_v2(
        zip_.get(),
        name_in_zip.c_str(),
        data,
        size,
        /*pComment*/ nullptr,
        /*comment_size*/ 0,
        /*level_and_flags*/ ZIP_STORED,
        /*uncomp_size*/ 0,
        /*uncomp_crc32*/ 0,
        /*last_modified*/ &ts,
        /*user_extra_data_local*/ nullptr,
        /*user_extra_data_local_len*/ 0,
        /*user_extra_data_central*/ nullptr,
        /*user_extra_data_central_len*/ 0
    );

    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_writer_add_mem_ex_v2", zip_.get());
    }
}

void ZipWriter::finalize() {
    if (finalized_) {
        return;
    }

    if (in_memory_) {
        // We don't return bytes here; for heap-based archives, prefer finalize_to_vector().
        void* p = nullptr;
        size_t n = 0;

        auto status = mz_zip_writer_finalize_heap_archive(zip_.get(), &p, &n);
        if (status == MZ_FALSE) {
            throw mz_error("mz_zip_writer_finalize_heap_archive", zip_.get());
        }

        if (p != nullptr) {
            MZ_FREE(p);
        }
    } else {
        auto status = mz_zip_writer_finalize_archive(zip_.get());
        if (status == MZ_FALSE) {
            throw mz_error("mz_zip_writer_finalize_archive", zip_.get());
        }
    }

    auto status = mz_zip_writer_end(zip_.get());
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_writer_end", zip_.get());
    }

    finalized_ = true;
}

std::vector<uint8_t> ZipWriter::finalize_to_vector() {
    if (finalized_) {
        return {};
    }

    if (!in_memory_) {
        throw std::runtime_error("ZipWriter::finalize_to_vector: not a memory archive; use finalize()");
    }

    void* p = nullptr;
    size_t n = 0;
    auto status = mz_zip_writer_finalize_heap_archive(zip_.get(), &p, &n);
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_writer_finalize_heap_archive", zip_.get());
    }

    status = mz_zip_writer_end(zip_.get());
    if (status == MZ_FALSE) {
        if (p != nullptr) {
            MZ_FREE(p);
        }
        throw mz_error("mz_zip_writer_end", zip_.get());
    }

    finalized_ = true;

    std::vector<uint8_t> out(n);
    if (n != 0 && p != nullptr) {
        std::memcpy(out.data(), p, n);
    }

    if (p != nullptr) {
        MZ_FREE(p);
    }

    return out;
}

void ZipWriter::close_no_throw() {
    if (!finalized_) {
        // we don't try to finalize in-memory archive, since we will not use the
        // data. We try to create non-broken on-disk archives, even though this
        // should not be called (instead user should call finalize()).
        if (!in_memory_) {
            mz_zip_writer_finalize_archive(zip_.get());
        }
        mz_zip_writer_end(zip_.get());
    }
}

// ============================================================================
// ZipReader
// ============================================================================

ZipReader::ZipReader(const std::string& path):
    zip_(std::make_unique<mta_zip_archive>())
{
    mz_zip_zero_struct(zip_.get());
    buffer_ = nullptr;
    buffer_size_ = 0;

    auto status = mz_zip_reader_init_file(zip_.get(), path.c_str(), 0);
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_reader_init_file", zip_.get());
    }

    in_memory_ = false;
}

ZipReader::ZipReader(const void* data, size_t size):
    zip_(std::make_unique<mta_zip_archive>())
{
    if (data == nullptr && size != 0) {
        throw std::invalid_argument("ZipReader::open_memory: null data with non-zero size");
    }

    mz_zip_zero_struct(zip_.get());
    buffer_ = data;
    buffer_size_ = size;

    auto status = mz_zip_reader_init_mem(zip_.get(), data, size, /*flags=*/0);
    if (status == MZ_FALSE) {
        throw mz_error("mz_zip_reader_init_mem", zip_.get());
    }

    in_memory_ = true;
}

ZipReader::~ZipReader() {
    mz_zip_reader_end(zip_.get());
}

ZipReader::ZipReader(ZipReader&& other) noexcept: zip_(nullptr) {
    *this = std::move(other);
}

ZipReader& ZipReader::operator=(ZipReader&& other) noexcept {
    mz_zip_reader_end(zip_.get());

    std::swap(zip_, other.zip_);
    std::swap(buffer_, other.buffer_);
    std::swap(buffer_size_, other.buffer_size_);
    std::swap(in_memory_, other.in_memory_);

    return *this;
}

bool ZipReader::has(const std::string& name_in_zip) const {
    int idx = mz_zip_reader_locate_file(
        zip_.get(),
        name_in_zip.c_str(),
        nullptr,
        0 /* exact, case-sensitive */
    );
    return idx >= 0;
}

std::vector<std::string> ZipReader::list() const {
    std::vector<std::string> out;
    const mz_uint total = mz_zip_reader_get_num_files(zip_.get());
    out.reserve(total);

    char buf[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
    for (mz_uint i = 0; i < total; ++i) {
        mz_uint n = mz_zip_reader_get_filename(zip_.get(), i, buf, sizeof(buf));
        if (n == 0) {
            continue;
        }
        out.emplace_back(buf);
    }
    return out;
}

std::vector<uint8_t> ZipReader::read(const std::string& name_in_zip) const {
    size_t sz = 0;
    void* ptr = mz_zip_reader_extract_file_to_heap(
        zip_.get(),
        name_in_zip.c_str(),
        &sz,
        0 /* flags */
    );

    if (ptr == nullptr) {
        throw mz_error("mz_zip_reader_extract_file_to_heap", zip_.get());
    }

    std::vector<uint8_t> bytes(sz);
    if (sz != 0) {
        std::memcpy(bytes.data(), ptr, sz);
    }
    MZ_FREE(ptr);
    return bytes;
}
