#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// We can not use a foward declaration here because mz_zip_archive is a typedef
// to an **anonymous** struct. Instead, we declare a new struct which inherits
// from `mz_zip_archive`.
struct mta_zip_archive;

namespace metatomic_torch {

// Explicit constant to request STORED (no compression) in miniz
inline constexpr unsigned ZIP_STORED = 0;

/**
 * RAII wrapper around miniz for writing ZIP archives.
 * Can target a file or an in-memory buffer.
 */
class ZipWriter {
public:
    // Construct and open to a file
    explicit ZipWriter(const std::string& path);

    // Construct and open to an in-memory buffer
    explicit ZipWriter(size_t initial_allocation_size);

    ~ZipWriter();

    ZipWriter(const ZipWriter&) = delete;
    ZipWriter& operator=(const ZipWriter&) = delete;
    ZipWriter(ZipWriter&& other) noexcept;
    ZipWriter& operator=(ZipWriter&& other) noexcept;

    // Add a file from memory into the archive at name_in_zip
    void add_file(const std::string& name_in_zip, const void* data, size_t size);
    void add_file(const std::string& name_in_zip, const std::vector<uint8_t>& buf) {
        add_file(name_in_zip, buf.data(), buf.size());
    }

    // For file writer: finalize and close (writes central directory)
    void finalize();

    // For in-memory writer: finalize and return the whole archive as bytes
    std::vector<uint8_t> finalize_to_vector();

private:
    std::unique_ptr<mta_zip_archive> zip_;
    bool in_memory_ = false;
    bool finalized_ = false;

    void close_no_throw();
};


/**
 * RAII wrapper around miniz for reading ZIP archives.
 * Can read from a file or an in-memory buffer.
 */
class ZipReader {
public:
    // Construct and open from a file
    explicit ZipReader(const std::string& path);

    // Construct and open from an in-memory buffer
    explicit ZipReader(const void* data, size_t size);

    ~ZipReader();

    ZipReader(const ZipReader&) = delete;
    ZipReader& operator=(const ZipReader&) = delete;
    ZipReader(ZipReader&& other) noexcept;
    ZipReader& operator=(ZipReader&& other) noexcept;

    bool has(const std::string& name_in_zip) const;
    std::vector<uint8_t> read(const std::string& name_in_zip) const;
    std::vector<std::string> list() const;

private:
    std::unique_ptr<mta_zip_archive> zip_;
    const void* buffer_ = nullptr;
    size_t buffer_size_ = 0;
    bool in_memory_ = false;
};

} // namespace metatomic_torch
