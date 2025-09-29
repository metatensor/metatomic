#include "npy.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <limits>
#include <sstream>
#include <cctype>

using namespace metatomic_torch;

static bool is_ascii(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (p[i] > 0x7F) {
            return false;
        }
    }
    return true;
}

struct Header {
    std::string descr;           // '<f8', '<i8', '<i4', '|b1'
    bool fortran_order;          // must be false
    std::vector<size_t> shape;  // non-negative sizes
};

// Very small, permissive parser for the dict literal we emit.
// Assumes ASCII input. Tolerates arbitrary spaces and trailing commas like NumPy.
static Header parse_header_ascii(const std::string& s) {
    auto skip_whitespaces = [&](size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\x0C')) {
            i++;
        }
    };

    auto expect = [&](size_t& i, char c) {
        skip_whitespaces(i);
        if (i >= s.size() || s[i] != c) {
            throw std::runtime_error(std::string("npy: header parse: expected '") + c + "'");
        }
        i++;
    };

    auto parse_ident_string = [&](size_t& i) -> std::string {
        skip_whitespaces(i);
        if (i >= s.size()) {
            throw std::runtime_error("npy: header parse: unexpected end while reading string");
        }
        char q = s[i];
        if (q != '\'' && q != '"') {
            throw std::runtime_error("npy: header parse: expected quoted string");
        }
        i++;

        std::string out;
        while (i < s.size() && s[i] != q) { out.push_back(s[i]); i++; }
        if (i >= s.size()) {
            throw std::runtime_error("npy: header parse: unterminated string");
        }
        i++;

        return out;
    };

    auto parse_bool = [&](size_t& i) -> bool {
        skip_whitespaces(i);
        if (s.compare(i, 4, "True") == 0)  {
            i += 4;
            return true;
        }
        if (s.compare(i, 5, "False") == 0) {
            i += 5;
            return false;
        }
        throw std::runtime_error("npy: header parse: expected True/False");
    };

    auto parse_shape = [&](size_t& i) -> std::vector<size_t> {
        std::vector<size_t> dims;
        skip_whitespaces(i);
        expect(i, '(');
        skip_whitespaces(i);

        // Allow empty tuple () as scalar, but we'll treat as 0-d (not used here).
        while (i < s.size() && s[i] != ')') {
        // parse unsigned integer
        skip_whitespaces(i);
        if (!(i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))) {
            throw std::runtime_error("npy: header parse: expected integer in shape");
        }
        size_t val = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            auto digit = static_cast<unsigned>(s[i] - '0');
            if (val > (std::numeric_limits<size_t>::max() - digit) / 10) {
                throw std::runtime_error("npy: header parse: shape integer overflow");
            }
            val = (val * 10) + digit;
            i++;
        }
        dims.push_back(val);

        skip_whitespaces(i);
        if (i < s.size() && s[i] == ',') { i++; skip_whitespaces(i); }
        }
        expect(i, ')');
        return dims;
    };

    size_t i = 0;
    skip_whitespaces(i);
    expect(i, '{');

    bool have_descr = false;
    bool have_fortran = false;
    bool have_shape = false;
    Header h{};

    while (true) {
        skip_whitespaces(i);
        if (i < s.size() && s[i] == '}') { i++; break; }

        std::string key = parse_ident_string(i);
        skip_whitespaces(i);
        expect(i, ':');

        if (key == "descr") {
            // scalar string like '<f8' or '|b1'
            h.descr = parse_ident_string(i);
            have_descr = true;
        } else if (key == "fortran_order") {
            h.fortran_order = parse_bool(i);
            have_fortran = true;
        } else if (key == "shape") {
            h.shape = parse_shape(i);
            have_shape = true;
        } else {
            throw std::runtime_error("npy: header parse: unknown key: " + key);
        }

        skip_whitespaces(i);
        if (i < s.size() && s[i] == ',') { i++; skip_whitespaces(i); }
    }

    if (!have_descr) {
        throw std::runtime_error("npy: header parse: missing 'descr'");
    }
    if (!have_fortran) {
        throw std::runtime_error("npy: header parse: missing 'fortran_order'");
    }
    if (!have_shape) {
        throw std::runtime_error("npy: header parse: missing 'shape'");
    }
    return h;
}

static std::string make_header_dict_literal(const Header& h) {
  // Follow the same spacing/trailing comma style as NumPy and your Rust reference.
    // Example: "{ 'descr': '<f8', 'fortran_order': False, 'shape': (3, 2, ) }"
    std::ostringstream oss;
    oss << "{ 'descr': '" << h.descr << "', "
        << "'fortran_order': " << (h.fortran_order ? "True" : "False") << ", "
        << "'shape': (";
    for (size_t k = 0; k < h.shape.size(); ++k) {
        oss << h.shape[k] << ", ";
    }
    oss << ") }";
    return oss.str();
}

static std::vector<uint8_t> build_header(const std::string& dict_ascii) {
    // Try v1.0 then v2.0 depending on header length constraints, pad to multiple of 64 bytes.
    static const uint8_t MAGIC[] = { 0x93, 'N','U','M','P','Y' };
    auto mk = [&](uint8_t major, uint8_t minor, size_t header_len_field_bytes) -> std::vector<uint8_t> {
        // header so far: magic + version + header_len_field
        const size_t prefix_len = sizeof(MAGIC) + 2 + header_len_field_bytes;
        // total = prefix + dict + padding + '\n'; total % 64 == 0
        size_t unpadded_total = prefix_len + dict_ascii.size() + 1; // + '\n'
        size_t padding = (64 - (unpadded_total % 64)) % 64;
        size_t total = unpadded_total + padding;
        size_t header_len = total - prefix_len;

        // format header_len little-endian (u16 for v1.0, u32 for v2.0)
        std::vector<uint8_t> out;
        out.reserve(total);
        out.insert(out.end(), std::begin(MAGIC), std::end(MAGIC));
        out.push_back(major);
        out.push_back(minor);

        if (header_len_field_bytes == 2) {
            if (header_len > std::numeric_limits<uint16_t>::max()) {
                return {};
            }
            auto hl = static_cast<uint16_t>(header_len);
            out.push_back(static_cast<uint8_t>(hl & 0xFF));
            out.push_back(static_cast<uint8_t>((hl >> 8) & 0xFF));
        } else {
            if (header_len > std::numeric_limits<uint32_t>::max()) {
                return {};
            }
            auto hl = static_cast<uint32_t>(header_len);
            out.push_back(static_cast<uint8_t>(hl & 0xFF));
            out.push_back(static_cast<uint8_t>((hl >> 8) & 0xFF));
            out.push_back(static_cast<uint8_t>((hl >> 16) & 0xFF));
            out.push_back(static_cast<uint8_t>((hl >> 24) & 0xFF));
        }

        out.insert(out.end(), dict_ascii.begin(), dict_ascii.end());
        out.insert(out.end(), padding, static_cast<uint8_t>(' '));
        out.push_back('\n');
        return out;
    };

    // Prefer v1.0 if it fits, else v2.0.
    auto v1 = mk(0x01, 0x00, 2);
    if (!v1.empty()) {
        return v1;
    }

    auto v2 = mk(0x02, 0x00, 4);
    if (!v2.empty()) {
        return v2;
    }
    throw std::runtime_error("npy: header too long for v1.0/v2.0");
}


std::vector<uint8_t> metatomic_torch::npy_write(const torch::Tensor& t_in) {
    if (!t_in.defined()) {
        throw std::runtime_error("npy: tensor is undefined");
    }

    // Require CPU & contiguous. (We mimic Python's behavior & make it explicit.)
    torch::Tensor t = t_in.contiguous().cpu();

    // Ensure we are on little endian
    static bool is_little_endian = [] {
        uint16_t x = 0x1;
        return *reinterpret_cast<uint8_t*>(&x) == 0x1;
    }();
    if (!is_little_endian) {
        throw std::runtime_error("npy_write: big-endian architectures are not supported");
    }

    // Map dtype -> descr and itemsize
    std::string descr;
    size_t itemsize = 0;
    auto dtype = t.scalar_type();
    switch (dtype) {
        case torch::kFloat64:
            descr = "<f8";
            itemsize = 8;
            break;
        case torch::kInt64:
            descr = "<i8";
            itemsize = 8;
            break;
        case torch::kInt32:
            descr = "<i4";
            itemsize = 4;
            break;
        case torch::kBool:
            descr = "|b1";
            itemsize = 1;
            break;
        default:
            throw std::runtime_error("npy_write: unsupported dtype (only f64, i64, i32, bool)");
    }

    // Build shape
    std::vector<size_t> shape;
    shape.reserve(t.dim());
    for (int d = 0; d < t.dim(); ++d) {
        size_t s = t.size(d);
        if (s < 0) {
            throw std::runtime_error("npy: negative dimension");
        }
        shape.push_back(s);
    }

    Header h{descr, /*fortran_order=*/false, shape};
    std::string dict = make_header_dict_literal(h);

    // Compose header bytes
    std::vector<uint8_t> header = build_header(dict);

    // Append raw data (C-order)
    auto count = static_cast<size_t>(t.numel());
    auto n_bytes = count * itemsize;
    std::vector<uint8_t> out;
    out.reserve(header.size() + n_bytes);
    out.insert(out.end(), header.begin(), header.end());

    if (n_bytes > 0) {
        const void* src = t.data_ptr();
        const auto* p = static_cast<const uint8_t*>(src);
        out.insert(out.end(), p, p + n_bytes);
    }

    return out;
}

torch::Tensor metatomic_torch::npy_read(const uint8_t* data, size_t size) {
    if (!(data != nullptr || size == 0)) {
        throw std::runtime_error("npy: null data with non-zero size");
    }
    if (size < 10) {
        throw std::runtime_error("npy: buffer too small");
    }

    // Magic
    static const uint8_t MAGIC[] = { 0x93, 'N','U','M','P','Y' };
    if (size < sizeof(MAGIC)) {
        throw std::runtime_error("npy: buffer too small for magic");
    }
    if (std::memcmp(data, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::runtime_error("npy: bad magic");
    }

    size_t off = sizeof(MAGIC);

    // Version
    if (size < off + 2) {
        throw std::runtime_error("npy: truncated version");
    }

    uint8_t ver_major = data[off + 0];
    uint8_t ver_minor = data[off + 1];
    off += 2;

    if (ver_minor != 0) {
        throw std::runtime_error("npy: unsupported minor version");
    }

    if (!(ver_major == 1 || ver_major == 2 || ver_major == 3)) {
        throw std::runtime_error("npy: unsupported major version");
    }

    // Header length
    size_t header_len = 0;
    if (ver_major == 1) {
        if (size < off + 2) {
            throw std::runtime_error("npy: truncated header length");
        }
        auto hl = static_cast<uint16_t>(data[off] | (data[off + 1] << 8));
        header_len = hl;
        off += 2;
    } else {
        if (size < off + 4) {
            throw std::runtime_error("npy: truncated header length");
        }
        uint32_t hl = (uint32_t)data[off] |
                    ((uint32_t)data[off + 1] << 8) |
                    ((uint32_t)data[off + 2] << 16) |
                    ((uint32_t)data[off + 3] << 24);
        header_len = hl;
        off += 4;
    }

    if (size < off + header_len) {
        throw std::runtime_error("npy: truncated header");
    }
    const uint8_t* hbeg = data + off;
    const uint8_t* hend = hbeg + header_len;
    off += header_len;

    if (!(header_len >= 1 && hend[-1] == '\n')) {
        throw std::runtime_error("npy: missing newline at end of header");
    }

    // For v1.0/v2.0 the header must be ASCII
    if (ver_major == 1 || ver_major == 2) {
        if (!is_ascii(hbeg, header_len - 1)) {
        throw std::runtime_error("npy: non-ASCII header in v1/v2");
        }
    }

    // Parse dict (skip the trailing '\n')
    std::string header_str(reinterpret_cast<const char*>(hbeg), header_len - 1);
    Header h = parse_header_ascii(header_str);

    // Only C-order supported
    if (h.fortran_order) {
        throw std::runtime_error("npy: Fortran-order arrays are not supported");
    }

    // Map descr -> dtype and itemsize
    torch::ScalarType st;
    size_t itemsize = 0;
    if (h.descr == "<f8") { st = torch::kFloat64; itemsize = 8; }
    else if (h.descr == "<i8") { st = torch::kInt64; itemsize = 8; }
    else if (h.descr == "<i4") { st = torch::kInt32; itemsize = 4; }
    else if (h.descr == "|b1") { st = torch::kBool; itemsize = 1; }
    else {
        throw std::runtime_error("npy_read: unsupported descr: " + h.descr);
    }

    // Compute data size and validate
    size_t count = 1;
    for (auto d : h.shape) {
        if (d == 0) { count = 0; break; }
        if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
        throw std::runtime_error("npy_read: shape too large");
        }
        count *= static_cast<size_t>(d);
    }
    size_t needed = count * itemsize;
    if (size < off + needed) {
        throw std::runtime_error("npy: truncated data payload");
    }

    // Create tensor
    std::vector<int64_t> sizes(h.shape.begin(), h.shape.end());
    if (sizes.empty()) {
        // 0-D not expected in our use; treat as size {1}
        sizes.push_back(1);
    }

    torch::Tensor t = torch::empty(sizes, torch::TensorOptions().dtype(st).device(torch::kCPU));
    if (needed > 0) {
        std::memcpy(t.data_ptr(), data + off, needed);
    }
    return t;
}
