#ifndef METATOMIC_TORCH_LEGACY_REGISTER_HPP
#define METATOMIC_TORCH_LEGACY_REGISTER_HPP

#include <torch/script.h>

namespace metatomic_torch::details {
    void register_legacy(torch::Library& m);
}

#endif
