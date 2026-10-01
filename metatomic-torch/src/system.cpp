#include <cctype>
#include <cstring>

#include <sstream>
#include <utility>

#include <torch/torch.h>
#include <torch/version.h>
#if defined(__APPLE__)
#include <ATen/detail/MPSHooksInterface.h>
#endif
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>
// this must be included after metatensor, to make sure we use metatensor's
// DLPack header
#include <ATen/DLConvertor.h>

#include "metatomic/torch/system.hpp"
#include "metatomic/torch/units.hpp"

#include "./internal/utils.hpp"


using namespace metatomic_torch;

PairListOptionsHolder::PairListOptionsHolder(
    double cutoff,
    bool full_list,
    bool strict,
    std::string requestor
): options_(metatomic::PairListOptions::Builder().cutoff(cutoff).full_list(full_list).strict(strict).build())
{
    this->add_requestor(std::move(requestor));
}


void PairListOptionsHolder::add_requestor(std::string requestor) {
    options_.add_requestor(std::move(requestor));
}

void PairListOptionsHolder::set_length_unit(std::string length_unit) {
    details::validate_unit("length", length_unit);
    this->length_unit_ = std::move(length_unit);
}

double PairListOptionsHolder::engine_cutoff(const std::string& engine_length_unit) const {
    return this->cutoff() * unit_conversion_factor(length_unit_, engine_length_unit);
}

std::string PairListOptionsHolder::repr() const {
    auto ss = std::ostringstream();

    ss << "PairListOptions\n";
    ss << "    cutoff: " << std::to_string(this->cutoff());
    if (!length_unit_.empty()) {
        ss << " " << length_unit_;
    }
    ss << "\n    full_list: " << (this->full_list() ? "True" : "False") << "\n";
    ss << "    strict: " << (this->strict() ? "True" : "False") << "\n";

    auto requestors = this->requestors();
    if (!requestors.empty()) {
        ss << "    requested by:\n";
        for (const auto& requestor: requestors) {
            ss << "        - " << requestor << "\n";
        }
    }

    return ss.str();
}

std::string PairListOptionsHolder::str() const {
    return "PairListOptions(cutoff=" + std::to_string(this->cutoff()) + \
        ", full_list=" + (this->full_list() ? "True" : "False") + \
        ", strict=" + (this->strict() ? "True" : "False") + ")";
}

std::string PairListOptionsHolder::to_json() const {
    nlohmann::json result = this->options_;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

PairListOptions PairListOptionsHolder::from_json(const std::string& json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for PairListOptions, expected an object");
    }


    if (data.contains("class")) {
        // Legacy JSON format, used in metatomic-torch<0.11
        if (!data["class"].is_string() || data["class"] != "NeighborListOptions") {
            throw std::runtime_error("'class' in legacy JSON for PairListOptions must be 'NeighborListOptions'");
        }

        data.erase("class");
        data["type"] = "metatomic_pair_list_options";

        if (data.contains("cutoff")) {
            auto cutoff = data["cutoff"].get<int64_t>();
            std::ostringstream oss;
            oss << "0x" << std::hex << cutoff;
            data["cutoff"] = oss.str();
        }
    }

    auto options = data.get<metatomic::PairListOptions>();

    return torch::make_intrusive<PairListOptionsHolder>(std::move(options));
}

// ========================================================================== //

torch::Tensor NeighborsAutograd::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor positions,
    torch::Tensor cell,
    metatensor_torch::TensorBlock neighbors,
    bool check_consistency
) {
    auto distances = neighbors->values();

    if (check_consistency) {
        auto n_atoms = positions.size(0);
        auto epsilon = 1e-6;
        if (distances.scalar_type() != torch::kFloat64) {
            epsilon = 1e-4;
        }

        auto samples = neighbors->samples()->values();
        for (int64_t sample_i=0; sample_i<samples.size(0); sample_i++) {
            auto atom_i = samples[sample_i][0];
            auto atom_j = samples[sample_i][1];

            auto atom_i_cpu = atom_i.to(torch::kCPU).item<int32_t>();
            auto atom_j_cpu = atom_j.to(torch::kCPU).item<int32_t>();
            if (atom_i_cpu < 0 || atom_i_cpu >= n_atoms) {
                C10_THROW_ERROR(ValueError,
                    "checking internal consistency: 'first_atom' in neighbor list (" +
                    std::to_string(atom_i_cpu) + ") is out of bounds (we have " +
                    std::to_string(n_atoms) + " atoms in the system)"
                );
            }

            if (atom_j_cpu < 0 || atom_j_cpu >= n_atoms) {
                C10_THROW_ERROR(ValueError,
                    "checking internal consistency: 'second_atom' in neighbor list (" +
                    std::to_string(atom_j_cpu) + ") is out of bounds (we have " +
                    std::to_string(n_atoms) + " atoms in the system)"
                );
            }

            auto cell_shift = samples.index({sample_i, torch::indexing::Slice(2, 5)}).to(positions.scalar_type());

            auto actual_distance = distances[sample_i].reshape({3});
            auto expected_distance = positions[atom_j] - positions[atom_i] + cell_shift.matmul(cell);

            auto diff_norm = (actual_distance - expected_distance).norm().to(torch::kCPU).to(torch::kF64);
            if (diff_norm.item<double>() > epsilon) {
                std::ostringstream oss;

                oss << "checking internal consistency: one neighbor pair does not match its metadata: ";
                oss << "the pair between atom " << atom_i.item<int32_t>();
                oss << " and atom " << atom_j.item<int32_t>() << " for the ";

                auto cell_shift_i32 = samples.index({sample_i, torch::indexing::Slice(2, 5)});
                oss << "[" << cell_shift_i32[0].item<int32_t>() << ", ";
                oss << cell_shift_i32[1].item<int32_t>() << ", ";
                oss << cell_shift_i32[2].item<int32_t>() << "] cell shift ";

                auto expected_f64 = expected_distance.to(torch::kCPU).to(torch::kF64);
                oss << "should have a distance vector of ";
                oss << "[" << expected_f64[0].item<double>() << ", ";
                oss << expected_f64[1].item<double>() << ", ";
                oss << expected_f64[2].item<double>() << "] ";


                auto actual_f64 = actual_distance.to(torch::kCPU).to(torch::kF64);
                oss << "but has a distance vector of ";
                oss << "[" << actual_f64[0].item<double>() << ", ";
                oss << actual_f64[1].item<double>() << ", ";
                oss << actual_f64[2].item<double>() << "] ";

                oss << "norm difference is " << diff_norm.item<double>();

                C10_THROW_ERROR(ValueError, oss.str());
            }
        }
    }

    ctx->save_for_backward({positions, cell, neighbors->values(), neighbors->samples()->values()});

    return distances;
}


std::vector<torch::Tensor> NeighborsAutograd::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> outputs_grad
) {
    const auto& distances_grad = outputs_grad[0];

    const auto& saved_variables = ctx->get_saved_variables();
    const auto& positions = saved_variables[0];
    const auto& cell = saved_variables[1];

    const auto& distances = saved_variables[2];
    const auto& samples = saved_variables[3];

    auto positions_grad = torch::Tensor();
    if (positions.requires_grad()) {
        positions_grad = torch::zeros_like(positions);
        positions_grad = torch::index_add(
            positions_grad,
            /*dim=*/0,
            /*index=*/samples.index({torch::indexing::Slice(), 1}),
            /*source=*/distances_grad.squeeze(-1),
            /*alpha=*/1.0
        );
        positions_grad = torch::index_add(
            positions_grad,
            /*dim=*/0,
            /*index=*/samples.index({torch::indexing::Slice(), 0}),
            /*source=*/distances_grad.squeeze(-1),
            /*alpha=*/-1.0
        );
    }

    auto cell_grad = torch::Tensor();
    if (cell.requires_grad()) {
        auto cell_shifts = samples.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(2, 5)
        }).to(cell.scalar_type());
        cell_grad = cell_shifts.t().matmul(distances_grad.squeeze(-1));
    }

    return {positions_grad, cell_grad, torch::Tensor(), torch::Tensor()};
}

void metatomic_torch::register_autograd_neighbors(
    System system,
    metatensor_torch::TensorBlock neighbors,
    bool check_consistency
) {
    auto distances = neighbors->values();
    if (distances.requires_grad()) {
        C10_THROW_ERROR(ValueError,
            "`neighbors` is already part of a computational graph, "
            "detach it before calling `register_autograd_neighbors()`"
        );
    }

    // these checks should be fine in a normal use case, but might be false if
    // someone gives weird data to the function. `check_consistency=True` should
    // help debug this kind of issues.
    if (check_consistency) {
        if (system->positions().device() != distances.device()) {
            C10_THROW_ERROR(ValueError,
                "`system` and `neighbors` must be on the same device, "
                "got " + system->positions().device().str() + " and " +
                distances.device().str()
            );
        }

        if (system->positions().scalar_type() != distances.scalar_type()) {
            C10_THROW_ERROR(ValueError,
                "`system` and `neighbors` must have the same dtype, "
                "got " + scalar_type_name(system->positions().scalar_type()) +
                " and " + scalar_type_name(distances.scalar_type())
            );
        }

        auto expected_names = std::vector<std::string>{
            "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
        };
        if (neighbors->samples()->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected sample names to be ['first_atom', "
                "'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c']"
            );
        }

        expected_names = std::vector<std::string>{"xyz"};
        if (neighbors->components().size() != 1 || neighbors->components()[0]->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected component names to be ['xyz']"
            );
        }

        expected_names = std::vector<std::string>{"distance"};
        if (neighbors->properties()->names() != expected_names) {
            C10_THROW_ERROR(ValueError,
                "invalid `neighbors`: expected property names to be ['distance']"
            );
        }
    }

    // WARNING: we are not using `torch::autograd::Function` in the usual way.
    // Instead we pass already computed data (in `neighbors->values()`), and
    // directly return the corresponding tensor in `forward()`. This still
    // causes torch to register the right function for `backward`.
    auto _ = NeighborsAutograd::apply(
        system->positions(),
        system->cell(),
        neighbors,
        check_consistency
    );
}

// ========================================================================== //

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 9
// at::toDLPackVersioned() and at::fromDLPackVersioned() are only available
// starting with torch 2.9, so we convert between the legacy `DLManagedTensor`
// used by older versions of torch and `DLManagedTensorVersioned` manually.

static void legacy_to_versioned_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        auto* legacy = static_cast<DLManagedTensor*>(self->manager_ctx);
        if (legacy != nullptr && legacy->deleter != nullptr) {
            legacy->deleter(legacy);
        }
        delete self;
    }
}

static void versioned_to_legacy_deleter(DLManagedTensor* self) {
    if (self != nullptr) {
        auto* versioned = static_cast<DLManagedTensorVersioned*>(self->manager_ctx);
        if (versioned != nullptr && versioned->deleter != nullptr) {
            versioned->deleter(versioned);
        }
        delete self;
    }
}
#endif

/// Data kept alive alongside a DLPack tensor exported from a meta tensor
struct MetaDLPackContext {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

static void meta_dlpack_deleter(DLManagedTensorVersioned* self) {
    if (self != nullptr) {
        delete static_cast<MetaDLPackContext*>(self->manager_ctx);
        delete self;
    }
}

/// torch can not export meta tensors to DLPack, so we do it manually. Like
/// metatensor-torch, we use `kDLExtDev` as the device and a NULL data pointer,
/// since there is no data associated with meta tensors.
static metatomic::DLPackTensor meta_tensor_to_dlpack(const torch::Tensor& tensor) {
    auto* context = new MetaDLPackContext{
        tensor.sizes().vec(),
        tensor.strides().vec(),
    };

    auto* versioned = new DLManagedTensorVersioned();
    versioned->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    versioned->manager_ctx = context;
    versioned->deleter = meta_dlpack_deleter;
    versioned->flags = 0;

    versioned->dl_tensor.data = nullptr;
    versioned->dl_tensor.device = DLDevice{kDLExtDev, 0};
    versioned->dl_tensor.ndim = static_cast<int32_t>(tensor.dim());
    versioned->dl_tensor.dtype = at::getDLDataType(tensor);
    versioned->dl_tensor.shape = context->shape.data();
    versioned->dl_tensor.strides = context->strides.data();
    versioned->dl_tensor.byte_offset = 0;

    return metatomic::DLPackTensor(versioned);
}

/// Inverse of `meta_tensor_to_dlpack`
static torch::Tensor meta_tensor_from_dlpack(const metatomic::DLPackTensor& tensor) {
    const auto& dl_tensor = tensor->dl_tensor;
    auto shape = torch::IntArrayRef(dl_tensor.shape, dl_tensor.ndim);
    auto options = torch::TensorOptions()
        .dtype(at::toScalarType(dl_tensor.dtype))
        .device(torch::kMeta);

    if (dl_tensor.strides == nullptr) {
        return torch::empty(shape, options);
    } else {
        auto strides = torch::IntArrayRef(dl_tensor.strides, dl_tensor.ndim);
        return torch::empty_strided(shape, strides, options);
    }
}

static metatomic::DLPackTensor tensor_to_dlpack(const torch::Tensor& tensor) {
    if (tensor.is_meta()) {
        return meta_tensor_to_dlpack(tensor);
    }

    // gradient tracking information is kept in the torch tensors stored in
    // `SystemHolder`, the DLPack tensor only needs to share the data.
    auto detached = tensor.detach();

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9
    return metatomic::DLPackTensor(at::toDLPackVersioned(detached));
#else
    DLManagedTensor* legacy = at::toDLPack(detached);

    auto* versioned = new DLManagedTensorVersioned();
    versioned->version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    versioned->manager_ctx = legacy;
    versioned->deleter = legacy_to_versioned_deleter;
    versioned->flags = 0;
    versioned->dl_tensor = legacy->dl_tensor;

    return metatomic::DLPackTensor(versioned);
#endif
}

static torch::Tensor dlpack_to_tensor(metatomic::DLPackTensor tensor) {
    if (!tensor) {
        throw std::runtime_error("internal error: got a null DLPack tensor");
    }

    if (tensor->dl_tensor.device.device_type == kDLExtDev) {
        return meta_tensor_from_dlpack(tensor);
    }

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 10
    // torch<2.10 ignores `byte_offset` when importing DLPack tensors, so we
    // fold it into the data pointer instead.
    auto& dl_tensor = tensor->dl_tensor;
    if (dl_tensor.byte_offset != 0) {
        dl_tensor.data = static_cast<char*>(dl_tensor.data) + dl_tensor.byte_offset;
        dl_tensor.byte_offset = 0;
    }
#endif

#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9
    // on success, torch takes ownership of the DLPack tensor
    auto* versioned = tensor.as_dlpack();
    auto result = at::fromDLPackVersioned(versioned);
    tensor.release();
    return result;
#else
    if (tensor->version.major != DLPACK_MAJOR_VERSION) {
        throw std::runtime_error(
            "unsupported DLPack version: got v" +
            std::to_string(tensor->version.major) + "." +
            std::to_string(tensor->version.minor) + ", but we support v" +
            std::to_string(DLPACK_MAJOR_VERSION) + "." +
            std::to_string(DLPACK_MINOR_VERSION)
        );
    }

    auto* legacy = new DLManagedTensor();
    legacy->dl_tensor = tensor->dl_tensor;
    legacy->manager_ctx = tensor.release();
    legacy->deleter = versioned_to_legacy_deleter;

    // on success, torch takes ownership of the DLPack tensor
    try {
        return at::fromDLPack(legacy);
    } catch (...) {
        versioned_to_legacy_deleter(legacy);
        throw;
    }
#endif
}

static torch::Tensor cast_types_to_int32(torch::Tensor types) {
    if (torch::canCast(types.scalar_type(), torch::kInt32)) {
        return types.to(torch::kInt32);
    } else {
        C10_THROW_ERROR(ValueError,
            "`types` must be a tensor of integers, got " +
            scalar_type_name(types.scalar_type()) + " instead"
        );
    }
}

/// Synchronize torch's Metal command queues if the system data is on a Metal
/// device. This is the C++ equivalent of `torch.mps.synchronize()`.
///
/// metatomic validates the system tensors with compute kernels running on
/// their own Metal command queue, which reads the data of the torch tensors
/// the system is created from. Since torch executes operations on Metal
/// devices asynchronously on its own queues, the tensors' data might not be
/// written yet when the validation runs, leading to reading stale data.
/// Synchronizing torch's queues makes sure all pending operations are
/// completed first.
static void synchronize_mps_if_needed(
    const torch::Tensor& types,
    const torch::Tensor& positions,
    const torch::Tensor& cell,
    const torch::Tensor& pbc
) {
#if defined(__APPLE__)
    bool any_tensor_is_mps = false;
    any_tensor_is_mps |= types.device().type() == torch::kMPS;
    any_tensor_is_mps |= positions.device().type() == torch::kMPS;
    any_tensor_is_mps |= cell.device().type() == torch::kMPS;
    any_tensor_is_mps |= pbc.device().type() == torch::kMPS;

    if (any_tensor_is_mps) {
        // what torch.mps.synchronize() calls internally
        at::detail::getMPSHooks().deviceSynchronize();
    }
#else
    (void)types;
    (void)positions;
    (void)cell;
    (void)pbc;
#endif
}

static metatomic::System create_system(
    const std::string& length_unit,
    torch::Tensor& types,
    torch::Tensor& positions,
    torch::Tensor& cell,
    torch::Tensor& pbc
) {
    synchronize_mps_if_needed(types, positions, cell, pbc);

    return metatomic::System(
        length_unit,
        tensor_to_dlpack(cast_types_to_int32(types)),
        tensor_to_dlpack(positions),
        tensor_to_dlpack(cell),
        tensor_to_dlpack(pbc)
    );
}

SystemHolder::SystemHolder(
    std::string length_unit,
    torch::Tensor types,
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc
):
    system_(create_system(length_unit, types, positions, cell, pbc)),
    types_(std::move(types)),
    positions_(std::move(positions)),
    cell_(std::move(cell)),
    pbc_(std::move(pbc))
{}

SystemHolder::SystemHolder(
    metatomic::System system,
    torch::IValue parent
):
    system_(std::move(system)),
    parent_(std::move(parent)),
    types_(dlpack_to_tensor(system_.types())),
    positions_(dlpack_to_tensor(system_.positions())),
    cell_(dlpack_to_tensor(system_.cell())),
    pbc_(dlpack_to_tensor(system_.pbc()))
{}


metatomic::System SystemHolder::release() {
    if (!parent_.isNone()) {
        throw std::runtime_error(
            "can not release this System, it is a view inside another object"
        );
    }

    return std::move(system_);
}

void SystemHolder::swap(SystemHolder& other) noexcept {
    std::swap(this->system_, other.system_);
    std::swap(this->parent_, other.parent_);
    std::swap(this->types_, other.types_);
    std::swap(this->positions_, other.positions_);
    std::swap(this->cell_, other.cell_);
    std::swap(this->pbc_, other.pbc_);
}

System SystemHolder::from_metatomic(metatomic::System system) {
    // The tensors in a `SystemHolder` wrapping `system` directly would be
    // views of data owned by `system`, and metatomic-core does not allow
    // modifying a system (e.g. adding pairs or custom data) while such views
    // exist. Instead, we create a new system sharing the same data.
    auto view = SystemHolder(std::move(system), torch::IValue());
    return view.to();
}

System SystemHolder::view_from_metatomic(metatomic::System system, torch::IValue parent) {
    return torch::make_intrusive<SystemHolder>(std::move(system), std::move(parent));
}

System SystemHolder::to(
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    bool non_blocking
) const {
    auto system = torch::make_intrusive<SystemHolder>(
        this->system_.length_unit(),
        this->types().to(
            /*dtype*/ torch::nullopt,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            non_blocking,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->positions().to(
            dtype,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            non_blocking,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->cell().to(
            dtype,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            non_blocking,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        ),
        this->pbc().to(
            /*dtype*/ torch::nullopt,
            /*layout*/ torch::nullopt,
            device,
            /*pin_memory*/ torch::nullopt,
            non_blocking,
            /*copy*/ false,
            /*memory_format*/ torch::MemoryFormat::Preserve
        )
    );

    for (const auto& options: this->known_pairs()) {
        auto pairs = this->system_.pairs(options->as_metatomic());
        auto pairs_torch = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            std::move(pairs), torch::IValue()
        );
        system->add_pairs(options, pairs_torch->to(dtype, device, non_blocking));
    }

    for (const auto& name: this->known_custom_data()) {
        auto data = this->system_.custom_data(name);
        auto data_torch = torch::make_intrusive<metatensor_torch::TensorMapHolder>(
            std::move(data), torch::IValue()
        );
        system->add_custom_data(name, data_torch->to(dtype, device, non_blocking));
    }

    return system;
}

/// Parse the arguments to the `to` function
static std::tuple<torch::optional<torch::Dtype>, torch::optional<torch::Device>>
to_arguments_parse(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    std::string context
) {
    // handle first positional argument
    if (positional_1.isNone()) {
        // all good, nothing to do
    } else if (positional_1.isDevice()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = positional_1.toDevice();
        }
    } else if (positional_1.isString()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = torch::Device(positional_1.toString()->string());
        }
    } else if (positional_1.isInt()) {
        if (dtype.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a dtype twice in " + context);
        } else {
            dtype = static_cast<torch::Dtype>(positional_1.toInt());
        }
    } else {
        C10_THROW_ERROR(TypeError, "unexpected type in " + context + ": "+ positional_1.type()->str());
    }

    // handle second positional argument
    if (positional_2.isNone()) {
        // all good, nothing to do
    } else if (positional_2.isDevice()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = positional_2.toDevice();
        }
    } else if (positional_2.isString()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = torch::Device(positional_2.toString()->string());
        }
    } else if (positional_2.isInt()) {
        if (dtype.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a dtype twice in " + context);
        } else {
            dtype = static_cast<torch::Dtype>(positional_2.toInt());
        }
    } else {
        C10_THROW_ERROR(TypeError, "unexpected type in " + context + ": "+ positional_2.type()->str());
    }

    return std::make_tuple(dtype, device);
}

System SystemHolder::to_positional(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    bool non_blocking
) const {
    auto [parsed_dtype, parsed_device] = to_arguments_parse(
        positional_1,
        positional_2,
        dtype,
        device,
        "`System.to`"
    );

    return this->to(parsed_dtype, parsed_device, non_blocking);
}

static bool check_1d_labels(metatensor_torch::Labels labels, const char* name, torch::Tensor values) {
    assert(values.sizes().size() == 1);

    if (labels->count() != values.size(0)) {
        return false;
    }

    if (labels->size() != 1) {
        return false;
    }

    if (labels->names()[0] != name) {
        return false;
    }

    auto labels_values = labels->values().reshape(-1);
    if (!labels_values.is_meta()) {
        return torch::all(labels_values == values.to(labels_values.device())).item<bool>();
    }

    return true;
}

void SystemHolder::add_pairs(PairListOptions options, metatensor_torch::TensorBlock pairs) {
    // make a shallow copy (sharing the same tensors) to release, so `pairs`
    // stays valid for the caller and views (e.g. from another system) work
    pairs = pairs->copy(/*deep=*/false);
    this->system_.add_pairs(options->as_metatomic(), pairs->release());
}

metatensor_torch::TensorBlock SystemHolder::pairs(System self, PairListOptions options) {
    auto pairs = self->system_.pairs(options->as_metatomic());
    return torch::make_intrusive<metatensor_torch::TensorBlockHolder>(std::move(pairs), self);
}

std::vector<PairListOptions> SystemHolder::known_pairs() const {
    auto results = std::vector<PairListOptions>();
    for (const auto& options: this->system_.known_pairs()) {
        results.push_back(torch::make_intrusive<PairListOptionsHolder>(options));
    }
    return results;
}

void SystemHolder::add_custom_data(
    std::string name,
    metatensor_torch::TensorMap data
) {
    // make a shallow copy of the `data` to release, so it stays valid for the
    // caller and views (e.g. from another system).
    auto copy = data->copy(/*deep=*/false);
    this->system_.add_custom_data(name, copy->release());
}

metatensor_torch::TensorMap SystemHolder::custom_data(System self, std::string name) {
    auto data = self->system_.custom_data(name);
    return torch::make_intrusive<metatensor_torch::TensorMapHolder>(std::move(data), self);
}


std::vector<std::string> SystemHolder::known_custom_data() const {
   return this->system_.known_custom_data();
}

std::string SystemHolder::str() const {
    auto result = std::ostringstream();
    result << "System with " << this->size() << " atoms, ";

    auto cell = cell_.to(torch::kCPU, torch::kF64);
    if (torch::all(cell == torch::zeros_like(cell)).item<bool>()) {
        result << "non periodic";
    } else {
        result << "periodic cell: [";
        for (int64_t i=0; i<3; i++) {
            for (int64_t j=0; j<3; j++) {
                result << cell_.index({i, j}).item<double>();
                if (j != 2 || i != 2) {
                    result << ", ";
                }
            }
        }
        result << "]";
    }

    return result.str();
}
