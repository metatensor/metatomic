#include <vector>

#include <torch/torch.h>
#include <torch/version.h>

#include <metatensor/torch.hpp>
// this must be included after metatensor, to make sure we use metatensor's
// DLPack header
#include <ATen/DLConvertor.h>

#include "./dlpack.hpp"

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

metatomic::DLPackTensor metatomic_torch::details::tensor_to_dlpack(const torch::Tensor& tensor) {
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

torch::Tensor metatomic_torch::details::dlpack_to_tensor(metatomic::DLPackTensor tensor) {
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
