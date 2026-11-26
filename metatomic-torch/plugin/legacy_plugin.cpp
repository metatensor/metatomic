// A metatomic plugin loading models exported with the legacy TorchScript-based
// `metatomic.torch.AtomisticModel` API (typically `.pt` files), and running
// them through `metatomic_torch::LegacyModel`.
//
// The supported options are `extensions_directory`, the directory containing
// the TorchScript extensions the model requires, and `device`, the device on
// which to run the model.

#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <optional>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/mps.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include <metatomic.hpp>

#include "metatomic/torch/system.hpp"
#include "metatomic/torch/metadata.hpp"
#include "metatomic/torch/legacy/model.hpp"
#include "metatomic/torch/legacy/output.hpp"
#include "metatomic/torch/legacy/capabilities.hpp"


using namespace metatomic_torch;
using metatomic::Gradients;

// #include <caffe2/serialize/inline_container.h>

////////////////////////////////////////////////////////////////////////////////
//////                                                                    //////
//////                             HELPERS                                //////
//////                                                                    //////
////////////////////////////////////////////////////////////////////////////////

// Conversions from metatensor (core) to metatensor-torch. `to()` gives blocks
// storing torch tensors, sharing the data of the original arrays when they are
// already on `device`. Autograd needs these for the pairs: metatensor-torch
// reads other arrays through a new tensor every time.
//
// The original blocks can be views of the pairs and custom data owned by the
// engine's systems, with no torch parent to keep alive. The engine keeps these
// systems alive during the whole `execute_model` call, and the systems given to
// the legacy model do not outlive `execute_inner`.

/// Convert `labels` to metatensor-torch, on `device`
static metatensor_torch::Labels labels_to_torch(metatensor::Labels labels, torch::Device device) {
    auto result = torch::make_intrusive<metatensor_torch::LabelsHolder>(std::move(labels));
    return result->device() == device ? result : result->to(device);
}

/// Convert `block` to metatensor-torch, with values and labels on `device`
static metatensor_torch::TensorBlock block_to_torch(metatensor::TensorBlock block, torch::Device device) {
    auto result = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(std::move(block), torch::IValue());
    return result->to(torch::nullopt, device);
}

/// Convert `tensor` to metatensor-torch, with values and labels on `device`
static metatensor_torch::TensorMap tensor_to_torch(metatensor::TensorMap tensor, torch::Device device) {
    auto result = torch::make_intrusive<metatensor_torch::TensorMapHolder>(std::move(tensor));
    return result->to(torch::nullopt, device);
}

/// Convert all the metatensor-torch `labels` to metatensor-core
static std::vector<metatensor::Labels> labels_to_core(const std::vector<metatensor_torch::Labels>& labels) {
    auto result = std::vector<metatensor::Labels>();
    for (const auto& entry: labels) {
        result.push_back(entry->as_metatensor());
    }
    return result;
}

/// Create a block with `values`, `samples` and the labels of `block`,
/// `new_components` being added before the components of `block`.
static metatensor::TensorBlock block_with_values(
    torch::Tensor values,
    const metatensor::Labels& samples,
    std::vector<metatensor::Labels> new_components,
    const metatensor_torch::TensorBlock& block
) {
    for (auto& component: labels_to_core(block->components())) {
        new_components.push_back(std::move(component));
    }

    return metatensor::TensorBlock(
        std::make_unique<metatensor_torch::TorchDataArray>(std::move(values)),
        samples,
        new_components,
        block->properties()->as_metatensor()
    );
}

/// Convert `block` and its gradients to metatensor-core, detaching all values
/// from the computational graph.
static metatensor::TensorBlock block_to_core(const metatensor_torch::TensorBlock& block) {
    auto result = block_with_values(block->values().detach(), block->samples()->as_metatensor(), {}, block);
    for (const auto& [parameter, gradient]: metatensor_torch::TensorBlockHolder::gradients(block)) {
        result.add_gradient(parameter, block_to_core(gradient));
    }
    return result;
}

/// Can the gradients of `quantity` be computed with autograd? This is the case
/// for per-system energies and custom quantities, as in Python's TorchModel.
static bool supports_autograd(const metatomic::Quantity& quantity) {
    if (quantity.sample_kind() != metatomic::SampleKind::System) {
        return false;
    }

    const auto& name = quantity.name();
    if (name.find("::") != std::string::npos) {
        return true;
    }

    auto base_name = name.substr(0, name.find('/'));
    return base_name == "energy" || base_name == "energy_ensemble" || base_name == "energy_uncertainty";
}

/// Is `gradient` one of the `gradients`?
static bool contains(const std::vector<Gradients>& gradients, Gradients gradient) {
    return std::find(gradients.begin(), gradients.end(), gradient) != gradients.end();
}

/// Copy of `quantity`, with its gradients replaced by `gradients`
static metatomic::Quantity with_gradients(const metatomic::Quantity& quantity, std::vector<Gradients> gradients) {
    return metatomic::Quantity::builder()
        .name(quantity.name())
        .unit(quantity.unit())
        .sample_kind(quantity.sample_kind())
        .description(quantity.description())
        .gradients(std::move(gradients))
        .build();
}

/// Find the output in `capabilities` with the same name and sample kind as
/// `request`, if any
static std::optional<metatomic::Quantity> find_output(
    const metatomic::ModelCapabilities& capabilities,
    const metatomic::Quantity& request
) {
    for (const auto& output: capabilities.outputs()) {
        if (output.name() == request.name() && output.sample_kind() == request.sample_kind()) {
            return output;
        }
    }
    return std::nullopt;
}

/// Convert the capabilities of the legacy `model` to metatomic-core, keeping
/// only the outputs declared by the model and their explicit gradients
static metatomic::ModelCapabilities legacy_capabilities(metatensor_torch::Module& model) {
    auto capabilities = model.run_method("capabilities").toCustomClass<LegacyModelCapabilitiesHolder>();
    auto all_outputs = capabilities->outputs();

    // AtomisticModel also lists outputs under their deprecated or current
    // names, only keep the names declared by the model
    auto declared = model.attr("_model_capabilities_outputs_names");
    auto outputs = std::vector<metatomic::Quantity>();
    for (const auto& item: declared.toListRef()) {
        const auto& name = item.toStringRef();
        auto output = all_outputs.at(name);
        auto gradients = std::vector<Gradients>();
        for (const auto& gradient: output->explicit_gradients) {
            gradients.push_back(nlohmann::json(gradient).get<Gradients>());
        }

        auto sample_kinds = std::vector<std::string>{output->sample_kind()};
        if (output->sample_kind() == "atom") {
            // the legacy API allowed requesting per-atom outputs per-system
            sample_kinds.emplace_back("system");
        }

        for (const auto& sample_kind: sample_kinds) {
            outputs.push_back(metatomic::Quantity::builder()
                .name(name)
                .unit(output->unit())
                .sample_kind(nlohmann::json(sample_kind).get<metatomic::SampleKind>())
                .description(output->description)
                .gradients(gradients)
                .build()
            );
        }
    }

    auto devices = std::vector<metatomic::ModelCapabilities::Device>();
    for (const auto& device: capabilities->supported_devices) {
        // legacy device names that are spelled differently in metatomic
        auto renamed = device == "mps" ? std::string("metal") : device;
        devices.push_back(nlohmann::json(renamed).get<metatomic::ModelCapabilities::Device>());
    }

    auto atomic_types = std::vector<int32_t>();
    for (auto atomic_type: capabilities->atomic_types) {
        atomic_types.push_back(static_cast<int32_t>(atomic_type));
    }

    return metatomic::ModelCapabilities::builder()
        .outputs(std::move(outputs))
        .atomic_types(std::move(atomic_types))
        .interaction_range(capabilities->interaction_range)
        .length_unit(capabilities->length_unit())
        .supported_devices(std::move(devices))
        .dtype(nlohmann::json(capabilities->dtype()).get<metatomic::ModelCapabilities::DType>())
        .build();
}

static metatomic::ModelCapabilities full_capabilities(metatomic::ModelCapabilities declared) {
    // add the gradients computed with autograd to the declared outputs
    auto outputs = std::vector<metatomic::Quantity>();
    for (const auto& output: declared.outputs()) {
        auto gradients = output.gradients();
        if (supports_autograd(output)) {
            for (auto gradient: {Gradients::Positions, Gradients::Strain}) {
                if (!contains(gradients, gradient)) {
                    gradients.push_back(gradient);
                }
            }
        }
        outputs.push_back(with_gradients(output, std::move(gradients)));
    }

    auto devices = declared.supported_devices();
    // the systems are moved to the requested device, so engines can keep them
    // on the CPU even if the model can not run there
    auto cpu = metatomic::ModelCapabilities::Device::CPU;
    if (std::find(devices.begin(), devices.end(), cpu) == devices.end()) {
        devices.push_back(cpu);
    }

    return metatomic::ModelCapabilities::builder()
        .outputs(std::move(outputs))
        .atomic_types(declared.atomic_types())
        .interaction_range(declared.interaction_range())
        .length_unit(declared.length_unit())
        .supported_devices(std::move(devices))
        .dtype(declared.dtype())
        .build();
}

/// Get the inputs requested by the legacy `model`, as metatomic-core quantities
static std::vector<metatomic::Quantity> legacy_requested_inputs(torch::jit::Module& model) {
    auto requested = model.run_method("requested_inputs", /*use_new_names*/ true);

    auto inputs = std::vector<metatomic::Quantity>();
    for (const auto& item: requested.toGenericDict()) {
        auto input = item.value().toCustomClass<ModelOutputHolder>();
        inputs.push_back(metatomic::Quantity::builder()
            .name(item.key().toStringRef())
            .unit(input->unit())
            .sample_kind(nlohmann::json(input->sample_kind()).get<metatomic::SampleKind>())
            .description(input->description)
            .build()
        );
    }
    return inputs;
}

/// Get the pair lists requested by the legacy `model`
static std::vector<metatomic::PairListOptions> legacy_pair_lists(torch::jit::Module& model) {
    auto result = std::vector<metatomic::PairListOptions>();
    for (const auto& ivalue: model.run_method("requested_neighbor_lists").toListRef()) {
        auto options = ivalue.toCustomClass<PairListOptionsHolder>()->as_metatomic();
        options.add_requestor("LegacyModel");
        result.push_back(options);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
//////                                                                    //////
//////                         LEGACY MODEL                               //////
//////                                                                    //////
////////////////////////////////////////////////////////////////////////////////

/// Run a model exported with `metatomic.torch.AtomisticModel` (typically a
/// `.pt` file) with `metatomic::execute_model`. This behaves like the Python
/// `metatomic.LegacyModel`.
class LegacyModel final: public metatomic::BaseModel {
public:
    /// Load the model exported at `path`, loading the TorchScript extensions
    /// it requires from `extensions_directory`.
    explicit LegacyModel(
        metatensor_torch::Module model,
        c10::optional<std::string> device
    ):
        module_(model.attr("module").toModule()),
        declared_(legacy_capabilities(model)),
        capabilities_(full_capabilities(declared_)),
        metadata_(model.run_method("metadata").toCustomClass<ModelMetadataHolder>()->as_metatomic()),
        pair_lists_(legacy_pair_lists(model)),
        inputs_(legacy_requested_inputs(model))
    {
        auto capabilities = model.run_method("capabilities").toCustomClass<LegacyModelCapabilitiesHolder>();
        if (capabilities->length_unit().empty()) {
            C10_THROW_ERROR(ValueError,
                "this model does not declare a length unit, and can not be used "
                "with LegacyModel"
            );
        }

        if (device.has_value()) {
            // check that the model and the machine support this device
            pick_device(capabilities->supported_devices, device);
            this->requested_device_ = torch::Device(device.value());
        }
    }

    ~LegacyModel() override = default;

    LegacyModel(const LegacyModel&) = delete;
    LegacyModel& operator=(const LegacyModel&) = delete;
    LegacyModel(LegacyModel&&) = default;
    LegacyModel& operator=(LegacyModel&&) = default;

    metatomic::ModelCapabilities capabilities() const override {
        return capabilities_;
    }

    metatomic::ModelMetadata metadata() const override {
        return metadata_;
    }

    std::vector<metatomic::PairListOptions> requested_pair_lists(
        const std::vector<metatomic::Quantity>& requested_outputs
    ) const override {
        (void)requested_outputs;
        return pair_lists_;
    }

    std::vector<metatomic::Quantity> requested_inputs(
        const std::vector<metatomic::Quantity>& requested_outputs
    ) const override {
        (void)requested_outputs;
        return inputs_;
    }

    std::vector<metatensor::TensorMap> execute_inner(
        const std::vector<metatomic::System>& systems,
        std::optional<metatensor::Labels> selected_atoms,
        const std::vector<metatomic::Quantity>& requested_outputs,
        std::optional<int64_t> stream
    ) override;

private:
    /// The `AtomisticModel.module` of the legacy model, which is called
    /// directly: `execute_model` already converts units and checks the
    /// inputs and outputs
    metatensor_torch::Module module_;
    /// Device on which `module_` currently is
    torch::Device device_ = torch::kCPU;
    /// Device on which the model always runs, if the user gave one
    c10::optional<torch::Device> requested_device_;

    /// Capabilities of the legacy model, with its explicit gradients only
    metatomic::ModelCapabilities declared_;
    /// Full capabilities of this model, with the gradients computed with
    /// autograd added.
    metatomic::ModelCapabilities capabilities_;
    metatomic::ModelMetadata metadata_;

    /// Pair lists requested by the legacy model, used to add the pairs given
    /// by the engine to the systems
    std::vector<metatomic::PairListOptions> pair_lists_;
    std::vector<metatomic::Quantity> inputs_;
};

/// Tensors used as inputs for autograd in a single system
struct AutogradInputs {
    torch::Tensor positions;
    torch::Tensor strain;
};

/// Count the backward calls needed to compute all the gradients with
/// autograd, to only keep the computational graph alive until the last of
/// them.
class BackwardCalls {
public:
    BackwardCalls(
        const std::vector<metatensor_torch::TensorMap>& outputs,
        const std::vector<std::vector<Gradients>>& gradients
    ) {
        // this must match the calls to `torch::autograd::grad` in
        // `autograd_gradients()`
        for (size_t i = 0; i < outputs.size(); i++) {
            if (gradients[i].empty()) {
                continue;
            }

            for (const auto& block: metatensor_torch::TensorMapHolder::blocks(outputs[i])) {
                auto values = block->values();
                if (values.requires_grad()) {
                    // one call for each entry along the components and properties
                    auto shape = values.sizes().slice(1);
                    remaining_ += std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
                }
            }
        }
    }

    /// Register a new backward call, and check if the graph should be kept
    /// alive after it.
    bool retain_graph() {
        remaining_ -= 1;
        return remaining_ > 0;
    }

private:
    int64_t remaining_ = 0;
};

/// Split `requested_outputs` between the gradients computed by the model and
/// the ones computed with autograd. This returns the requests to give to the
/// model, and the gradients to compute with autograd for each output.
static std::pair<std::vector<metatomic::Quantity>, std::vector<std::vector<Gradients>>> split_gradients(
    const metatomic::ModelCapabilities& declared,
    const std::vector<metatomic::Quantity>& requested_outputs
) {
    auto inner_outputs = std::vector<metatomic::Quantity>();
    auto autograd = std::vector<std::vector<Gradients>>();
    for (const auto& request: requested_outputs) {
        auto output = find_output(declared, request);
        auto explicit_gradients = std::vector<Gradients>();
        auto autograd_gradients = std::vector<Gradients>();
        for (auto gradient: request.gradients()) {
            if (output.has_value() && contains(output->gradients(), gradient)) {
                explicit_gradients.push_back(gradient);
            } else {
                autograd_gradients.push_back(gradient);
            }
        }

        if (!autograd_gradients.empty() && !supports_autograd(request)) {
            C10_THROW_ERROR(ValueError,
                "gradients of '" + request.name() + "' with sample kind '" +
                nlohmann::json(request.sample_kind()).get<std::string>() + "' can not be computed with autograd"
            );
        }

        inner_outputs.push_back(with_gradients(request, std::move(explicit_gradients)));
        autograd.push_back(std::move(autograd_gradients));
    }

    return {std::move(inner_outputs), std::move(autograd)};
}

/// Split the indices of `outputs` in batches where all names are unique. The
/// legacy API is keyed by name, so the same name requested with different
/// sample kinds needs separate calls.
static std::vector<std::vector<size_t>> unique_name_batches(const std::vector<metatomic::Quantity>& outputs) {
    auto counts = std::unordered_map<std::string, size_t>();
    auto batches = std::vector<std::vector<size_t>>();
    for (size_t i = 0; i < outputs.size(); i++) {
        // the n-th request for a given name goes in the n-th batch
        auto batch = counts[outputs[i].name()]++;
        if (batch == batches.size()) {
            batches.emplace_back();
        }
        batches[batch].push_back(i);
    }
    return batches;
}

/// Create a `System` sharing all its data with `system`, using torch tensors.
/// The positions and cell are set up to compute `gradients` with autograd,
/// and the corresponding inputs are stored in `inputs`.
static System torch_system(
    const metatomic::System& system,
    const std::vector<metatomic::PairListOptions>& pair_lists,
    bool positions_gradients,
    bool strain_gradients,
    AutogradInputs& inputs
) {
    // tensors sharing the data of the engine's system through DLPack
    auto view = SystemHolder::view_from_metatomic(
        metatomic::System::unsafe_view_from_ptr(system.as_mta_system_t()),
        torch::IValue()
    );

    auto positions = view->positions();
    auto cell = view->cell();
    if (positions_gradients) {
        positions = positions.detach().requires_grad_(true);
        inputs.positions = positions;
    }

    if (strain_gradients) {
        inputs.strain = torch::eye(3, positions.options().requires_grad(true));
        positions = positions.matmul(inputs.strain);
        cell = cell.matmul(inputs.strain);
    }

    // metatomic-core does not allow adding pairs to the engine's system, so the
    // model gets a new system, sharing the same data
    auto result = torch::make_intrusive<SystemHolder>(
        view->length_unit(), view->types(), positions, cell, view->pbc()
    );

    for (const auto& options: pair_lists) {
        auto pairs = block_to_torch(system.pairs(options), view->device());
        if (positions_gradients || strain_gradients) {
            register_autograd_neighbors(result, pairs, /*check_consistency*/ false);
        }
        result->add_pairs(torch::make_intrusive<PairListOptionsHolder>(options), pairs);
    }

    for (const auto& name: system.known_custom_data()) {
        result->add_custom_data(name, tensor_to_torch(system.custom_data(name), view->device()));
    }

    return result;
}

/// Compute the `gradients` of the per-system `block` with respect to the
/// positions and/or strain of `systems`, using autograd.
///
/// The gradients are computed separately for each entry along the components
/// and properties of `block`, but simultaneously for all samples, since each
/// system has a single sample and the different systems are independent.
static std::map<std::string, metatensor::TensorBlock> autograd_gradients(
    const metatensor_torch::TensorBlock& block,
    const std::vector<Gradients>& gradients,
    const std::vector<System>& systems,
    const std::vector<AutogradInputs>& inputs,
    BackwardCalls& backward_calls
) {
    auto values = block->values();
    auto systems_column = block->samples()->column("system").to(torch::kCPU, torch::kInt64).contiguous();
    auto sample_systems = std::vector<int64_t>(
        systems_column.data_ptr<int64_t>(),
        systems_column.data_ptr<int64_t>() + systems_column.numel()
    );
    auto n_samples = values.size(0);
    // shape of the components and properties, shared with the gradients
    auto values_shape = values.sizes().slice(1).vec();
    auto flat_values = values.reshape({n_samples, -1});

    auto do_positions = contains(gradients, Gradients::Positions);
    auto do_strain = contains(gradients, Gradients::Strain);

    // inputs for autograd, for all the systems
    auto positions = std::vector<torch::Tensor>();
    auto strain = std::vector<torch::Tensor>();
    for (const auto& system_inputs: inputs) {
        if (do_positions) {
            positions.push_back(system_inputs.positions);
        }
        if (do_strain) {
            strain.push_back(system_inputs.strain);
        }
    }

    // a single backward call computes the gradients w.r.t. both
    auto parameters = positions;
    parameters.insert(parameters.end(), strain.begin(), strain.end());

    // all_grads[i][column] is the gradient of `flat_values[:, column]` w.r.t. parameters[i]
    auto all_grads = std::vector<std::vector<torch::Tensor>>(parameters.size());
    for (int64_t column = 0; column < flat_values.size(1); column++) {
        auto grads = std::vector<torch::Tensor>(parameters.size());
        if (values.requires_grad()) {
            auto retain_graph = backward_calls.retain_graph();
            if (!parameters.empty()) {
                grads = torch::autograd::grad(
                    {flat_values.index({torch::indexing::Slice(), column}).sum()},
                    parameters,
                    /*grad_outputs*/ {},
                    retain_graph,
                    /*create_graph*/ false,
                    /*allow_unused*/ true
                );
            }
        }

        for (size_t i = 0; i < parameters.size(); i++) {
            all_grads[i].push_back(grads[i].defined() ? grads[i].detach() : torch::zeros_like(parameters[i]));
        }
    }

    // stack the different components & properties as the last dimensions
    auto stacked = std::vector<torch::Tensor>();
    for (size_t i = 0; i < parameters.size(); i++) {
        auto shape = parameters[i].sizes().vec();
        shape.insert(shape.end(), values_shape.begin(), values_shape.end());
        if (all_grads[i].empty()) {
            stacked.push_back(torch::zeros(shape, values.options()));
        } else {
            stacked.push_back(torch::stack(all_grads[i], -1).reshape(shape));
        }
    }

    auto n_positions = static_cast<std::ptrdiff_t>(positions.size());
    auto positions_grads = std::vector<torch::Tensor>(stacked.begin(), stacked.begin() + n_positions);
    auto strain_grads = std::vector<torch::Tensor>(stacked.begin() + n_positions, stacked.end());

    auto gradient_shape = [&](std::vector<int64_t> shape) {
        shape.insert(shape.end(), values_shape.begin(), values_shape.end());
        return shape;
    };

    // labels using the data of `values` directly, without a copy
    auto labels_from_torch = [](const std::vector<std::string>& names, torch::Tensor values) {
        return metatensor::Labels(
            names,
            metatensor::DataArrayBase::to_mts_array(
                std::make_unique<metatensor_torch::TorchDataArray>(std::move(values))
            ),
            metatensor::assume_unique{}
        );
    };

    auto results = std::map<std::string, metatensor::TensorBlock>();
    if (do_positions) {
        auto n_grad_samples = int64_t{0};
        for (auto system: sample_systems) {
            n_grad_samples += systems[static_cast<size_t>(system)]->size();
        }

        auto grad_samples = torch::empty({n_grad_samples, 3}, torch::kInt32);
        auto grad_samples_data = grad_samples.accessor<int32_t, 2>();
        auto grad_values = std::vector<torch::Tensor>();
        auto grad_sample = int64_t{0};
        for (int64_t sample = 0; sample < n_samples; sample++) {
            auto system = sample_systems[static_cast<size_t>(sample)];
            for (int64_t atom = 0; atom < systems[static_cast<size_t>(system)]->size(); atom++) {
                grad_samples_data[grad_sample][0] = static_cast<int32_t>(sample);
                grad_samples_data[grad_sample][1] = static_cast<int32_t>(system);
                grad_samples_data[grad_sample][2] = static_cast<int32_t>(atom);
                grad_sample += 1;
            }
            grad_values.push_back(positions_grads[static_cast<size_t>(system)]);
        }

        results.emplace("positions", block_with_values(
            grad_values.empty() ? torch::zeros(gradient_shape({0, 3}), values.options()) : torch::cat(grad_values),
            labels_from_torch({"sample", "system", "atom"}, std::move(grad_samples)),
            {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})},
            block
        ));
    }

    if (do_strain) {
        auto grad_values = std::vector<torch::Tensor>();
        for (auto system: sample_systems) {
            grad_values.push_back(strain_grads[static_cast<size_t>(system)]);
        }

        results.emplace("strain", block_with_values(
            grad_values.empty() ? torch::zeros(gradient_shape({0, 3, 3}), values.options()) : torch::stack(grad_values),
            labels_from_torch({"sample"}, torch::arange(n_samples, torch::kInt32).reshape({-1, 1})),
            {
                metatensor::Labels({"xyz_1"}, {{0}, {1}, {2}}),
                metatensor::Labels({"xyz_2"}, {{0}, {1}, {2}}),
            },
            block
        ));
    }

    return results;
}

/// Make the torch code running while this object is alive follow the contract
/// of `execute_inner` for `stream` (see `mta_execute_model`), for systems on
/// `device`. `finish()` must be called once all the work has been submitted.
///
/// On CUDA, the torch code runs with `stream` as the current torch stream. If
/// there is no stream, `finish()` waits for all the work to finish. Torch can
/// not use a Metal command queue it does not own, so on MPS this waits for the
/// work on `stream` to finish before running the code, and `finish()` waits for
/// the work submitted by torch to finish.
class StreamContext {
public:
    StreamContext(DLDevice device, std::optional<int64_t> stream): device_(device), stream_(stream) {
        if (device.device_type == kDLCUDA && stream.has_value()) {
            auto torch_device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device.device_id));
            if (stream.value() == 1) {
                // legacy default stream, which is also torch's default stream
                guard_.emplace(c10::Stream(c10::Stream::DEFAULT, torch_device));
            } else {
                // torch represents external CUDA streams (created outside of
                // torch) with the value of the `cudaStream_t` pointer as stream
                // id, this is what `c10::cuda::getStreamFromExternal` does.
                // Using it directly allows this code to work without linking
                // to torch's CUDA libraries.
                guard_.emplace(c10::Stream(c10::Stream::UNSAFE, torch_device, stream.value()));
            }
        } else if (device.device_type == kDLMetal) {
            metatomic::stream_synchronize(device, stream);
        }
    }

    void finish() {
        if (device_.device_type == kDLCUDA && !stream_.has_value()) {
            // only synchronize if the engine did not give a stream
            torch::cuda::synchronize(device_.device_id);
        } else if (device_.device_type == kDLMetal) {
            // always synchronize, since torch can not use a Metal command queue
            // it does not own
            torch::mps::synchronize();
        }
    }

private:
    DLDevice device_;
    std::optional<int64_t> stream_;
    std::optional<c10::StreamGuard> guard_;
};

std::vector<metatensor::TensorMap> LegacyModel::execute_inner(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const std::vector<metatomic::Quantity>& requested_outputs,
    std::optional<int64_t> stream
) {
    auto stream_context = StreamContext(
        systems.empty() ? DLDevice{kDLCPU, 0} : systems[0].positions()->dl_tensor.device,
        stream
    );

    auto [inner_outputs, autograd] = split_gradients(declared_, requested_outputs);
    auto positions_gradients = false;
    auto strain_gradients = false;
    for (const auto& gradients: autograd) {
        positions_gradients |= contains(gradients, Gradients::Positions);
        strain_gradients |= contains(gradients, Gradients::Strain);
    }

    // the engine might be running without gradients
    auto enable_grad = torch::AutoGradMode(true);

    auto torch_systems = std::vector<System>();
    auto inputs = std::vector<AutogradInputs>(systems.size());
    for (size_t i = 0; i < systems.size(); i++) {
        torch_systems.push_back(torch_system(
            systems[i], pair_lists_, positions_gradients, strain_gradients, inputs[i]
        ));
    }

    auto systems_device = torch_systems.empty() ? torch::Device(torch::kCPU) : torch_systems[0]->device();
    auto device = requested_device_.value_or(systems_device);
    if (device != systems_device) {
        // autograd can record the move to another device just fine
        for (auto& system: torch_systems) {
            system = system->to(torch::nullopt, device);
        }
    }

    if (device != device_) {
        module_.to(device);
        device_ = device;
    }

    auto torch_selected = c10::optional<metatensor_torch::Labels>();
    if (selected_atoms.has_value()) {
        torch_selected = labels_to_torch(std::move(selected_atoms.value()), device);
    }

    auto outputs = std::vector<metatensor_torch::TensorMap>(requested_outputs.size());
    for (const auto& batch: unique_name_batches(inner_outputs)) {
        // like AtomisticModel, give the requests to the module as-is: it
        // computes outputs in the declared units, and execute_model converts
        // them
        auto legacy_outputs = c10::Dict<std::string, ModelOutput>();
        for (auto i: batch) {
            const auto& request = inner_outputs[i];
            auto gradients = std::vector<std::string>();
            for (auto gradient: request.gradients()) {
                gradients.push_back(nlohmann::json(gradient).get<std::string>());
            }

            legacy_outputs.insert(request.name(), torch::make_intrusive<ModelOutputHolder>(
                /*quantity*/ "",
                request.unit(),
                nlohmann::json(request.sample_kind()).get<std::string>(),
                std::move(gradients),
                request.description()
            ));
        }

        auto values = module_.forward({
            c10::List<System>(torch_systems), legacy_outputs, torch_selected
        }).toGenericDict();

        for (auto i: batch) {
            const auto& name = inner_outputs[i].name();
            if (!values.contains(name)) {
                C10_THROW_ERROR(ValueError,
                    "the model did not compute the requested '" + name + "' output"
                );
            }
            outputs[i] = values.at(name).toCustomClass<metatensor_torch::TensorMapHolder>();
            if (device != systems_device) {
                outputs[i] = outputs[i]->to(torch::nullopt, systems_device);
            }
        }
    }

    auto backward_calls = BackwardCalls(outputs, autograd);
    auto results = std::vector<metatensor::TensorMap>();
    for (size_t i = 0; i < outputs.size(); i++) {
        auto blocks = std::vector<metatensor::TensorBlock>();
        for (const auto& block: metatensor_torch::TensorMapHolder::blocks(outputs[i])) {
            auto new_block = block_to_core(block);
            if (!autograd[i].empty()) {
                for (auto& [parameter, gradient]: autograd_gradients(
                    block, autograd[i], torch_systems, inputs, backward_calls
                )) {
                    new_block.add_gradient(parameter, std::move(gradient));
                }
            }
            blocks.push_back(std::move(new_block));
        }

        results.emplace_back(outputs[i]->keys()->as_metatensor(), std::move(blocks));
    }

    stream_context.finish();

    return results;
}

////////////////////////////////////////////////////////////////////////////////
//////                                                                    //////
//////                             PLUGIN                                 //////
//////                                                                    //////
////////////////////////////////////////////////////////////////////////////////

/// Check if `path` is a file exported by `AtomisticModel` with metatomic-torch
/// v0.1, from its `metatomic-version` record.
static bool is_legacy_model(const std::string& path) {
    try {
        auto reader = caffe2::serialize::PyTorchStreamReader(path);
        if (!reader.hasRecord("extra/metatomic-version")) {
            return false;
        }

        auto [data, size] = reader.getRecord("extra/metatomic-version");
        auto version = std::string(static_cast<const char*>(data.get()), size);
        return version.rfind("0.1.", 0) == 0;
    } catch (const std::exception&) {
        // not a file, or not a TorchScript archive
        return false;
    }
}

static std::unique_ptr<metatomic::BaseModel> load_model(
    const std::string& load_from,
    const std::map<std::string, std::string>& options
) {
    if (!is_legacy_model(load_from)) {
        return nullptr;
    }

    auto extensions_directory = c10::optional<std::string>();
    auto device = c10::optional<std::string>();
    for (const auto& [name, value]: options) {
        if (name == "device") {
            device = value;
        } else if (name == "extensions_directory") {
            extensions_directory = value;
        } else {
            throw metatomic::Error("unknown option for metatomic-torch models: '" + name + "'");
        }
    }

    auto model = load_atomistic_model(load_from, std::move(extensions_directory));

    return std::make_unique<LegacyModel>(model, device);
}

MTA_REGISTER_CXX_PLUGIN("metatomic-torch", load_model);
