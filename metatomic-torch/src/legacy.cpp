#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/legacy.hpp"
#include "metatomic/torch/model.hpp"

#include "./internal/dlpack.hpp"

using namespace metatomic_torch;
using metatomic::Gradients;

// ========================================================================== //
// Conversions between metatensor (core) and metatensor-torch, sharing the
// data instead of copying it. metatensor-torch can only read blocks containing
// `TorchDataArray`, so the values of other arrays go through DLPack.

static metatensor_torch::Labels labels_to_torch(metatensor::Labels labels, torch::Device device) {
    auto result = torch::make_intrusive<metatensor_torch::LabelsHolder>(std::move(labels));
    return result->device() == device ? result : result->to(device);
}

static metatensor_torch::TensorBlock block_to_torch(metatensor::TensorBlock block, torch::Device device) {
    auto array = block.mts_array();
    auto values = details::dlpack_to_tensor(metatomic::DLPackTensor(array.as_dlpack(
        array.device(), /*stream*/ nullptr, {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION}
    )));

    auto components = std::vector<metatensor_torch::Labels>();
    for (auto& component: block.components()) {
        components.push_back(labels_to_torch(std::move(component), device));
    }

    auto result = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
        std::move(values),
        labels_to_torch(block.samples(), device),
        std::move(components),
        labels_to_torch(block.properties(), device)
    );

    for (const auto& parameter: block.gradients_list()) {
        result->add_gradient(parameter, block_to_torch(block.gradient(parameter), device));
    }

    return result;
}

static metatensor_torch::TensorMap tensor_to_torch(metatensor::TensorMap tensor, torch::Device device) {
    auto blocks = std::vector<metatensor_torch::TensorBlock>();
    auto keys = tensor.keys();
    for (uintptr_t i = 0; i < keys.count(); i++) {
        blocks.push_back(block_to_torch(tensor.block_by_id(i), device));
    }
    return torch::make_intrusive<metatensor_torch::TensorMapHolder>(
        labels_to_torch(std::move(keys), device), std::move(blocks)
    );
}

static std::vector<metatensor::Labels> labels_to_core(const std::vector<metatensor_torch::Labels>& labels) {
    auto result = std::vector<metatensor::Labels>();
    for (const auto& entry: labels) {
        result.push_back(entry->as_metatensor());
    }
    return result;
}

/// Create a block with `values` and the labels of `block`, `components` being
/// added before the components of `block`.
static metatensor::TensorBlock block_with_values(
    torch::Tensor values,
    const metatensor::Labels& samples,
    std::vector<metatensor::Labels> components,
    const metatensor_torch::TensorBlock& block
) {
    for (auto& component: labels_to_core(block->components())) {
        components.push_back(std::move(component));
    }

    return metatensor::TensorBlock(
        std::make_unique<metatensor_torch::TorchDataArray>(std::move(values)),
        samples,
        components,
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

// ========================================================================== //

static std::string json_string(const nlohmann::json& value) {
    return value.get<std::string>();
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

static bool contains(const std::vector<Gradients>& gradients, Gradients gradient) {
    return std::find(gradients.begin(), gradients.end(), gradient) != gradients.end();
}

static metatomic::Quantity with_gradients(const metatomic::Quantity& quantity, std::vector<Gradients> gradients) {
    return metatomic::Quantity::builder()
        .name(quantity.name())
        .unit(quantity.unit())
        .sample_kind(quantity.sample_kind())
        .description(quantity.description())
        .gradients(std::move(gradients))
        .build();
}

static const metatomic::Quantity* find_output(
    const metatomic::ModelCapabilities& capabilities,
    const metatomic::Quantity& request
) {
    for (const auto& output: capabilities.outputs()) {
        if (output.name() == request.name() && output.sample_kind() == request.sample_kind()) {
            return &output;
        }
    }
    return nullptr;
}

// ========================================================================== //

static torch::jit::Module checked_atomistic_model(torch::jit::Module model) {
    for (const auto* method: {"capabilities", "metadata", "requested_neighbor_lists", "requested_inputs"}) {
        if (!model.find_method(method)) {
            C10_THROW_ERROR(TypeError,
                "this TorchScript module is not an AtomisticModel: it does not "
                "have a '" + std::string(method) + "' method"
            );
        }
    }

    if (!model.hasattr("module")) {
        C10_THROW_ERROR(TypeError,
            "this TorchScript module is not an AtomisticModel: it does not "
            "have a 'module' attribute"
        );
    }

    auto capabilities = model.run_method("capabilities").toCustomClass<ModelCapabilitiesHolder>();
    if (capabilities->length_unit().empty()) {
        C10_THROW_ERROR(ValueError,
            "this model does not declare a length unit, and can not be used "
            "with LegacyModel"
        );
    }

    return model;
}

static metatomic::ModelCapabilities declared_capabilities(torch::jit::Module& model) {
    auto capabilities = model.run_method("capabilities").toCustomClass<ModelCapabilitiesHolder>();
    auto all_outputs = capabilities->outputs();

    // AtomisticModel also lists outputs under their deprecated or current
    // names, only keep the names declared by the model. Models exported before
    // this list existed did not do this.
    auto names = std::vector<std::string>();
    if (model.hasattr("_model_capabilities_outputs_names")) {
        for (const auto& name: model.attr("_model_capabilities_outputs_names").toListRef()) {
            names.push_back(name.toStringRef());
        }
    } else {
        for (const auto& item: all_outputs) {
            names.push_back(item.key());
        }
    }

    auto outputs = std::vector<metatomic::Quantity>();
    for (const auto& name: names) {
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

static std::vector<metatomic::Quantity> legacy_requested_inputs(torch::jit::Module& model) {
    auto method = model.get_method("requested_inputs");
    // `use_new_names` was added to AtomisticModel.requested_inputs() together
    // with the deprecated names, older models do not take it
    auto arguments = std::vector<c10::IValue>();
    if (method.function().getSchema().arguments().size() > 1) {
        arguments.emplace_back(/*use_new_names*/ true);
    }

    auto inputs = std::vector<metatomic::Quantity>();
    for (const auto& item: method(std::move(arguments)).toGenericDict()) {
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

static std::vector<PairListOptions> legacy_pair_lists(torch::jit::Module& model) {
    auto result = std::vector<PairListOptions>();
    for (const auto& options: model.run_method("requested_neighbor_lists").toListRef()) {
        result.push_back(options.toCustomClass<PairListOptionsHolder>());
    }
    return result;
}

LegacyModel::LegacyModel(const std::string& path, c10::optional<std::string> extensions_directory):
    LegacyModel(load_atomistic_model(path, std::move(extensions_directory)))
{}

LegacyModel::LegacyModel(torch::jit::Module model):
    module_(checked_atomistic_model(model).attr("module").toModule()),
    declared_(declared_capabilities(model)),
    metadata_(model.run_method("metadata").toCustomClass<ModelMetadataHolder>()->as_metatomic()),
    pair_lists_(legacy_pair_lists(model)),
    inputs_(legacy_requested_inputs(model))
{}

metatomic::ModelCapabilities LegacyModel::capabilities() const {
    // add the gradients computed with autograd to the declared outputs
    auto outputs = std::vector<metatomic::Quantity>();
    for (const auto& output: declared_.outputs()) {
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

    return metatomic::ModelCapabilities::builder()
        .outputs(std::move(outputs))
        .atomic_types(declared_.atomic_types())
        .interaction_range(declared_.interaction_range())
        .length_unit(declared_.length_unit())
        .supported_devices(declared_.supported_devices())
        .dtype(declared_.dtype())
        .build();
}

std::vector<metatomic::PairListOptions> LegacyModel::requested_pair_lists(
    const std::vector<metatomic::Quantity>& /*requested_outputs*/
) const {
    auto result = std::vector<metatomic::PairListOptions>();
    for (const auto& options: pair_lists_) {
        result.push_back(options->as_metatomic());
        result.back().add_requestor("LegacyModel");
    }
    return result;
}

std::vector<metatomic::Quantity> LegacyModel::requested_inputs(
    const std::vector<metatomic::Quantity>& /*requested_outputs*/
) const {
    return inputs_;
}

// ========================================================================== //

namespace {
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
}

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
        const auto* output = find_output(declared, request);
        auto explicit_gradients = std::vector<Gradients>();
        auto autograd_gradients = std::vector<Gradients>();
        for (auto gradient: request.gradients()) {
            if (output != nullptr && contains(output->gradients(), gradient)) {
                explicit_gradients.push_back(gradient);
            } else {
                autograd_gradients.push_back(gradient);
            }
        }

        if (!autograd_gradients.empty() && !supports_autograd(request)) {
            C10_THROW_ERROR(ValueError,
                "gradients of '" + request.name() + "' with sample kind '" +
                json_string(request.sample_kind()) + "' can not be computed with autograd"
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
    const std::vector<PairListOptions>& pair_lists,
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

    // metatomic-core does not allow adding pairs to the engine's system, and
    // metatensor-torch can not read pairs stored in non-torch arrays, so the
    // model gets a new system, sharing the same data
    auto result = torch::make_intrusive<SystemHolder>(
        view->length_unit(), view->types(), positions, cell, view->pbc()
    );

    for (const auto& options: pair_lists) {
        auto pairs = block_to_torch(system.pairs(options->as_metatomic()), view->device());
        if (positions_gradients || strain_gradients) {
            register_autograd_neighbors(result, pairs, /*check_consistency*/ false);
        }
        result->add_pairs(options, pairs);
    }

    for (const auto& name: system.known_custom_data()) {
        result->add_custom_data(name, tensor_to_torch(system.custom_data(name), view->device()));
    }

    return result;
}

static metatensor::Labels xyz_labels(const std::string& name) {
    return metatensor::Labels({name}, {{0}, {1}, {2}});
}

/// Compute the `gradients` of the per-system `block` with respect to the
/// positions and/or strain of `systems`, using autograd.
///
/// The gradients are computed separately for each entry along the components
/// and properties of `block`, but simultaneously for all samples, since each
/// system has a single sample and the different systems are independent.
static std::vector<std::pair<std::string, metatensor::TensorBlock>> autograd_gradients(
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

    auto positions = contains(gradients, Gradients::Positions);
    auto strain = contains(gradients, Gradients::Strain);

    // inputs for autograd: positions and/or strain for all the systems
    auto parameters = std::vector<torch::Tensor>();
    for (const auto& system_inputs: inputs) {
        if (positions) {
            parameters.push_back(system_inputs.positions);
        }
        if (strain) {
            parameters.push_back(system_inputs.strain);
        }
    }

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

    auto n_parameters = static_cast<int64_t>(positions) + static_cast<int64_t>(strain);
    auto grads_for = [&](int64_t system, bool is_strain) {
        return stacked[static_cast<size_t>(system * n_parameters + (is_strain && positions ? 1 : 0))];
    };

    auto gradient_shape = [&](std::vector<int64_t> shape) {
        shape.insert(shape.end(), values_shape.begin(), values_shape.end());
        return shape;
    };

    auto results = std::vector<std::pair<std::string, metatensor::TensorBlock>>();
    if (positions) {
        auto grad_samples = std::vector<int32_t>();
        auto grad_values = std::vector<torch::Tensor>();
        for (int64_t sample = 0; sample < n_samples; sample++) {
            auto system = sample_systems[static_cast<size_t>(sample)];
            for (int64_t atom = 0; atom < systems[static_cast<size_t>(system)]->size(); atom++) {
                grad_samples.insert(grad_samples.end(), {
                    static_cast<int32_t>(sample), static_cast<int32_t>(system), static_cast<int32_t>(atom)
                });
            }
            grad_values.push_back(grads_for(system, false));
        }

        auto grad_samples_labels = metatensor::Labels(
            {"sample", "system", "atom"}, grad_samples.data(), grad_samples.size() / 3, metatensor::assume_unique{}
        );
        results.emplace_back("positions", block_with_values(
            grad_values.empty() ? torch::zeros(gradient_shape({0, 3}), values.options()) : torch::cat(grad_values),
            grad_samples_labels,
            {xyz_labels("xyz")},
            block
        ));
    }

    if (strain) {
        auto grad_values = std::vector<torch::Tensor>();
        auto grad_samples = std::vector<int32_t>();
        for (int64_t sample = 0; sample < n_samples; sample++) {
            grad_values.push_back(grads_for(sample_systems[static_cast<size_t>(sample)], true));
            grad_samples.push_back(static_cast<int32_t>(sample));
        }

        auto grad_samples_labels = metatensor::Labels(
            {"sample"}, grad_samples.data(), grad_samples.size(), metatensor::assume_unique{}
        );
        results.emplace_back("strain", block_with_values(
            grad_values.empty() ? torch::zeros(gradient_shape({0, 3, 3}), values.options()) : torch::stack(grad_values),
            grad_samples_labels,
            {xyz_labels("xyz_1"), xyz_labels("xyz_2")},
            block
        ));
    }

    return results;
}

std::vector<metatensor::TensorMap> LegacyModel::execute_inner(
    const std::vector<metatomic::System>& systems,
    std::optional<metatensor::Labels> selected_atoms,
    const std::vector<metatomic::Quantity>& requested_outputs
) {
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

    auto device = torch_systems.empty() ? torch::Device(torch::kCPU) : torch_systems[0]->device();
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
                gradients.push_back(json_string(gradient));
            }

            legacy_outputs.insert(request.name(), torch::make_intrusive<ModelOutputHolder>(
                /*quantity*/ "",
                request.unit(),
                json_string(request.sample_kind()),
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

    return results;
}
