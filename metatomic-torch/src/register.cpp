#include <torch/script.h>

#include "metatomic/torch/system.hpp"
#include "metatomic/torch/model.hpp"
#include "metatomic/torch/misc.hpp"
#include "metatomic/torch/outputs.hpp"

using namespace metatomic_torch;

std::string pick_device_pywrapper(
    const std::vector<std::string> &model_devices,
    const c10::optional<std::string> &requested_device
) {
    try {
        torch::optional<std::string> desired = torch::nullopt;
        if (requested_device.has_value() && !requested_device->empty()) {
            desired = requested_device.value();
        }

        c10::DeviceType devtype = metatomic_torch::pick_device(model_devices, desired);

        if (desired.has_value()) {
            // User requested a specific device (possibly with an index like "cuda:1").
            // We return it normalized (e.g. "CUDA:1" -> "cuda:1").
            return torch::Device(desired.value()).str();
        } else {
            // Automatic selection: return the device type name (e.g. "cuda").
            return torch::Device(devtype).str();
        }

    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("pick_device failed: ") + e.what());
    }
}

TORCH_LIBRARY(metatomic, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/metatensor_torch/metatensor/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING;

    m.class_<NeighborListOptionsHolder>("NeighborListOptions")
        .def(
            torch::init<double, bool, bool, std::string>(), DOCSTRING,
            {torch::arg("cutoff"), torch::arg("full_list"), torch::arg("strict"), torch::arg("requestor") = ""}
        )
        .def_property("cutoff", &NeighborListOptionsHolder::cutoff)
        .def_property("length_unit", &NeighborListOptionsHolder::length_unit, &NeighborListOptionsHolder::set_length_unit)
        .def("engine_cutoff", &NeighborListOptionsHolder::engine_cutoff,
            DOCSTRING, {torch::arg("engine_length_unit")}
        )
        .def_property("full_list", &NeighborListOptionsHolder::full_list)
        .def_property("strict", &NeighborListOptionsHolder::strict)
        .def("requestors", &NeighborListOptionsHolder::requestors)
        .def("add_requestor", &NeighborListOptionsHolder::add_requestor, DOCSTRING,
            {torch::arg("requestor")}
        )
        .def("__repr__", &NeighborListOptionsHolder::repr)
        .def("__str__", &NeighborListOptionsHolder::str)
        .def("__eq__", static_cast<bool (*)(const NeighborListOptions&, const NeighborListOptions&)>(operator==))
        .def("__ne__", static_cast<bool (*)(const NeighborListOptions&, const NeighborListOptions&)>(operator!=))
        .def_pickle(
            [](const NeighborListOptions& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> NeighborListOptions {
                return NeighborListOptionsHolder::from_json(data);
            }
        );


    m.class_<SystemHolder>("System")
        .def(
            torch::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(), DOCSTRING,
            {torch::arg("types"), torch::arg("positions"), torch::arg("cell"), torch::arg("pbc")}
        )
        .def_property("types", &SystemHolder::types, &SystemHolder::set_types)
        .def_property("positions", &SystemHolder::positions, &SystemHolder::set_positions)
        .def_property("cell", &SystemHolder::cell, &SystemHolder::set_cell)
        .def_property("pbc", &SystemHolder::pbc, &SystemHolder::set_pbc)
        .def("__len__", &SystemHolder::size)
        .def("__str__", &SystemHolder::str)
        .def("__repr__", &SystemHolder::str)
        .def_property("device", &SystemHolder::device)
        .def_property("dtype", &SystemHolder::scalar_type)
        .def("to", &SystemHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("non_blocking") = false
        })
        .def("add_neighbor_list", &SystemHolder::add_neighbor_list, DOCSTRING,
            {torch::arg("options"), torch::arg("neighbors")}
        )
        .def("get_neighbor_list", &SystemHolder::get_neighbor_list, DOCSTRING,
            {torch::arg("options")}
        )
        .def("known_neighbor_lists", &SystemHolder::known_neighbor_lists)
        .def("add_data", &SystemHolder::add_data, DOCSTRING,
            {torch::arg("name"), torch::arg("tensor"), torch::arg("override") = false}
        )
        .def("get_data", &SystemHolder::get_data, DOCSTRING,
            {torch::arg("name")}
        )
        .def("known_data", &SystemHolder::known_data)
        .def_pickle(
            // __getstate__: System -> torch.uint8 tensor (1D on CPU)
            [](const System& self) -> torch::Tensor { return save_buffer(self); },
            // __setstate__: torch.uint8 tensor (bytes) -> System
            [](const torch::Tensor& buffer) -> System { return load_system_buffer(buffer); }
        );

    m.class_<ModelMetadataHolder>("ModelMetadata")
        .def(
            torch::init<
                std::string,
                std::string,
                std::vector<std::string>,
                torch::Dict<std::string, std::vector<std::string>>,
                torch::Dict<std::string, std::string>
            >(),
            DOCSTRING, {
                torch::arg("name") = "",
                torch::arg("description") = "",
                torch::arg("authors") = std::vector<std::string>(),
                torch::arg("references") = torch::Dict<std::string, std::vector<std::string>>(),
                torch::arg("extra") = torch::Dict<std::string, std::string>(),
            }
        )
        .def("__repr__", &ModelMetadataHolder::print)
        .def("__str__", &ModelMetadataHolder::print)
        .def_readwrite("name", &ModelMetadataHolder::name)
        .def_readwrite("description", &ModelMetadataHolder::description)
        .def_readwrite("authors", &ModelMetadataHolder::authors)
        .def_readwrite("references", &ModelMetadataHolder::references)
        .def_readwrite("extra", &ModelMetadataHolder::extra)
        .def_pickle(
            [](const ModelMetadata& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelMetadata {
                return ModelMetadataHolder::from_json(data);
            }
        );


    m.class_<ModelOutputHolder>("ModelOutput")
        .def(
            torch::init<
                std::string,
                std::string,
                bool,
                std::vector<std::string>,
                std::string
            >(),
            DOCSTRING, {
                torch::arg("quantity") = "",
                torch::arg("unit") = "",
                torch::arg("per_atom") = false,
                torch::arg("explicit_gradients") = std::vector<std::string>(),
                torch::arg("description") = "",
            }
        )
        .def_readwrite("description", &ModelOutputHolder::description)
        .def_property("quantity", &ModelOutputHolder::quantity, &ModelOutputHolder::set_quantity)
        .def_property("unit", &ModelOutputHolder::unit, &ModelOutputHolder::set_unit)
        .def_readwrite("per_atom", &ModelOutputHolder::per_atom)
        .def_readwrite("explicit_gradients", &ModelOutputHolder::explicit_gradients)
        .def_pickle(
            [](const ModelOutput& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelOutput {
                return ModelOutputHolder::from_json(data);
            }
        );


    m.class_<ModelCapabilitiesHolder>("ModelCapabilities")
        .def(
            torch::init<
                torch::Dict<std::string, ModelOutput>,
                std::vector<int64_t>,
                double,
                std::string,
                std::vector<std::string>,
                std::string
            >(),
            DOCSTRING, {
                torch::arg("outputs") = torch::Dict<std::string, ModelOutput>(),
                torch::arg("atomic_types") = std::vector<int64_t>(),
                torch::arg("interaction_range") = -1.0,
                torch::arg("length_unit") = "",
                torch::arg("supported_devices") = std::vector<std::string>{},
                torch::arg("dtype") = "",
            }
        )
        .def_property("outputs", &ModelCapabilitiesHolder::outputs, &ModelCapabilitiesHolder::set_outputs)
        .def_readwrite("atomic_types", &ModelCapabilitiesHolder::atomic_types)
        .def_readwrite("interaction_range", &ModelCapabilitiesHolder::interaction_range)
        .def("engine_interaction_range", &ModelCapabilitiesHolder::engine_interaction_range)
        .def_property("length_unit", &ModelCapabilitiesHolder::length_unit, &ModelCapabilitiesHolder::set_length_unit)
        .def_readwrite("supported_devices", &ModelCapabilitiesHolder::supported_devices)
        .def_property("dtype", &ModelCapabilitiesHolder::dtype, &ModelCapabilitiesHolder::set_dtype)
        .def_pickle(
            [](const ModelCapabilities& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelCapabilities {
                return ModelCapabilitiesHolder::from_json(data);
            }
        );


    m.class_<ModelEvaluationOptionsHolder>("ModelEvaluationOptions")
        .def(
            torch::init<
                std::string,
                torch::Dict<std::string, ModelOutput>,
                torch::optional<metatensor_torch::Labels>
            >(),
            DOCSTRING, {
                torch::arg("length_unit") = "",
                torch::arg("outputs") = torch::Dict<std::string, ModelOutput>(),
                torch::arg("selected_atoms") = torch::nullopt,
            }
        )
        .def_property("length_unit", &ModelEvaluationOptionsHolder::length_unit, &ModelEvaluationOptionsHolder::set_length_unit)
        .def_readwrite("outputs", &ModelEvaluationOptionsHolder::outputs)
        .def_property("selected_atoms",
            &ModelEvaluationOptionsHolder::get_selected_atoms,
            &ModelEvaluationOptionsHolder::set_selected_atoms
        )
        .def_pickle(
            [](const ModelEvaluationOptions& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelEvaluationOptions {
                return ModelEvaluationOptionsHolder::from_json(data);
            }
        );

    // standalone functions
    m.def("version() -> str", version);
    // Expose pick_device to Python. The C++ helper returns a c10::DeviceType;
    // build a torch::Device from it (with default index policy) and return its
    // string representation to Python (backwards-compatible).
    m.def(
        "pick_device(str[] model_devices, str? requested_device = None) -> str",
        &pick_device_pywrapper
    );
    m.def("pick_output(str requested_output, Dict(str, __torch__.torch.classes.metatomic.ModelOutput) outputs, str? desired_variant = None) -> str", pick_output);

    m.def("read_model_metadata(str path) -> __torch__.torch.classes.metatomic.ModelMetadata", read_model_metadata);
    m.def("unit_conversion_factor(str quantity, str from_unit, str to_unit) -> float", unit_conversion_factor);

    // manually construct the schema for "check_atomistic_model(str path) -> ()",
    // so we can set AliasAnalysisKind to CONSERVATIVE. In turn, this make it so
    // the TorchScript compiler knows this function has side-effects, and does
    // not remove it from the graph.
    auto schema = c10::FunctionSchema(
        /*name=*/"check_atomistic_model",
        /*overload_name=*/"check_atomistic_model",
        /*arguments=*/{
            c10::Argument("path", c10::getTypePtr<std::string>()),
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), check_atomistic_model);

    // "load_model_extensions(str path, str? extensions_directory) -> ()"
    schema = c10::FunctionSchema(
        /*name=*/"load_model_extensions",
        /*overload_name=*/"load_model_extensions",
        /*arguments=*/{
            c10::Argument("path", c10::getTypePtr<std::string>()),
            c10::Argument("extensions_directory", c10::getTypePtr<c10::optional<std::string>>()),
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), load_model_extensions);

    // "register_autograd_neighbors("
    //     "__torch__.torch.classes.metatomic.System system, "
    //     "__torch__.torch.classes.metatensor.TensorBlock neighbors, "
    //     "bool check_consistency = False"
    // ") -> ()",
    schema = c10::FunctionSchema(
        /*name=*/"register_autograd_neighbors",
        /*overload_name=*/"register_autograd_neighbors",
        /*arguments=*/{
            c10::Argument("system", c10::getTypePtr<System>()),
            c10::Argument("neighbors", c10::getTypePtr<metatensor_torch::TensorBlock>()),
            c10::Argument("check_consistency", c10::getTypePtr<bool>(), c10::nullopt, /*default_value=*/false),
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), register_autograd_neighbors);

    m.def("save_buffer(__torch__.torch.classes.metatomic.System system) -> Tensor",
        [&](const System& system) { return save_buffer(system); }
    );
    m.def("load_system_buffer(Tensor buffer) -> __torch__.torch.classes.metatomic.System",
        [&](const torch::Tensor& buffer) -> System { return load_system_buffer(buffer); }
    );

    m.def("save(str path, __torch__.torch.classes.metatomic.System system) -> ()",
        [&](const std::string& path, const System& system) {
            save(path, system);
        }
    );
    m.def("load_system(str path) -> __torch__.torch.classes.metatomic.System",
        [&](const std::string& path) -> System {
            return load_system(path);
        }
    );

    // "_check_outputs("
    //     "__torch__.torch.classes.metatomic.System[] systems, "
    //     "Dict[str, __torch__.torch.classes.metatomic.ModelOutput] requested, "
    //     "__torch__.torch.classes.metatensor.Labels? selected_atoms, "
    //     "Dict[str, __torch__.torch.classes.metatensor.TensorMap] outputs, "
    //     "str model_dtype"
    // ") -> ()",
    schema = c10::FunctionSchema(
        /*name=*/"_check_outputs",
        /*overload_name=*/"_check_outputs",
        /*arguments=*/{
            c10::Argument("systems", c10::getTypePtr<std::vector<System>>()),
            c10::Argument("requested", c10::getTypePtr<c10::Dict<std::string, ModelOutput>>()),
            c10::Argument("selected_atoms", c10::getTypePtr<torch::optional<metatensor_torch::Labels>>()),
            c10::Argument("outputs", c10::getTypePtr<c10::Dict<std::string, metatensor_torch::TensorMap>>()),
            c10::Argument("model_dtype", c10::getTypePtr<std::string>()),
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), check_outputs);
}
