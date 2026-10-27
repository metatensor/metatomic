#include <torch/script.h>

#include "metatomic/torch/system.hpp"
#include "metatomic/torch/model.hpp"
#include "metatomic/torch/misc.hpp"
#include "metatomic/torch/quantities.hpp"
#include "metatomic/torch/units.hpp"

#include "./internal/utils.hpp"

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

/// Small wrapper around unit_conversion_factor, with flexible
/// positional/keyword arguments.
///
/// This keeps the signature of older versions of this function, which also
/// accepted a 3-arg form `(quantity, from_unit, to_unit)`, since this signature
/// is required to load models exported with these versions. Calls to the old
/// function can be baked inside the models, and they will fail to load/execute
/// if we don't provide a function with the same signature.
static double unit_conversion_factor_backward_compatible(
    const torch::optional<std::string>& _0,
    const torch::optional<std::string>& _1,
    const torch::optional<std::string>& _2,
    const torch::optional<std::string>& quantity,
    const torch::optional<std::string>& from_unit,
    const torch::optional<std::string>& to_unit
) {
    // Count positional arguments
    int positional_count = (_0.has_value() ? 1 : 0) + (_1.has_value() ? 1 : 0) + (_2.has_value() ? 1 : 0);
    // Detect the removed 3-arg form:
    // - quantity keyword provided WITH from_unit or to_unit, OR
    // - 3 positional args (e.g., ("length", "angstrom", "nm")), OR
    // - 2 positional + to_unit keyword (e.g., ("length", "angstrom", to_unit="nm")), OR
    // - 1 positional + both from_unit and to_unit keywords (e.g., ("length", from_unit="angstrom", to_unit="nm"))
    // Note: quantity alone is treated as 2-arg form (quantity becomes from_unit)
    bool is_3arg = (quantity.has_value() && (from_unit.has_value() || to_unit.has_value())) ||
                        (positional_count == 3) ||
                        (positional_count == 2 && to_unit.has_value()) ||
                        (positional_count == 1 && from_unit.has_value() && to_unit.has_value());
    if (is_3arg) {
        throw std::runtime_error(
            "the 3-argument unit_conversion_factor(quantity, from_unit, to_unit) "
            "has been removed, use unit_conversion_factor(from_unit, to_unit) instead"
        );
    }

    // 2-arg form: (from_unit, to_unit)
    // Can be called as:
    //   - unit_conversion_factor("angstrom", "nm") → _0, _1
    //   - unit_conversion_factor(from_unit="angstrom", to_unit="nm") → keywords
    //   - unit_conversion_factor("angstrom", to_unit="nm") → _0, to_unit
    auto actual_from_unit = _0.has_value() ? _0.value() : from_unit.value_or("");
    auto actual_to_unit = _1.has_value() ? _1.value() : to_unit.value_or("");

    // Validate that both arguments are provided in 2-arg form
    bool has_first = _0.has_value() || from_unit.has_value();
    bool has_second = _1.has_value() || to_unit.has_value();
    bool has_any = has_first || has_second;

    if (!has_any || (has_first && !has_second) || (!has_first && has_second)) {
        throw std::runtime_error(
            "unit_conversion_factor requires 2 arguments: from_unit and to_unit"
        );
    }

    return unit_conversion_factor(actual_from_unit, actual_to_unit);
}

/// Copy all pair lists and custom data from `self` to `new_system`.
static void copy_data(System new_system, const System& self) {
    for (const auto& options: self->known_pairs()) {
        auto pairs = SystemHolder::pairs(self, options);
        new_system->add_pairs(options, pairs);
    }

    for (const auto& key: self->known_custom_data()) {
        auto data = SystemHolder::custom_data(self, key);
        new_system->add_custom_data(key, data);
    }
}

/// Implementation of a setter for the `type` property, that creates a new
/// System with the new types and copies all data over.
static void set_new_types(System self, torch::Tensor types) {
    auto new_system = torch::make_intrusive<SystemHolder>(
        self->length_unit(),
        std::move(types),
        self->positions(),
        self->cell(),
        self->pbc()
    );

    copy_data(new_system, self);
    self->swap(*new_system);
}

/// Implementation of a setter for the `positions` property, that creates a new
/// System with the new positions and copies all data over.
static void set_new_positions(System self, torch::Tensor positions) {
    auto new_system = torch::make_intrusive<SystemHolder>(
        self->length_unit(),
        self->types(),
        std::move(positions),
        self->cell(),
        self->pbc()
    );

    copy_data(new_system, self);
    self->swap(*new_system);
}

/// Implementation of a setter for the `cell` property, that creates a new
/// System with the new cell and copies all data over.
static void set_new_cell(System self, torch::Tensor cell) {
    auto new_system = torch::make_intrusive<SystemHolder>(
        self->length_unit(),
        self->types(),
        self->positions(),
        std::move(cell),
        self->pbc()
    );

    copy_data(new_system, self);
    self->swap(*new_system);
}



/// Implementation of a setter for the `pbc` property, that creates a new
/// System with the new pbc and copies all data over.
static void set_new_pbc(System self, torch::Tensor pbc) {
    auto new_system = torch::make_intrusive<SystemHolder>(
        self->length_unit(),
        self->types(),
        self->positions(),
        self->cell(),
        std::move(pbc)
    );

    copy_data(new_system, self);
    self->swap(*new_system);
}

TORCH_LIBRARY(metatomic, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/metatensor_torch/metatensor/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING;

    m.class_<PairListOptionsHolder>("NeighborListOptions")
        .def(
            torch::init<double, bool, bool, std::string>(), DOCSTRING,
            {torch::arg("cutoff"), torch::arg("full_list"), torch::arg("strict"), torch::arg("requestor") = ""}
        )
        .def_property("cutoff", &PairListOptionsHolder::cutoff)
        .def_property("length_unit", &PairListOptionsHolder::length_unit, &PairListOptionsHolder::set_length_unit)
        .def("engine_cutoff", &PairListOptionsHolder::engine_cutoff,
            DOCSTRING, {torch::arg("engine_length_unit")}
        )
        .def_property("full_list", &PairListOptionsHolder::full_list)
        .def_property("strict", &PairListOptionsHolder::strict)
        .def("requestors", &PairListOptionsHolder::requestors)
        .def("add_requestor", &PairListOptionsHolder::add_requestor, DOCSTRING,
            {torch::arg("requestor")}
        )
        .def("__repr__", &PairListOptionsHolder::repr)
        .def("__str__", &PairListOptionsHolder::str)
        .def("__eq__", static_cast<bool (*)(const PairListOptions&, const PairListOptions&)>(operator==))
        .def("__ne__", static_cast<bool (*)(const PairListOptions&, const PairListOptions&)>(operator!=))
        .def_pickle(
            [](const PairListOptions& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> PairListOptions {
                return PairListOptionsHolder::from_json(data);
            }
        );


    m.class_<SystemHolder>("System")
        .def(
            torch::init(
                [](torch::Tensor types, torch::Tensor positions, torch::Tensor cell, torch::Tensor pbc) {
                    return torch::make_intrusive<SystemHolder>(
                        /*length_unit=*/"",
                        std::move(types),
                        std::move(positions),
                        std::move(cell),
                        std::move(pbc)
                    );
                }
            ),
            DOCSTRING,
            {torch::arg("types"), torch::arg("positions"), torch::arg("cell"), torch::arg("pbc")}
        )
        .def_property("types", &SystemHolder::types, set_new_types)
        .def_property("positions", &SystemHolder::positions, set_new_positions)
        .def_property("cell", &SystemHolder::cell, set_new_cell)
        .def_property("pbc", &SystemHolder::pbc, set_new_pbc)
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
        .def("add_pairs", &SystemHolder::add_pairs, DOCSTRING,
            {torch::arg("options"), torch::arg("pairs")}
        )
        .def("pairs", &SystemHolder::pairs, DOCSTRING, {torch::arg("options")})
        .def("known_pairs", &SystemHolder::known_pairs)
        .def("add_custom_data", &SystemHolder::add_custom_data, DOCSTRING,
            {torch::arg("name"), torch::arg("data")}
        )
        .def("custom_data", &SystemHolder::custom_data, DOCSTRING, {torch::arg("name")})
        .def("known_custom_data", &SystemHolder::known_custom_data)
        // Deprecated functions, kept for backward compatibility with
        // previously exported TorchScript models.
        .def("add_neighbor_list", [](System self, PairListOptions options, metatensor_torch::TensorBlock neighbors) {
            WARN_DEPRECATION_ONCE(
                "`add_neighbor_list` is deprecated, use `add_pairs` instead"
            );
            self->add_pairs(std::move(options), std::move(neighbors));
        }, DOCSTRING,
            {torch::arg("options"), torch::arg("neighbors")}
        )
        .def("get_neighbor_list", [](System self, PairListOptions options) {
            WARN_DEPRECATION_ONCE(
                "`get_neighbor_list` is deprecated, use `pairs` instead"
            );
            return SystemHolder::pairs(std::move(self), std::move(options));
        }, DOCSTRING,
            {torch::arg("options")}
        )
        .def("known_neighbor_lists", [](System self) {
            WARN_DEPRECATION_ONCE(
                "`known_neighbor_lists` is deprecated, use `known_pairs` instead"
            );
            return self->known_pairs();
        })
        .def("add_data", [](System self, std::string name, metatensor_torch::TensorMap tensor, bool override, bool warn_on_deprecated) {
            if (warn_on_deprecated) {
                WARN_DEPRECATION_ONCE(
                    "`add_data` is deprecated, use `add_custom_data` instead"
                );
            }
            self->add_custom_data(std::move(name), std::move(tensor));
        }, DOCSTRING,
            {torch::arg("name"), torch::arg("tensor"), torch::arg("override") = true, torch::arg("_private_warn_on_deprecated") = true}
        )
        .def("get_data", [](System self, std::string name) {
            WARN_DEPRECATION_ONCE(
                "`get_data` is deprecated, use `custom_data` instead"
            );
            return SystemHolder::custom_data(std::move(self), std::move(name));
        }, DOCSTRING,
            {torch::arg("name")}
        )
        .def("known_data", [](System self) {
            WARN_DEPRECATION_ONCE(
                "`known_data` is deprecated, use `known_custom_data` instead"
            );
            return self->known_custom_data();
        })
        .def_pickle(
            // __getstate__: System -> torch.uint8 tensor (1D on CPU)
            [](const System& self) -> torch::Tensor { return save_buffer(self); },
            // __setstate__: torch.uint8 tensor (bytes) -> System
            [](const torch::Tensor& buffer) -> System { return load_buffer(buffer); }
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


#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

    m.class_<ModelOutputHolder>("ModelOutput")
        .def(
            torch::init<
                std::string,
                std::string,
                torch::IValue,
                std::vector<std::string>,
                std::string,
                torch::optional<bool>,
                torch::optional<std::string>
            >(),
            DOCSTRING, {
                torch::arg("quantity") = "",
                torch::arg("unit") = "",
                // support both positional and keyword argument for per_atom/sample_kind
                torch::arg("_2") = torch::IValue(),
                torch::arg("explicit_gradients") = std::vector<std::string>(),
                torch::arg("description") = "",
                torch::arg("per_atom") = std::nullopt,
                torch::arg("sample_kind") = std::nullopt,
            }
        )
        .def_readwrite("description", &ModelOutputHolder::description)
        .def_property("quantity", &ModelOutputHolder::quantity, &ModelOutputHolder::set_quantity)
        .def_property("unit", &ModelOutputHolder::unit, &ModelOutputHolder::set_unit)
        .def_property("per_atom", &ModelOutputHolder::get_per_atom, &ModelOutputHolder::set_per_atom)
        .def_property("sample_kind", &ModelOutputHolder::sample_kind, &ModelOutputHolder::set_sample_kind)
        .def_readwrite("explicit_gradients", &ModelOutputHolder::explicit_gradients)
        .def_pickle(
            [](const ModelOutput& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelOutput {
                return ModelOutputHolder::from_json(data);
            }
        );

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

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
        .def_property("outputs", &ModelCapabilitiesHolder::outputs, (void (ModelCapabilitiesHolder::*)(torch::Dict<std::string, ModelOutput>))&ModelCapabilitiesHolder::set_outputs)
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

    m.def(
        "unit_conversion_factor("
            "str? _0 = None, "
            "str? _1 = None, "
            "str? _2 = None, "
            "str? quantity = None, "
            "str? from_unit = None, "
            "str? to_unit = None) -> float",
        unit_conversion_factor_backward_compatible
    );

    m.def("unit_dimension_for_quantity(str name) -> str", unit_dimension_for_quantity);

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
        [&](const torch::Tensor& buffer) -> System { return load_buffer(buffer); }
    );

    m.def("save(str path, __torch__.torch.classes.metatomic.System system) -> ()",
        [&](const std::string& path, const System& system) {
            save(path, system);
        }
    );
    m.def("load_system(str path) -> __torch__.torch.classes.metatomic.System",
        [&](const std::string& path) -> System {
            return load(path);
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
    m.def(std::move(schema), [](
        std::vector<System> systems,
        c10::Dict<std::string, ModelOutput> requested,
        torch::optional<metatensor_torch::Labels> selected_atoms,
        c10::Dict<std::string, metatensor_torch::TensorMap> outputs,
        std::string model_dtype
    ) {
        WARN_DEPRECATION_ONCE(
            "_check_outputs is deprecated and will be removed in a future version. "
            "Please re-export this model with a recent metatomic."
        );
        check_quantities(systems, requested, selected_atoms, outputs, model_dtype, "outputs");
    });

    schema = c10::FunctionSchema(
        /*name=*/"_check_quantities",
        /*overload_name=*/"_check_quantities",
        /*arguments=*/{
            c10::Argument("systems", c10::getTypePtr<std::vector<System>>()),
            c10::Argument("requested", c10::getTypePtr<c10::Dict<std::string, ModelOutput>>()),
            c10::Argument("selected_atoms", c10::getTypePtr<torch::optional<metatensor_torch::Labels>>()),
            c10::Argument("values", c10::getTypePtr<c10::Dict<std::string, metatensor_torch::TensorMap>>()),
            c10::Argument("model_dtype", c10::getTypePtr<std::string>()),
            c10::Argument("inputs_or_outputs", c10::getTypePtr<std::string>()),
        },
        /*returns=*/{}
    );
    schema.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
    m.def(std::move(schema), check_quantities);
}
