#include <torch/script.h>

#include "./register.hpp"

#include "metatomic/torch/legacy/output.hpp"
#include "metatomic/torch/legacy/model.hpp"
#include "metatomic/torch/legacy/quantities.hpp"
#include "metatomic/torch/legacy/capabilities.hpp"


#include "../internal/utils.hpp"

using namespace metatomic_torch;

void metatomic_torch::details::register_legacy(torch::Library& m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/metatensor_torch/metatensor/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING;


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

    m.def("pick_output(str requested_output, Dict(str, __torch__.torch.classes.metatomic.ModelOutput) outputs, str? desired_variant = None) -> str", pick_output);

    m.def("read_model_metadata(str path) -> __torch__.torch.classes.metatomic.ModelMetadata", read_model_metadata);

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
