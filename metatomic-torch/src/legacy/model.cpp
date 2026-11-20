#include <cstring>
#include <iostream>

#include <sstream>
#include <filesystem>
#include <vector>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor/torch.hpp>

#include "metatomic/torch/legacy/model.hpp"
#include "metatomic/torch/misc.hpp"

#include "../internal/shared_libraries.hpp"

using namespace metatomic_torch;

static void read_vector_string_json(
    std::vector<std::string>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_string()) {
            throw std::runtime_error(context + " must be an array of string");
        }
        output.emplace_back(value);
    }
}

template<typename T>
static void read_vector_int_json(
    std::vector<T>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_number_integer()) {
            throw std::runtime_error(context + " must be an array of integers");
        }
        output.emplace_back(value);
    }
}


static void check_selected_atoms(const torch::optional<metatensor_torch::Labels>& selected_atoms) {
    if (selected_atoms) {
        if (selected_atoms.value()->names() != std::vector<std::string>{"system", "atom"}) {
            std::ostringstream oss;
            oss << '[';
            for (const auto& name: selected_atoms.value()->names()) {
                oss << '\'' << name << "', ";
            }
            oss << ']';

            C10_THROW_ERROR(ValueError,
                "invalid `selected_atoms` names: expected ['system', 'atom'], "
                "got " + oss.str()
            );
        }
    }
}

void ModelEvaluationOptionsHolder::set_length_unit(std::string unit) {
    this->length_unit_ = std::move(unit);
}

ModelEvaluationOptionsHolder::ModelEvaluationOptionsHolder(
    std::string length_unit_,
    torch::Dict<std::string, ModelOutput> outputs_,
    torch::optional<metatensor_torch::Labels> selected_atoms
):
    outputs(outputs_),
    selected_atoms_(std::move(selected_atoms))
{
    this->set_length_unit(std::move(length_unit_));
    check_selected_atoms(selected_atoms_);
}


void ModelEvaluationOptionsHolder::set_selected_atoms(torch::optional<metatensor_torch::Labels> selected_atoms) {
    check_selected_atoms(selected_atoms);
    selected_atoms_ = std::move(selected_atoms);
}


std::string ModelEvaluationOptionsHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelEvaluationOptions";
    result["length_unit"] = this->length_unit();

    if (this->selected_atoms_) {
        const auto& selected_atoms = this->selected_atoms_.value();

        auto selected_json = nlohmann::json::object();
        selected_json["names"] = selected_atoms->names();
        auto values = selected_atoms->values().to(torch::kCPU).contiguous();
        auto size = static_cast<size_t>(selected_atoms->size() * selected_atoms->count());
        selected_json["values"] = std::vector<int32_t>(
            values.data_ptr<int32_t>(),
            values.data_ptr<int32_t>() + size
        );

        result["selected_atoms"] = std::move(selected_json);
    } else {
        result["selected_atoms"] = nlohmann::json();
    }

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = details::model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelEvaluationOptions ModelEvaluationOptionsHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelEvaluationOptions, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelEvaluationOptions, did not find it");
    }

    if (data["class"] != "ModelEvaluationOptions") {
        throw std::runtime_error("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'");
    }

    auto result = torch::make_intrusive<ModelEvaluationOptionsHolder>();
    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelEvaluationOptions must be a string");
        }
        result->set_length_unit(data["length_unit"]);
    }

    if (data.contains("selected_atoms")) {
        if (data["selected_atoms"].is_null()) {
            // nothing to do
        } else {
            if (!data["selected_atoms"].is_object()) {
                throw std::runtime_error("'selected_atoms' in JSON for ModelEvaluationOptions must be an object");
            }

            if (!data["selected_atoms"].contains("names")) {
                throw std::runtime_error("'selected_atoms.names' in JSON for ModelEvaluationOptions must be an array");
            }

            auto names = std::vector<std::string>();
            read_vector_string_json(
                names,
                data["selected_atoms"]["names"],
                "'selected_atoms.names' in JSON for ModelEvaluationOptions"
            );

            if (!data["selected_atoms"].contains("values")) {
                throw std::runtime_error("'selected_atoms.values' in JSON for ModelEvaluationOptions must be an array");
            }

            auto values = std::vector<int32_t>();
            read_vector_int_json(
                values,
                data["selected_atoms"]["values"],
                "'selected_atoms.values' in JSON for ModelEvaluationOptions"
            );
            assert(values.size() % 2 == 0);

            result->set_selected_atoms(torch::make_intrusive<metatensor_torch::LabelsHolder>(
                std::move(names),
                torch::tensor(values).reshape({-1, 2})
            ));
        }
    }

    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelEvaluationOptions must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), ModelOutputHolder::from_json(output.value().dump()));
        }
    }

    return result;
}

/******************************************************************************/

struct Version {
    Version(std::string version): string(std::move(version)) {
        size_t pos = 0;
        try {
            this->major = std::stoi(this->string, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        if (this->string[pos] != '.' || this->string.size() == pos) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        auto minor_version = this->string.substr(pos + 1);
        try {
            this->minor = std::stoi(minor_version, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }
    }

    /// Check if two version are compatible. `same_minor` indicates whether two
    /// versions should have the same major AND minor number to be considered
    /// compatible.
    bool is_compatible(const Version& other, bool same_minor = false) const {
        if (this->major != other.major) {
            return false;
        }

        if (this->major == 0) {
            same_minor = true;
        }

        if (same_minor && this->minor != other.minor) {
            return false;
        }

        return true;
    }

    std::string string;
    int major = 0;
    int minor = 0;
};

struct Library {
    std::string name;
    std::string path;
};

void from_json(const nlohmann::json& json, Library& extension) {
    json.at("name").get_to(extension.name);
    json.at("path").get_to(extension.path);
}


/// Convert (ptr, len) tuple to a string
static std::string record_to_string(std::tuple<at::DataPtr, size_t> data) {
    return std::string(
        static_cast<char*>(std::get<0>(data).get()),
        std::get<1>(data)
    );
}


/// Check if a library is already loaded. To handle multiple platforms, this
/// does fuzzy matching on the file name; assuming that the name of the library
/// is the same across platforms.
static bool library_already_loaded(
    const std::vector<std::string>& loaded_libraries,
    const std::string& name
) {
    for (const auto& library: loaded_libraries) {
        auto filename = std::filesystem::path(library).filename().string();
        if (filename.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}


/// Load a shared library (either TorchScript extension or dependency of
/// extension) in the process
static void load_library(
    const Library& library,
    c10::optional<std::string> extensions_directory,
    bool is_dependency
) {
    auto candidates = std::vector<std::string>();
    if (library.path[0] == '/') {
        candidates.push_back(library.path);
    }

    if (extensions_directory) {
        candidates.push_back(extensions_directory.value() + "/" + library.path);
    }

    auto loaded = details::load_library(library.name, candidates);

    if (!loaded) {
        std::ostringstream oss;
        oss << "failed to load ";
        if (is_dependency) {
            oss << "extension dependency ";
        } else {
            oss << "TorchScript extension ";
        }
        oss << library.name << ". We tried the following:\n";
        for (const auto& candidate: candidates) {
            oss << " - " << candidate << "\n";
        }
        oss << " - loading " << library.name << " directly by name\n";

        if (getenv("METATOMIC_DEBUG_EXTENSIONS_LOADING") == nullptr) {
            oss << "You can set `METATOMIC_DEBUG_EXTENSIONS_LOADING=1` ";
            oss << "in your environemnt for more information\n";
        }

        TORCH_WARN(oss.str());
    }
}

void metatomic_torch::load_model_extensions(
    std::string path,
    c10::optional<std::string> extensions_directory
) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatomic-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatomic model"
        );
    }

    auto debug = getenv("METATOMIC_DEBUG_EXTENSIONS_LOADING") != nullptr;
    auto loaded_libraries = metatomic_torch::details::get_loaded_libraries();

    std::vector<Library> dependencies = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions-deps")
    ));
    for (const auto& dep: dependencies) {
        if (!library_already_loaded(loaded_libraries, dep.name)) {
            load_library(dep, extensions_directory, /*is_dependency=*/true);
        } else if (debug) {
            std::cerr << dep.name << " dependency was already loaded" << std::endl;
        }
    }

    std::vector<Library> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));
    for (const auto& ext: extensions) {
        if (ext.name == "metatensor_torch") {
            continue;
        }

        if (!library_already_loaded(loaded_libraries, ext.name)) {
            load_library(ext, extensions_directory, /*is_dependency=*/false);
        } else if (debug) {
            std::cerr << ext.name << " extension was already loaded" << std::endl;
        }
    }
}

ModelMetadata metatomic_torch::read_model_metadata(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);
    if (!reader.hasRecord("extra/model-metadata")) {
        C10_THROW_ERROR(ValueError,
            "could not find model metadata in file at '" + path +
            "', did you export your model with metatensor-torch >=0.5.4?"
        );
    }

    return ModelMetadataHolder::from_json(
        record_to_string(reader.getRecord("extra/model-metadata"))
    );
}

void metatomic_torch::check_atomistic_model(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatomic-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatomic model"
        );
    }

    auto recorded_mta_version = Version(record_to_string(
        reader.getRecord("extra/metatomic-version")
    ));
    auto current_mta_version = Version(metatomic_torch::version());

    if (!current_mta_version.is_compatible(recorded_mta_version)) {
        TORCH_WARN(
            "Current metatomic version (", current_mta_version.string, ") ",
            "is not compatible with the version (", recorded_mta_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    // Check that the extensions loaded while the model was exported are also
    // loaded now. Since the model can be exported from a different machine, or
    // the extensions might change how they organize code, we only try to do
    // fuzzy matching on the file name, and warn if we can not find a match.
    std::vector<Library> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));

    auto loaded_libraries = metatomic_torch::details::get_loaded_libraries();

    for (const auto& extension: extensions) {
        if (!library_already_loaded(loaded_libraries, extension.name)) {
            TORCH_WARN(
                "The model at '", path, "' was exported with extension '",
                extension.name, "' loaded (from '", extension.path, "'), ",
                "but it does not seem to be currently loaded; proceed at your own risk."
            );
        }
    }
}

metatensor_torch::Module metatomic_torch::load_atomistic_model(
    std::string path,
    c10::optional<std::string> extensions_directory
) {
    load_model_extensions(path, extensions_directory);
    check_atomistic_model(path);

    torch::jit::Module model;
    try {
        model = torch::jit::load(path);
    } catch (const std::exception& e) {
        auto error_str = std::string(e.what());
        if (
            (error_str.find("Unknown type name '__torch__.torch.classes.") != std::string::npos)
            || (error_str.find("Unknown builtin op") != std::string::npos)
        ) {
            std::string extra;
            if (!extensions_directory) {
                extra = "\nMake sure to provide the path to the extensions directory "
                        "if your extensions are not installed system-wide.";
            } else {
                extra = "\nMake sure that all extensions are available in the "
                        "extensions directory you provided.";
            }

            throw std::runtime_error(
                "failed to load the model at '" + path + "': " + error_str + "\n"
                "This is likely due to missing TorchScript extensions." + extra
            );
        } else {
            throw;
        }
    }

    return metatensor_torch::Module(model);
}
