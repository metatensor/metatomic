// .. _cxx-tutorial-plugin:
//
// Creating a plugin
// =================
//
// The :ref:`previous tutorial <cxx-tutorial-model>` implemented a custom model
// and constructed it directly. A plugin lets simulation engines load models
// at runtime, without compiling the model implementation into each engine.
//
// Following the :ref:`C plugin tutorial <c-tutorial-plugin>`, we will define
// how to load a model, register the plugin, and load a model through metatomic.
// We use the existing C++ Lennard-Jones implementation in
// ``metatomic-core/lj-plugin/lennard_jones.hpp`` and ``lennard_jones.cpp``.
// This implementation supports system and per-atom energies; the previous
// tutorial presented a simpler version supporting only system energies.

#include <cassert>
#include <iostream>
#include <optional>
#include <string>

#include <metatomic.hpp>

// %%
//
// Loading a model
// ----------------
//
// A C++ plugin provides a loading function with two arguments: ``load_from``
// identifies the model (for example, a model name or file path), and
// ``options`` contains string keys and string values configuring the model.
// The function returns a ``std::unique_ptr<metatomic::BaseModel>``.
//
// Here is the loading function from the Lennard-Jones implementation:
//
// .. literalinclude:: /../../metatomic-core/lj-plugin/lennard_jones.cpp
//    :language: cpp
//    :start-at: std::unique_ptr<metatomic::BaseModel> load_model(
//    :end-at: return std::make_unique<LennardJones>(std::move(lj_options));
//    :append: }
//
// Returning ``nullptr`` means the plugin does not recognize ``load_from``.
// Metatomic can then try another registered plugin. This corresponds to
// returning ``MTA_UNSUPPORTED_MODEL_ERROR`` from a C plugin's callback.
//
// Once the model is recognized, ``LennardJonesOptions`` parses and validates
// its parameters, and the loader constructs the model. Invalid parameters
// raise an exception, which is reported to the caller instead of continuing
// the search for another plugin.

// %%
//
// Registering the plugin
// ----------------------
//
// The same implementation registers its loader with
// ``MTA_REGISTER_CXX_PLUGIN``:
//
// .. literalinclude:: /../../metatomic-core/lj-plugin/lennard_jones.cpp
//    :language: cpp
//    :start-at: MTA_REGISTER_CXX_PLUGIN(
//    :end-at: MTA_REGISTER_CXX_PLUGIN(
//
// The first argument is the plugin's name and the second is its loading
// function. Use this macro once per plugin shared library.
//
// The macro builds the C API's ``mta_plugin_t`` for us. It converts JSON options
// into a ``std::map<std::string, std::string>``, translates exceptions into
// status codes, and wraps the returned C++ model in an owning ``mta_model_t``.
// We do not need to write the C callbacks manually.

// %%
//
// Building the plugin
// -------------------
//
// To distribute the model as a plugin, compile ``lennard_jones.cpp`` into a
// shared library linked to metatomic. For example, with CMake, in a directory
// containing ``lennard_jones.hpp`` and ``lennard_jones.cpp``:
//
// .. code-block:: cmake
//
//    cmake_minimum_required(VERSION 3.22)
//    project(lennard-jones-plugin LANGUAGES CXX)
//    find_package(metatomic CONFIG REQUIRED)
//
//    add_library(lennard-jones-plugin SHARED lennard_jones.cpp)
//    target_link_libraries(lennard-jones-plugin PRIVATE metatomic)
//    target_compile_features(lennard-jones-plugin PRIVATE cxx_std_17)
//
// Configure with ``cmake -S . -B build -DCMAKE_PREFIX_PATH=<install-prefix>``
// and build with ``cmake --build build``. The prefix must contain an installed
// metatomic C++ library and its dependencies. An engine can then register the
// shared library with ``metatomic::load_plugin(path)`` before loading a model.
//
// For this runnable tutorial, we instead compile ``4-plugin.cpp`` together
// with the existing ``metatomic-core/lj-plugin/lennard_jones.cpp`` into one
// executable. We can then register the plugin from the current executable,
// as in the C tutorial, without needing a separate shared library.
//
// .. note::
//
//    On Linux, link this executable with ``-Wl,--export-dynamic`` so metatomic
//    can find the plugin registration symbols, or set the ``ENABLE_EXPORTS``
//    target property in CMake. The repository's example test target sets this
//    property automatically.

int main() {
    // The C API accepts nullptr to load the plugin from the current executable.
    auto status = mta_load_plugin(nullptr);
    if (status != MTA_SUCCESS) {
        const char* message = nullptr;
        mta_last_error(&message, nullptr, nullptr);
        std::cerr << "Failed to register plugin: " << message << "\n";
        return 1;
    }

// %%
//
// Using the plugin
// ----------------
//
// The engine passes options to ``metatomic::load_model`` as a JSON object
// whose values are strings. ``MTA_REGISTER_CXX_PLUGIN`` parses the JSON before
// invoking our loading function; ``LennardJonesOptions`` then converts the
// numeric strings to the types needed by the model.
//
// We use the same parameters as in the previous tutorial. Leaving the plugin
// name unspecified lets metatomic find a plugin that recognizes the model.

    auto options_json = std::string(R"({
        "sigma": "1.0",
        "epsilon": "1.0",
        "cutoff": "3.0",
        "atomic_type": "1",
        "length_unit": "Angstrom",
        "energy_unit": "eV"
    })");

    auto model = metatomic::load_model("metatomic-lj-model", options_json);

// %%
//
// The result is an ``ExternalModel``, which implements ``BaseModel``. We can
// query its metadata and capabilities, or pass it to
// ``metatomic::execute_model`` just like the model in the previous tutorial.
// Its destructor calls the underlying model's ``unload`` callback, freeing
// the C++ object automatically.

    auto metadata = model.metadata();
    auto capabilities = model.capabilities();
    assert(metadata.name() == "Lennard-Jones test model");
    assert(capabilities.interaction_range() == 3.0);
    assert(capabilities.length_unit() == "Angstrom");
    std::cout << "Loaded model: " << metadata.name() << "\n";

// %%
//
// We can also name the plugin explicitly, restricting the search to this
// plugin. Notice that the plugin name and model identifier are distinct:
// ``metatomic-lj-plugin`` is the registered plugin, while
// ``metatomic-lj-model`` is the identifier recognized by its loader.

    auto another_model = metatomic::load_model(
        "metatomic-lj-model", options_json, "metatomic-lj-plugin"
    );
    assert(another_model.metadata().name() == metadata.name());
    return 0;
}
