Running TorchScript models with metatomic-core
==============================================

Models exported with :py:class:`metatomic.torch.AtomisticModel` (typically
``.pt`` files) can be executed with ``metatomic::execute_model`` or
:c:func:`mta_execute_model`, by any engine using the metatomic C or C++ API,
with :cpp:class:`metatomic_torch::LegacyModel`. This is the C++ counterpart of
the Python ``metatomic.LegacyModel``, and behaves in the same way.

.. code-block:: cpp

    #include <metatomic.hpp>
    #include <metatomic/torch.hpp>

    auto model = metatomic_torch::LegacyModel("model.pt", "extensions/");

    auto energy = metatomic::Quantity::builder()
        .name("energy")
        .unit("eV")
        .sample_kind(metatomic::SampleKind::System)
        .gradients({metatomic::Gradients::Positions})
        .build();

    // the engine creates the systems, and adds the pairs requested by
    // `model.requested_pair_lists()` to them
    auto outputs = metatomic::execute_model(
        model, systems, std::nullopt, {energy}, /*check_consistency=*/ false
    );

The systems are not copied: the model sees new :cpp:type:`metatomic_torch::System`
sharing the types, positions, cell, pairs and custom data of the engine's
systems. Gradients with respect to ``"positions"`` and ``"strain"`` of
per-system energies and custom quantities are computed with autograd, and
explicit gradients declared by the model are used as-is.

.. doxygenclass:: metatomic_torch::LegacyModel
    :members:

metatomic-torch plugin
----------------------

Engines that do not link to libtorch can load these models through the
``metatomic-torch`` plugin, a shared library installed in
``<prefix>/libexec/metatomic/metatomic-torch-plugin.so`` (in ``<prefix>/bin``
on Windows). The Python package gives its path as
``metatomic_torch.utils.plugin_path``:

.. code-block:: bash

    python -c "import metatomic_torch; print(metatomic_torch.utils.plugin_path)"

Load this path with :c:func:`mta_load_plugin`, then load the exported model with
:c:func:`mta_load_model`. The plugin handles all files exported with
:py:class:`metatomic.torch.AtomisticModel`, so the plugin name can be omitted.
The only option is ``extensions_directory``, the directory containing the
TorchScript extensions required by the model.

.. code-block:: cpp

    metatomic::load_plugin(plugin_path);
    auto model = metatomic::load_model(
        "model.pt", R"({"extensions_directory": "extensions/"})", "metatomic-torch"
    );

The plugin is built when the ``METATOMIC_TORCH_PLUGIN`` CMake option is ``ON``
(the default).
