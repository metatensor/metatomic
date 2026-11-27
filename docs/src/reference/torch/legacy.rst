Legacy API
==========

The functions and classes documented here are considered part of the legacy API
from metatomic-torch v0.1. They are still available for backward compatibility,
but should no longer be used for new code.



.. autoclass:: metatomic_torch.ModelInterface
    :members:
    :show-inheritance:

.. autoclass:: metatomic_torch.AtomisticModel
    :members:

.. autoclass:: metatomic_torch.ModelEvaluationOptions
    :members:

.. autoclass:: metatomic_torch.LegacyModelCapabilities
    :members:

.. autoclass:: metatomic_torch.ModelOutput
    :members:

.. autofunction:: metatomic_torch.read_model_metadata

.. autofunction:: metatomic_torch.load_atomistic_model

.. autofunction:: metatomic_torch.check_atomistic_model

.. autofunction:: metatomic_torch.load_model_extensions

.. autofunction:: metatomic_torch.pick_device

.. autofunction:: metatomic_torch.pick_output

Running legacy models with metatomic-core
-----------------------------------------

Models exported with :py:class:`metatomic_torch.AtomisticModel` (typically
``.pt`` files) can be used by any engine using the metatomic C or C++ API
through the ``metatomic-torch`` plugin, and executed with
``metatomic::execute_model`` or :c:func:`mta_execute_model`. The plugin behaves
like the Python ``metatomic.LegacyModel``, and engines using it do not need to
link to libtorch.

The plugin is a shared library installed in
``<prefix>/libexec/metatomic/metatomic-torch-legacy-plugin.so`` (in ``<prefix>/bin``
on Windows). The Python package gives its path as
``metatomic_torch.utils.legacy_plugin_path``:

.. code-block:: bash

    python -c "import metatomic_torch; print(metatomic_torch.utils.legacy_plugin_path)"

Load this path with :c:func:`mta_load_plugin`, then load the exported model with
:c:func:`mta_load_model`. The plugin recognizes the files exported by
:py:class:`metatomic_torch.AtomisticModel` with metatomic-torch v0.1, so
:c:func:`mta_load_model` uses it even when no plugin name is given. The plugin accepts two options:

- ``extensions_directory``: the directory containing the TorchScript extensions
  required by the model;
- ``device``: the device on which to run the model, for example ``"cpu"`` or
  ``"cuda:1"``. The systems are moved to this device, and the outputs are moved
  back to the device of the systems. Engines can then keep all their data on the
  CPU, even if the model only runs on a GPU. Without this option, the model runs
  on the device of the systems.

.. code-block:: cpp

    #include <metatomic.hpp>

    metatomic::load_plugin(legacy_plugin_path);
    auto model = metatomic::load_model(
        "model.pt", R"({"extensions_directory": "extensions/"})", "metatomic-torch"
    );

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

Unless the ``device`` option moves them, the systems are not copied: the model
sees the types, positions, cell, pairs and custom data of the engine's systems.
Gradients with respect to ``"positions"`` and ``"strain"`` of per-system
energies and custom quantities are computed with autograd, and explicit
gradients declared by the model are used as-is.

The plugin is built when the ``METATOMIC_TORCH_LEGACY_PLUGIN`` CMake option is
``ON`` (the default).
