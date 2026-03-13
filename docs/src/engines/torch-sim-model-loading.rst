.. _torchsim-model-loading:

Loading models
==============

``MetatomicModel`` accepts several input formats. Each section below
shows one loading pattern.

From a saved ``.pt`` file
-------------------------

The most common case. Pass the path to a TorchScript-exported metatomic
model:

.. code-block:: python

   from metatomic_torchsim import MetatomicModel

   model = MetatomicModel("path/to/model.pt", device="cpu")

The file must exist and contain a valid ``AtomisticModel``. A
``ValueError`` is raised if the path does not exist.

From a metatrain checkpoint
---------------------------

Pass a ``.ckpt`` path to load a metatrain checkpoint directly. This
requires the ``metatrain`` package:

.. code-block:: python

   model = MetatomicModel("path/to/checkpoint.ckpt")

The checkpoint is exported to an ``AtomisticModel`` internally.

PET-MAD shortcut
----------------

The string ``"pet-mad"`` downloads and loads the PET-MAD universal
model:

.. code-block:: python

   model = MetatomicModel("pet-mad")

This also requires ``metatrain`` to be installed. The model weights are
fetched from HuggingFace on first use.

From a Python AtomisticModel
-----------------------------

If you already have an ``AtomisticModel`` instance (for example, built
programmatically):

.. code-block:: python

   from metatomic.torch import AtomisticModel

   atomistic_model = build_my_model()  # returns AtomisticModel
   model = MetatomicModel(atomistic_model, device="cuda")

From a TorchScript RecursiveScriptModule
-----------------------------------------

If you have a scripted model loaded via ``torch.jit.load``:

.. code-block:: python

   import torch

   scripted = torch.jit.load("model.pt")
   model = MetatomicModel(scripted, device="cpu")

The script module must have ``original_name == "AtomisticModel"``.
Otherwise a ``TypeError`` is raised.

Selecting a device
------------------

By default, ``MetatomicModel`` picks the best device from the model's
``supported_devices``. Override with the ``device`` parameter:

.. code-block:: python

   model = MetatomicModel("model.pt", device="cuda:0")

Extensions directory
--------------------

Some models require compiled TorchScript extensions. Point to their
location with ``extensions_directory``:

.. code-block:: python

   model = MetatomicModel(
       "model.pt",
       extensions_directory="path/to/extensions/",
   )
