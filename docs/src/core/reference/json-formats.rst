.. _core-json-formats:

JSON data formats
=================

Some metatomic data structures are exchanged across the C API as JSON-encoded
strings rather than dedicated C types. This page documents the exact JSON
representation of each such structure, so that engines and models written in any
language can produce and consume them.

.. _core-json-pair-options:

Pair list options
-----------------

The JSON representation of a requested pair list (also known as a neighbor
list). This is used for example by :c:func:`mta_system_add_pairs`,
:c:func:`mta_system_get_pairs` and :c:func:`mta_system_known_pairs`.

.. code-block:: json

    {
        "type": "metatomic_pair_options",
        "cutoff": "0x400c000000000000",
        "full_list": false,
        "strict": false,
        "requestors": ["my-model"]
    }

``type``
    Must be the string ``"metatomic_pair_options"``.

``cutoff``
    Cutoff radius for the pair list in the length unit of the model. Must be a
    positive finite number.

    It is stored as a string containing the hexadecimal representation of the
    64-bit integer with the same bit pattern as the ``cutoff`` floating-point
    value (i.e. reinterpreting the ``double`` as a ``uint64_t``).

``full_list``
    Boolean. If ``true``, the list is a full list containing both ``i -> j``
    and ``j -> i`` for each pair, if ``false``, it is a half list containing
    only ``i -> j``.

``strict``
    Boolean. If ``true``, the list is guaranteed to contain only atoms within
    the cutoff, if ``false``, it may also include some pairs slightly beyond the
    cutoff.

``requestors``
    Optional array of strings identifying who requested this pair list. May be
    omitted, in which case it is treated as an empty list.


.. _core-json-quantity:

Quantities
----------

The JSON representation of a physical quantity, used to represent custom models
inputs and outputs. This is used for example in
:c:member:`mta_model_t.requested_inputs` and
:c:member:`mta_model_t.supported_outputs`.

.. code-block:: json

    {
        "type": "metatomic_quantity",
        "name": "energy",
        "unit": "eV",
        "sample_kind": "system"
        "gradients": ["positions"]
        "description": "Potential energy of the system",
    }

``type``
    Must be the string ``"metatomic_quantity"``.

``name``
    Name of the quantity, this this can be a standard name from the list of
    :ref:`standard-quantities`, or a custom name of the form
    ``<namespace>::<name>[/<variant>]``

``unit``
    Unit of the quantity.

``gradients``
    Array of strings identifying the gradients for this quantity. This can be an
    empty array if the quantity has no gradients. Valid values for the gradients
    are ``"positions"``, and ``"strain"``.

``sample_kind``
    Kind of sample for which this quantity is defined. This can be one of the
    following: ``"atom"``, ``"system"`` or ``"atom_pair"``.


.. _core-json-model-metadata:

Model metadata
--------------

The JSON representation of a model's metadata. This is used for example by
:c:member:`mta_model_t.metadata`.

.. code-block:: json

    {
        "type": "metatomic_model_metadata",
        "name": "MyCoolModel v1.2",
        "authors": ["Alice Smith", "Bob Johnson <bobj@example.com>"],
        "description": "A machine learning potential for water",
        "references": {
            "model": ["doi:10.1234/model-paper"],
            "architecture": ["doi:10.1234/arch-paper"],
            "implementation": ["https://github.com/example/mycoolmodel"]
        },
        "extra": {
            "training_set": "QM9",
            "cutoff": "4.5"
        }
    }

``type``
    Must be the string ``"metatomic_model_metadata"``.

``name``
    Name of the model, e.g. ``"MyCoolModel v1.2"``.

``authors``
    Array of strings identifying the authors of the model. Each string can be a
    name or a name with an email address, e.g. ``"Alice Smith"`` or
    ``"Bob Johnson <bobj@example.com>"``.

``description``
    A free-text description of the model.

``references``
    An object with three keys, each containing an array of strings (DOIs, URLs,
    or any other format):

    ``model``
        References about the model as a whole, e.g. a paper describing the model
        or a website presenting it.

    ``architecture``
        References about the architecture of the model, e.g. papers describing
        the mathematical form of the model.

    ``implementation``
        References about the implementation of the model, e.g. a link to the
        source code repository or a paper describing the software.

``extra``
    An object with string values, providing any additional key-value pairs the
    model author wishes to include. This can be used for any purpose.

.. _core-json-model-capabilities:

Model capabilities
------------------

The JSON representation of a model's capabilities, describing which outputs it
provides, which atomic types it supports, and other constraints. This is used
for example by :c:member:`mta_model_t.capabilities`.

.. code-block:: json

    {
        "type": "metatomic_model_capabilities",
        "outputs": [
            {
                "type": "metatomic_quantity",
                "name": "energy",
                "unit": "eV",
                "sample_kind": "system",
                "gradients": ["positions"],
                "description": "Potential energy of the system"
            },
            {
                "type": "metatomic_quantity",
                "name": "energy/pbe0",
                "unit": "eV",
                "sample_kind": "system",
                "gradients": ["positions", "strain"],
                "description": "Potential energy of the system"
            },
        ],
        "atomic_types": [1, 6, 8],
        "interaction_range": 5.0,
        "length_unit": "angstrom",
        "supported_devices": ["cpu", "cuda"],
        "dtype": "float32"
    }

``type``
    Must be the string ``"metatomic_model_capabilities"``.

``outputs``
    Array of :ref:`quantity objects <core-json-quantity>` describing the
    outputs this model can provide.

``atomic_types``
    Array of integers listing the atomic types this model supports. The meaning
    of these integers is up to the model, and is not required to be the atomic
    numbers.

``interaction_range``
    The interaction range of the model in the length unit of the model. This is
    the maximum distance between two atoms for which the model's output can
    depend on their relative position. Must be a non-negative number.

``length_unit``
    String identifying the length unit used by the model, e.g. ``"angstrom"`` or
    ``"nanometer"``. This must be a valid :ref:`unit expression <units>` with
    dimensions compatible with length.

``supported_devices``
    Array of strings listing the devices on which the model can run. Valid
    values are ``"cpu"``, ``"cuda"``, ``"rocm"``, and ``"metal"``.

``dtype``
    The data type of the model, used for all inputs and outputs. Must be either
    ``"float32"`` or ``"float64"``. The model is free to use different data
    types for internal computations, but all inputs and outputs must be in this
    data type.
