.. _core-json-formats:

JSON data formats
=================

Some metatomic data structures are exchanged across the C API as JSON-encoded
strings rather than dedicated C types. This page documents the exact JSON
representation of each such structure, so that engines and models written in any
language can produce and consume them.

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
