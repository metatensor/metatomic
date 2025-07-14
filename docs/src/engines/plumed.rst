.. _engine-plumed:

PLUMED
======


.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://www.plumed.org/
     - In the official (development) version

The `metatomic` interface for `PLUMED <https://www.plumed.org/>`_ allows using
an exported :py:class:`AtomisticModel` to compute arbitrary functions of the
atomic coordinates, using them as collective variables to perform advanced
sampling such as metadynamics. Below we also provide a minimal example of the
implementation of a minimalist, model that is compatible with this interface.
See also `this recipe
<https://atomistic-cookbook.org/examples/metatomic-plumed/metatomic-plumed.html>`_
for more realistic, complex demonstrations.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

See the official `installation instructions`_ in the documentation of PLUMED.

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The model must provide a :ref:`features <features-output>` output, and it is
important that this output has a fixed size, and that the size can be determined
by executing the model with an empty system (as this is how PLUMED determines
internally the size of a CV). A minimal example of a model that computes the
distance between two atoms is given below. Note how the capabilities and outputs
of the model are defined to create an :py:class:`AtomisticModel` from a bare
``torch.nn.Module``, before exporting it as a torchscript file.

.. literalinclude:: plumed-model.py

How to use the model in PLUMED
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the official `syntax reference`_ in the PLUMED documentation.
An example of a PLUMED input to load the model above could read

.. code-block::

    dist: METATOMIC ...
        MODEL=mta-distance.pt
        SPECIES1=1-416  # no need for species
        SPECIES_TO_TYPES=0  # map everything to zero
        SELECTED_ATOMS=401,402  # indices of atoms (1-based)
    ...


.. _installation instructions: https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_t_o_m_i_c_m_o_d.html
.. _syntax reference: https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_t_o_m_i_c.html
