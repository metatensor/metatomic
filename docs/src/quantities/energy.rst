.. _energy-quantity:

Energy
^^^^^^

The potential energy is associated with the ``"energy"`` or
``"energy/<variant>"`` name (see :ref:`quantity-variants`), and must have the
following metadata:

.. list-table:: Metadata for ``"energy"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the keys must have a single dimension named ``"_"``, with a single entry
      set to ``0``. The ``"energy"`` quantity is always represented as a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - the samples should be named ``["system", "atom"]`` for per-atom quantities;
      or ``["system"]`` for per-system quantities.

      ``"system"`` must range from 0 to the number of systems given as input
      to the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the ``"energy"`` quantity must not have any components

  * - properties
    - ``"energy"``
    - the ``"energy"`` quantity must have a single property dimension named
      ``"energy"``, with a single entry set at ``0``.

The following simulation engines can use the ``"energy"`` quantity as output:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-eon
    :link-type: ref

    |eon-logo|
    |eon-logo-dark|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-gromacs
    :link-type: ref

    |gromacs-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ipi
    :link-type: ref

    |ipi-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-lammps
    :link-type: ref

    |lammps-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-torch-sim
    :link-type: ref

    |torch-sim-logo|


.. _energy-quantity-gradients:

Gradients of the ``"energy"`` quantity
--------------------------------------

Most of the time, when writing an atomistic model compatible with metatomic,
gradients will be handled implicitly and computed by the simulation engine using
a backward pass. Additionally, it is possible for the model to support explicit,
forward mode gradients

The following gradients can be defined and requested with
``explicit_gradients``:

- **"positions"** (:math:`r_j`) gradients will contain the negative of the
  forces :math:`F_j`.

  .. math::

      \frac{\partial E}{\partial r_j} = -F_j

.. list-table:: Metadata for positions gradients of the ``"energy"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - samples
    - ``["sample", "system", "atom"]``
    - ``"sample"`` indicates which of the values samples we are taking the
      gradient of, and ``("system", "atom")`` indicates which of the atom's
      positions we are taking the gradients with respect to; i.e. :math:`j` in
      the equation above.

  * - components
    - ``"xyz"``
    - there must be a single component named ``"xyz"`` with values 0, 1, 2;
      indicating the direction of the displacement of the atom in the gradient
      samples.

- **"strain"** (:math:`\epsilon`) gradients will contain the stress
  :math:`\sigma` acting on the system, multiplied by the volume :math:`V`
  (sometimes also called the *virial* of this system)

  .. math::

    \frac{\partial E}{\partial \epsilon} = V \sigma

.. list-table:: Metadata for strain gradients of the ``"energy"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - **samples**
    - ``"sample"``
    - There is a single gradient sample dimension, ``"sample"`` indicating which
      of the values samples we are taking the gradient of.

  * - **components**
    - ``["xyz_1", "xyz_2"]``
    - Both ``"xyz_1"`` and ``"xyz_2"`` have values ``[0, 1, 2]``, and correspond
      to the two axes of the 3x3 strain matrix :math:`\epsilon`.

.. _energy-ensemble-quantity:

Energy ensemble
---------------

An ensemble of energies is associated with the ``"energy_ensemble"``  or
``"energy_ensemble/<variant>"`` name (see :ref:`quantity-variants`). Such ensembles
are sometimes used to perform uncertainty quantification, using multiple
prediction to estimate an error on the mean prediction.

Energy ensembles must have the following metadata:

.. list-table:: Metadata for ``"energy_ensemble"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - samples
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - components
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - properties
    - ``"energy"``
    - the ``"energy_ensemble"`` quantity must have a single property dimension
      named ``"energy"``, with entries ranging from 0 to the number of members
      of the ensemble minus one.

The following simulation engines can use the ``"energy_ensemble"`` quantity as
output:

.. grid:: 1 1 1 1

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ipi
    :link-type: ref

    |ipi-logo|

Energy ensemble gradients
-------------------------

The gradient metadata for energy ensemble is the same as for the ``energy``
output (see :ref:`energy-quantity-gradients`).

.. _energy-uncertainty-quantity:

Energy uncertainty
------------------

The uncertainty on the ``"energy"`` quantity is associated with the
``"energy_uncertainty"``  or ``"energy_uncertainty/<variant>"`` name (see
:ref:`quantity-variants`). This corresponds to the expected standard deviation of
the predictions when compared to the ground truth.

The ``"energy_uncertainty"`` quantity must have the following metadata:

.. list-table:: Metadata for ``"energy_uncertainty"``
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - samples
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - components
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`

  * - properties
    - same as :ref:`energy-quantity`
    - same as :ref:`energy-quantity`


The following simulation engines can use the ``"energy_uncertainty"`` quantity
as output to automatically warn users about high-uncertainty predictions:

.. grid:: 1 3 3 3

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ase
    :link-type: ref

    |ase-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-ipi
    :link-type: ref

    |ipi-logo|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-eon
    :link-type: ref

    |eon-logo|
    |eon-logo-dark|

  .. grid-item-card::
    :text-align: center
    :padding: 1
    :link: engine-lammps
    :link-type: ref

    |lammps-logo|
