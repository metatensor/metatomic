.. _standard-quantities:

Standard quantities
===================

In order for multiple simulation engines to be able use arbitrary metatomic
models to compute atomic properties, we need all the models to use the same
metadata when handling the same quantity. If your model returns one of the
quantity defined in this documentation as output; or use them as input; then it
must follow the metadata structure described here.

If you need other quantities as inputs or outputs, you should use custom
quantity with a name containing ``::``, such as ``my_code::my_quantity``. For
such custom quantity, you are free to use any relevant metadata structure, but
if multiple people are using the same quantity, they are encouraged to come
together, define the metadata schema they need and add a new section to these
pages.

.. toctree::
  :maxdepth: 1
  :hidden:

  energy
  non_conservative
  mass
  position
  momentum
  velocity
  charge
  heat_flux
  spin_multiplicity
  feature
  variants

Variants
^^^^^^^^

Models can define variants of any quantity, for example to provide the same
output at different levels of theory in a single model. For more information on
variants, please refer to :ref:`the corresponding documentation
<quantity-variants>`.


Physical quantities
^^^^^^^^^^^^^^^^^^^

The first set of standardized quantities for metatomic models are physical
quantities, i.e. quantities with a well-defined physical meaning.

.. grid:: 1 2 2 2

    .. grid-item-card:: Energy
      :link: energy-quantity
      :link-type: ref

      .. image:: /../static/images/energy-quantity.png

      The potential energy associated with a given system configuration. This
      can be used to run molecular simulations with on machine learning based
      interatomic potentials.

    .. grid-item-card:: Energy ensemble
      :link: energy-ensemble-quantity
      :link-type: ref

      .. image:: /../static/images/energy-ensemble-quantity.png

      An ensemble of multiple potential energy predictions, generated
      when running multiple models simultaneously.

    .. grid-item-card:: Energy uncertainty
      :link: energy-uncertainty-quantity
      :link-type: ref

      .. image:: /../static/images/energy-uncertainty-quantity.png

      The uncertainty on the potential energies, useful to quantify the confidence of
      the model.

    .. grid-item-card:: Non-conservative force
      :link: non-conservative-force-quantity
      :link-type: ref

      .. image:: /../static/images/nc-force-quantity.png

      Forces directly predicted by the model, not derived from the potential
      energy.

    .. grid-item-card:: Non-conservative stress
      :link: non-conservative-stress-quantity
      :link-type: ref

      .. image:: /../static/images/nc-stress-quantity.png

      Stress directly predicted by the model, not derived from the potential
      energy.

    .. grid-item-card:: Mass
      :link: mass-quantity
      :link-type: ref

      .. image:: /../static/images/mass-quantity.png

      Atomic masses

    .. grid-item-card:: Position
      :link: position-quantity
      :link-type: ref

      .. image:: /../static/images/position-quantity.png

      Atomic positions predicted by the model, to be used in ML-driven simulations.

    .. grid-item-card:: Momentum
      :link: momentum-quantity
      :link-type: ref

      .. image:: /../static/images/momentum-quantity.png

      Atomic momenta, i.e. :math:`m \times \vec v`

    .. grid-item-card:: Velocity
      :link: velocity-quantity
      :link-type: ref

      .. image:: /../static/images/velocity-quantity.png

      Atomic velocities, i.e. :math:`\vec p / m`

    .. grid-item-card:: Charges
      :link: charge-quantity
      :link-type: ref

      .. image:: /../static/images/charge-quantity.png

      Atomic charges, e.g. formal or partial charges on atoms

    .. grid-item-card:: Heat flux
      :link: heat-flux-quantity
      :link-type: ref

      .. image:: /../static/images/heat-flux-quantity.png

      Heat flux, i.e. the amount of energy transferred per unit time, i.e.
      :math:`\sum_i E_i \times \vec v_i`

    .. grid-item-card:: Spin multiplicity
      :link: spin-multiplicity-quantity
      :link-type: ref

      .. image:: /../static/images/spin-multiplicity-quantity.png

      The spin multiplicity :math:`(2S + 1)` of the system, with :math:`S` the
      number of unpaired electrons.

Machine learning quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next set of standardized quantities in metatomic models are specific to
machine learning and related tools.

.. grid:: 1 2 2 2

    .. grid-item-card:: Features
      :link: feature-quantity
      :link-type: ref

      .. image:: /../static/images/feature-quantity.png

      Features are numerical vectors representing a given structure or atomic
      environment in an abstract n-dimensional space.
