Atomic Simulation Environment (ASE) integration
===============================================

The integration of metatomic with the Atomic Simulation Environment (ASE) was
moved into it's own package, ``metatomic-ase``, which is available on PyPI. The
documentation for this package can be found in the :ref:`corresponding section
of the documentation <ase-integration-api>`.

Both calculators classes are re-exported from the
``metatomic.torch.ase_calculator`` module for baclwards compatibility, but users
are encouraged to import them from the ``metatomic_ase`` package instead. The
old import paths will be removed in a future release.
