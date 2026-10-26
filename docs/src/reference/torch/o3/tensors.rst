Cartesian and spherical tensors
================================

:py:func:`~metatomic.torch.o3.cartesian_to_spherical` converts a Cartesian
tensor of arbitrary rank, stored on a product basis with one ``xyz_1``,
``xyz_2``, ... component axis per Cartesian index, into a set of blocks on a
basis of real spherical harmonics (with a single ``o3_mu`` component axis, and
``o3_lambda`` / ``o3_sigma`` keys). This is done by coupling the Cartesian
components two by two with Clebsch-Gordan coefficients, starting from the last
ones. For tensors of rank 3 and above, there are multiple ways to get the same
``o3_lambda``, and the coupling history is stored in additional ``l_{i}`` and
``k_{i}`` keys. :py:func:`~metatomic.torch.o3.spherical_to_cartesian` performs
the inverse transformation. These functions can be used from TorchScript, in
which case the Clebsch-Gordan coefficients have to be computed in advance with
:py:func:`~metatomic.torch.o3.calculate_cg_coefficients`.

--------------------------------------------------------------------------------

.. autofunction:: metatomic.torch.o3.cartesian_to_spherical

.. autofunction:: metatomic.torch.o3.spherical_to_cartesian

.. autofunction:: metatomic.torch.o3.calculate_cg_coefficients

Coupled and uncoupled spherical tensors
---------------------------------------

A block with several spherical component axes ``o3_mu_1``, ``o3_mu_2``, ...
(with the corresponding ``o3_lambda_1`` / ``o3_sigma_1``, ``o3_lambda_2`` /
``o3_sigma_2``, ... keys) transforms like a product of spherical harmonics. This
is the uncoupled form, used for example for the blocks of a Hamiltonian
expressed on a basis of atomic orbitals.
:py:func:`~metatomic.torch.o3.uncoupled_to_coupled` uses Clebsch-Gordan
coefficients to transform such a tensor into the coupled form, where each block
has a single ``o3_mu`` component axis and transforms like a single spherical
harmonic, with degree and parity given by the ``o3_lambda`` and ``o3_sigma``
keys. The degrees and parities of the uncoupled components are kept in the
``l_1`` / ``s_1``, ``l_2`` / ``s_2``, ... keys of the coupled tensor, and
:py:func:`~metatomic.torch.o3.coupled_to_uncoupled` uses them to perform the
inverse transformation. As for the conversion between Cartesian and spherical
tensors, the Clebsch-Gordan coefficients have to be computed in advance with
:py:func:`~metatomic.torch.o3.calculate_cg_coefficients` to use these functions
from TorchScript.

--------------------------------------------------------------------------------

.. autofunction:: metatomic.torch.o3.uncoupled_to_coupled

.. autofunction:: metatomic.torch.o3.coupled_to_uncoupled
