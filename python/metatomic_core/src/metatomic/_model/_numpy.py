import abc
from typing import Optional

from metatensor import Labels, TensorMap

from .._quantity import Quantity
from .._system import PairListOptions, System
from ._base import BaseModel


class NumpyModel(BaseModel):
    """
    Base class for NumPy-based atomistic models.

    This class provides a convenient way to implement models that use NumPy for their
    computations, by automatically setting the :py:attr:`System.arrays_backend` to
    ``"numpy"`` before calling the :py:meth:`execute_inner` method, ensuring that the
    system's data will be accessible as NumPy arrays.
    """

    def __init_subclass__(cls):
        """
        Change the ``execute_inner`` function of the subclass to make sure the system's
        data will be accessible as numpy arrays.
        """
        super().__init_subclass__()

        execute_inner_impl = cls.execute_inner

        def execute_inner_wrap(
            self,
            systems: list[System],
            selected_atoms: Optional[Labels],
            requested_outputs: list[Quantity],
        ) -> list[TensorMap]:
            for system in systems:
                system.arrays_backend = "numpy"

            return execute_inner_impl(self, systems, selected_atoms, requested_outputs)

        cls.execute_inner = execute_inner_wrap

    @abc.abstractmethod
    def execute_inner(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels],
        requested_outputs: list[Quantity],
    ) -> list[TensorMap]:
        """
        Run the model and compute the requested outputs.

        This method should not be called directly. It is intended to be used through
        :py:func:`execute_model`, which handles unit conversion and can check inputs and
        outputs for consistency.

        :param systems: systems to run the model on, with their
            :py:attr:`System.arrays_backend` set to ``"numpy"``
        :param selected_atoms: optional selection of atoms to compute outputs for, or
            ``None`` to use all atoms
        :param requested_outputs: outputs the model should compute
        :return: the computed outputs, one :py:class:`metatensor.TensorMap` per
            requested output, in the same order
        """

    def requested_pair_lists(
        self,
        requested_outputs: list[Quantity],
    ) -> list[PairListOptions]:
        """
        List the pair lists (neighbor lists) this model needs as input.

        This method should be overridden by subclasses to specify the pair lists that
        the model requires. By default, it returns an empty list, indicating that no
        pair lists are needed.
        """
        return []

    def requested_inputs(
        self,
        requested_outputs: list[Quantity],
    ) -> list[Quantity]:
        """
        List the additional per-system inputs this model needs.

        This method should be overridden by subclasses to specify any additional
        per-system inputs that the model requires. By default, it returns an empty list,
        indicating that no additional inputs are needed.
        """
        return []
