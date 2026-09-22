"""
Rotate systems and tensor maps under O(3) transformations, routing rows of multi-system
tensors by their ``"system"`` sample label.

:py:class:`O3Transformation` is a :class:`torch.nn.Module` holding a batch of one or
more operations. The tensor-transformation kernels in this module are TorchScript
compatible, so a scripted model can call the module's methods inside ``forward``. The
module must be constructed eagerly in ``__init__`` (the Wigner-D build requires the
``wigners`` package); scripted code only calls the transform methods.
"""

from numbers import Integral
from typing import List, Optional

import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap

from .. import System, register_autograd_neighbors
from ._utils import copy_tensormap_info, validate_integer
from ._wigner import build_packed_wigner_matrices, wigner_matrices_for_lambda


def _spherical_parity_factor(
    ell: int,
    sigma: int,
    is_improper: bool,
) -> int:
    """Return ``sigma * (-1) ** ell`` for an improper transformation, else ``1``."""
    if torch.jit.is_scripting():
        integer_sigma = sigma
    else:
        if isinstance(sigma, bool) or not isinstance(sigma, Integral):
            raise TypeError(f"sigma must be an integer, got {type(sigma).__name__}")
        integer_sigma = int(sigma)
    if integer_sigma not in (-1, 1):
        raise ValueError(f"sigma must be either -1 or +1, got {integer_sigma}")

    if is_improper:
        return integer_sigma * int((-1) ** ell)

    return 1


class O3Transformations(torch.nn.Module):
    """
    A batch of one or more O(3) transformations, represented by ``(N, 3, 3)`` rotation
    or improper-rotation matrices.

    The module stores the matrices, an improper-operations mask, and a packed buffer of
    Wigner-D matrices. It is constructed eagerly in ``__init__`` (building the Wigner-D
    matrices requires the ``wigners`` package); the transform methods are TorchScript
    compatible and can be called from a scripted ``forward``.

    Transform methods accept an ``indices`` argument to select a sub-batch of
    operations, and a ``add_inversion`` flag to compose every selected operation with
    the inversion. Inverse transformations are available through the
    ``inverse_transform_*`` methods, which share their kernel with the forward ones and
    use transposed matrices / Wigner-D matrices.
    """

    _max_angular_momentum: int

    def __init__(
        self,
        matrices: torch.Tensor,
        max_angular_momentum: int,
    ):
        """
        :param matrices: ``(N, 3, 3)`` rotation or improper-rotation matrices
        :param max_angular_momentum: non-negative maximum angular momentum for
            which Wigner-D matrices are built
        """
        super().__init__()

        self._max_angular_momentum = validate_integer(
            "max_angular_momentum", max_angular_momentum, 0
        )

        if (
            matrices.dim() != 3
            or matrices.size(0) == 0
            or matrices.size(1) != 3
            or matrices.size(2) != 3
        ):
            raise ValueError(
                f"O3Transformations `matrices` has shape {tuple(matrices.shape)}; "
                "expected (N, 3, 3)"
            )

        identity = torch.eye(3, device=matrices.device, dtype=matrices.dtype)
        if not torch.allclose(
            matrices @ matrices.transpose(1, 2),
            identity,
            atol=1e-5,
        ):
            raise ValueError("O3Transformations `matrices` must be orthogonal")

        improper = torch.linalg.det(matrices) < 0.0
        packed_wigner = build_packed_wigner_matrices(
            matrices, self._max_angular_momentum
        )

        self.register_buffer("_matrices", matrices.clone())
        self.register_buffer("_improper", improper)
        self.register_buffer("_packed_wigner", packed_wigner)

    @property
    def matrices(self) -> torch.Tensor:
        """The ``(N, 3, 3)`` batch of rotation or improper-rotation matrices."""
        return self._matrices

    @property
    def max_angular_momentum(self) -> int:
        """The maximum angular momentum with available Wigner-D matrices."""
        return self._max_angular_momentum

    @property
    def improper_mask(self) -> torch.Tensor:
        """Boolean mask marking the improper operations in the batch."""
        return self._improper

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the transformation matrices."""
        return self._matrices.dtype

    @property
    def device(self) -> torch.device:
        """The device of the transformation matrices."""
        return self._matrices.device

    def _validate_ell_range(self, ell: int) -> int:
        """Check that ``ell`` is an integer in ``[0, max_angular_momentum]``."""
        ell = validate_integer("ell", ell, 0)

        if ell > self._max_angular_momentum:
            raise ValueError(
                f"ell={ell} exceeds max_angular_momentum={self._max_angular_momentum}."
            )

        return ell

    def wigner_D_matrices(self, ell: int) -> torch.Tensor:
        """Return the proper-part Wigner-D matrices for ``ell``.

        For improper operations, the inversion-parity factor ``sigma * (-1) ** ell`` is
        applied separately when transforming spherical values.

        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :return: ``(N, 2*ell+1, 2*ell+1)`` stack of Wigner-D matrices
        """
        ell = self._validate_ell_range(ell)
        return wigner_matrices_for_lambda(
            self._packed_wigner, self._matrices.size(0), ell
        )

    def inverse_wigner_D_matrices(self, ell: int) -> torch.Tensor:
        """Return the transposed (inverse) proper-part Wigner-D matrices.

        The inverse of a real Wigner-D matrix is its transpose. This is the accessor
        used by projection integrals that contract values against the inverse
        representation directly.

        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :return: ``(N, 2*ell+1, 2*ell+1)`` stack of transposed Wigner-D matrices
        """
        return self.wigner_D_matrices(ell).transpose(1, 2)

    def _effective_matrices(
        self,
        add_inversion: bool,
        transpose: bool,
    ) -> torch.Tensor:
        """Return the (optionally inverted, optionally transposed) matrices."""
        matrices = self._matrices
        if add_inversion:
            matrices = -matrices
        if transpose:
            matrices = matrices.transpose(1, 2)
        return matrices

    def _effective_improper(self, add_inversion: bool) -> torch.Tensor:
        """Return the (optionally flipped) improper mask."""
        if add_inversion:
            return torch.logical_not(self._improper)
        return self._improper

    def _effective_wigner(self, transpose: bool) -> List[torch.Tensor]:
        """Return the per-``ell`` Wigner-D stacks, optionally transposed."""
        n_matrices = self._matrices.size(0)
        wigner_views: List[torch.Tensor] = []
        for ell in range(self._max_angular_momentum + 1):
            view = wigner_matrices_for_lambda(self._packed_wigner, n_matrices, ell)
            if transpose:
                view = view.transpose(1, 2)
            wigner_views.append(view)
        return wigner_views

    def transform_cartesian(
        self,
        vectors: torch.Tensor,
        add_inversion: bool = False,
    ) -> torch.Tensor:
        """Apply the transformations to Cartesian vectors.

        :param vectors: ``(..., 3)`` tensor of Cartesian vectors
        :param add_inversion: compose every operation with the inversion
        :return: transformed vectors, with the input shape for a single transformation
            or a leading batch axis (``(N, ..., 3)``) for a batch of more than one
        """
        matrices = self._effective_matrices(add_inversion, transpose=False)
        if matrices.dtype != vectors.dtype or matrices.device != vectors.device:
            raise ValueError(
                "vectors and transformation matrices must have the same dtype "
                "and device"
            )

        if matrices.size(0) == 1:
            return vectors @ matrices[0].transpose(0, 1)

        flattened = vectors.reshape(1, -1, 3)
        transformed = flattened @ matrices.transpose(1, 2)
        output_shape: List[int] = [matrices.size(0)]
        for size in vectors.shape:
            output_shape.append(size)
        return transformed.reshape(output_shape)

    def inverse_transform_cartesian(
        self,
        vectors: torch.Tensor,
        add_inversion: bool = False,
    ) -> torch.Tensor:
        """Apply the inverse transformations to Cartesian vectors.

        The inverse of an orthogonal matrix is its transpose.

        :param vectors: ``(..., 3)`` tensor of Cartesian vectors
        :param add_inversion: compose every operation with the inversion before
            inverting
        :return: transformed vectors, with the input shape for a single transformation
            or a leading batch axis for a batch of more than one
        """
        matrices = self._effective_matrices(add_inversion, transpose=True)
        if matrices.dtype != vectors.dtype or matrices.device != vectors.device:
            raise ValueError(
                "vectors and transformation matrices must have the same dtype "
                "and device"
            )

        if matrices.size(0) == 1:
            return vectors @ matrices[0].transpose(0, 1)

        flattened = vectors.reshape(1, -1, 3)
        transformed = flattened @ matrices.transpose(1, 2)
        output_shape: List[int] = [matrices.size(0)]
        for size in vectors.shape:
            output_shape.append(size)
        return transformed.reshape(output_shape)

    def transform_spherical(
        self,
        values: torch.Tensor,
        ell: int,
        sigma: int,
        add_inversion: bool = False,
    ) -> torch.Tensor:
        """Apply the transformations to spherical values.

        :param values: (..., 2*ell+1) tensor of spherical values
        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :param sigma: ``+1`` for a proper spherical representation or ``-1`` for a
            pseudo one. Under an improper transformation, the representation acquires
            the factor ``sigma * (-1) ** ell``.
        :param add_inversion: compose every operation with the inversion
        :return: transformed values, with the input shape for a single transformation or
            a leading batch axis for a batch of more than one
        """
        return self._apply_spherical(
            values, ell, sigma, add_inversion=add_inversion, transpose=False
        )

    def inverse_transform_spherical(
        self,
        values: torch.Tensor,
        ell: int,
        sigma: int,
        add_inversion: bool = False,
    ) -> torch.Tensor:
        """Apply the inverse transformations to spherical values.

        The inverse Wigner-D matrix is its transpose; the inversion-parity factor for an
        improper operation is unchanged by the transposition.

        :param values: (..., 2*ell+1) tensor of spherical values
        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :param sigma: ``+1`` for a proper spherical representation or ``-1`` for a
            pseudo one
        :param add_inversion: compose every operation with the inversion before
            inverting
        :return: transformed values, with the input shape for a single transformation or
            a leading batch axis for a batch of more than one
        """
        return self._apply_spherical(
            values, ell, sigma, add_inversion=add_inversion, transpose=True
        )

    def _apply_spherical(
        self,
        values: torch.Tensor,
        ell: int,
        sigma: int,
        add_inversion: bool,
        transpose: bool,
    ) -> torch.Tensor:
        """Shared kernel for :meth:`transform_spherical` and its inverse."""
        ell = self._validate_ell_range(ell)
        parity = _spherical_parity_factor(ell, sigma, True)
        wigner = self.wigner_D_matrices(ell)
        if transpose:
            wigner = wigner.transpose(1, 2)
        improper = self._effective_improper(add_inversion)

        if wigner.dtype != values.dtype or wigner.device != values.device:
            raise ValueError(
                "values and transformation matrices must have the same dtype and device"
            )

        if wigner.size(0) == 1:
            transformed = values @ wigner[0].transpose(0, 1)
            if parity != 1 and bool(torch.any(improper).item()):
                transformed = transformed * float(parity)
            return transformed

        dimension = 2 * ell + 1
        flattened = values.reshape(1, -1, dimension)
        transformed = flattened @ wigner.transpose(1, 2)
        if parity != 1 and bool(torch.any(improper).item()):
            factors = torch.where(
                improper,
                torch.tensor(float(parity), dtype=values.dtype, device=values.device),
                torch.tensor(1.0, dtype=values.dtype, device=values.device),
            )
            transformed = transformed * factors.view(-1, 1, 1)
        output_shape: List[int] = [wigner.size(0)]
        for size in values.shape:
            output_shape.append(size)
        return transformed.reshape(output_shape)

    def transform_systems(
        self,
        systems: List[System],
        add_inversion: bool = False,
    ) -> List[System]:
        """Apply transformations to a list of systems.

        ``systems[i]`` is transformed by operation ``i``. To apply a batch of
        transformations to the same input system, pass that system repeatedly in the
        list.

        Positions, cell vectors, neighbor-list displacements, and custom data following
        :ref:`o3-conventions` are transformed. Atomic types and periodic-boundary flags
        are preserved.

        :param systems: input systems, one per operation, matching the transformation
            matrices in dtype and device
        :param add_inversion: compose every operation with the inversion
        :return: one transformed :class:`System` per input system
        """
        matrices = self._effective_matrices(add_inversion, transpose=False)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if max_o3_lambda_in_any_system(systems) >= 0:
            wigner_matrices = self._effective_wigner(transpose=False)

        if matrices.size(0) != len(systems):
            raise ValueError(
                f"got {len(systems)} systems but {matrices.size(0)} operations"
            )

        return _transform_systems_batched(systems, matrices, wigner_matrices, improper)

    def inverse_transform_systems(
        self,
        systems: List[System],
        add_inversion: bool = False,
    ) -> List[System]:
        """Apply the inverse transformations to a list of systems.

        See :meth:`transform_systems` for the per-system routing; the inverse uses
        transposed matrices and Wigner-D matrices.

        :param systems: input systems, one per operation
        :param add_inversion: compose every operation with the inversion before
            inverting
        :return: one transformed :class:`System` per input system
        """
        matrices = self._effective_matrices(add_inversion, transpose=True)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if max_o3_lambda_in_any_system(systems) >= 0:
            wigner_matrices = self._effective_wigner(transpose=True)

        if matrices.size(0) != len(systems):
            raise ValueError(
                f"got {len(systems)} systems but {matrices.size(0)} operations"
            )

        return _transform_systems_batched(systems, matrices, wigner_matrices, improper)

    def transform_tensormap(
        self,
        tensor: TensorMap,
        system_ids: Optional[torch.Tensor] = None,
        add_inversion: bool = False,
    ) -> TensorMap:
        """Apply the transformations to a TensorMap and its gradients.

        Scalar, Cartesian, and spherical data are identified by their component-axis
        names, following :ref:`o3-conventions`. With a batch of more than one operation,
        the ``"system"`` sample label assigns each value row to an operation: when
        ``system_ids`` is ``None``, the labels index the batch directly, and otherwise
        rows labelled ``system_ids[i]`` use operation ``i``. Gradient rows use the
        operation of the value row referenced by their ``"sample"`` label. With a single
        operation, the ``"system"`` label is optional and ignored.

        :param tensor: TensorMap to transform, matching the transformation matrices in
            dtype and device
        :param system_ids: optional one-dimensional tensor with one distinct
            ``"system"`` sample label per operation
        :param add_inversion: compose every operation with the inversion
        :return: transformed TensorMap with the same metadata and global information
        """
        matrices = self._effective_matrices(add_inversion, transpose=False)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if max_o3_lambda_in_tensor(tensor) >= 0:
            wigner_matrices = self._effective_wigner(transpose=False)

        if system_ids is not None:
            system_ids = system_ids.to(dtype=torch.long)
        return _transform_tensormap_batched(
            tensor, matrices, wigner_matrices, improper, system_ids
        )

    def inverse_transform_tensormap(
        self,
        tensor: TensorMap,
        system_ids: Optional[torch.Tensor] = None,
        add_inversion: bool = False,
    ) -> TensorMap:
        """Apply the inverse transformations to a TensorMap and its gradients.

        The inverse of an orthogonal matrix is its transpose, and the inverse of a real
        Wigner-D matrix is its transpose, so this shares its kernel with
        :meth:`transform_tensormap` using transposed matrices and Wigner-D matrices. See
        :meth:`transform_tensormap` for row routing.

        :param tensor: TensorMap to transform, matching the transformation matrices in
            dtype and device
        :param system_ids: optional one-dimensional tensor with one distinct
            ``"system"`` sample label per operation
        :param add_inversion: compose every operation with the inversion before
            inverting
        :return: transformed TensorMap with the same metadata and global information
        """
        matrices = self._effective_matrices(add_inversion, transpose=True)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if max_o3_lambda_in_tensor(tensor) >= 0:
            wigner_matrices = self._effective_wigner(transpose=True)

        if system_ids is not None:
            system_ids = system_ids.to(dtype=torch.long)
        return _transform_tensormap_batched(
            tensor, matrices, wigner_matrices, improper, system_ids
        )

    def transform_block(
        self,
        key: LabelsEntry,
        block: TensorBlock,
        system_ids: Optional[torch.Tensor] = None,
        add_inversion: bool = False,
    ) -> TensorBlock:
        """Apply the transformations to a TensorBlock and its gradients.

        See :meth:`transform_tensormap` for the row-routing conventions. The ``key``
        must be the block's key, carrying the ``o3_lambda`` / ``o3_sigma`` metadata
        needed for spherical components.

        :param key: the block's key
        :param block: TensorBlock to transform, matching the transformation matrices in
            dtype and device
        :param system_ids: optional one-dimensional tensor with one distinct
            ``"system"`` sample label per operation
        :param add_inversion: compose every operation with an inversion
        :return: transformed TensorBlock with the same metadata
        """
        matrices = self._effective_matrices(add_inversion, transpose=False)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if _max_o3_lambda_in_block(key, block) >= 0:
            wigner_matrices = self._effective_wigner(transpose=False)

        if system_ids is not None:
            system_ids = system_ids.to(dtype=torch.long)
        return _transform_block_batched(
            key, block, matrices, wigner_matrices, improper, system_ids
        )

    def inverse_transform_block(
        self,
        key: LabelsEntry,
        block: TensorBlock,
        system_ids: Optional[torch.Tensor] = None,
        add_inversion: bool = False,
    ) -> TensorBlock:
        """Apply the inverse transformations to a TensorBlock and its gradients.

        See :meth:`transform_block` for the parameters; the inverse uses transposed
        matrices and Wigner-D matrices.

        :param key: the block's key
        :param block: TensorBlock to transform, matching the transformation matrices in
            dtype and device
        :param system_ids: optional one-dimensional tensor with one distinct
            ``"system"`` sample label per operation
        :param add_inversion: compose every operation with an inversion before inverting
        :return: transformed TensorBlock with the same metadata
        """
        matrices = self._effective_matrices(add_inversion, transpose=True)
        improper = self._effective_improper(add_inversion)
        wigner_matrices: List[torch.Tensor] = []
        if _max_o3_lambda_in_block(key, block) >= 0:
            wigner_matrices = self._effective_wigner(transpose=True)

        if system_ids is not None:
            system_ids = system_ids.to(dtype=torch.long)
        return _transform_block_batched(
            key, block, matrices, wigner_matrices, improper, system_ids
        )


def random_transformations(
    n: int,
    max_angular_momentum: int = 0,
    *,
    device: torch.device,
    dtype: torch.dtype,
    add_inversions: bool = False,
    generator: torch.Generator | None = None,
) -> O3Transformations:
    """Sample ``n`` transformations uniformly from SO(3), or from O(3) when
    inversions are included.

    Rotations are sampled from the Haar measure on SO(3) via random unit quaternions.
    When ``add_inversions`` is ``True``, each matrix is independently negated with
    probability 0.5, giving a uniform distribution over the full O(3) group.

    :param n: positive number of transformations to generate
    :param max_angular_momentum: non-negative maximum angular momentum for Wigner-D
        matrices
    :param device: target device for the output tensors
    :param dtype: target dtype for the output tensors; must be :attr:`torch.float32` or
        :attr:`torch.float64`
    :param add_inversions: if ``True``, sample from O(3) instead of SO(3)
    :param generator: optional :class:`torch.Generator` for reproducible sampling; when
        ``None`` the global RNG is used
    :return: a single :class:`O3Transformations` holding ``n`` operations
    """
    n = validate_integer("n", n, 1)
    max_angular_momentum = validate_integer(
        "max_angular_momentum", max_angular_momentum, 0
    )

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")

    q = torch.randn(n, 4, device=device, dtype=dtype, generator=generator)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    # Quaternion to rotation matrix (standard formula)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(n, 3, 3)

    if add_inversions:
        signs = torch.randint(0, 2, (n,), device=device, generator=generator) * 2 - 1
        R = R * signs.to(dtype=dtype).reshape(n, 1, 1)

    identity = torch.eye(
        3,
        device=R.device,
        dtype=R.dtype,
    ).expand(n, 3, 3)
    if not torch.allclose(
        R @ R.transpose(-1, -2),
        identity,
        atol=1e-5,
    ):
        raise ValueError("Generated transformations are not orthogonal.")

    return O3Transformations(R, max_angular_momentum)


def _component_axis_suffix(axis_name: str, prefix: str) -> tuple[bool, str]:
    """Match a component-axis name and return its supported suffix."""
    suffixes = ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]
    for suffix in suffixes:
        if axis_name == prefix + suffix:
            return True, suffix
    return False, ""


def _validate_component_axis_metadata(
    components: list[Labels],
    key: LabelsEntry,
) -> list[tuple[bool, int, int]]:
    """Validate component axes and return ``(is_spherical, ell, sigma)`` metadata."""
    if len(components) > 10:
        raise ValueError(
            f"can not transform a tensor with {len(components)} component axes; "
            "at most 10 are supported"
        )

    metadata: list[tuple[bool, int, int]] = []
    for component in components:
        axis_name = component.names[0]
        is_cartesian, _ = _component_axis_suffix(axis_name, "xyz")
        is_spherical, suffix = _component_axis_suffix(axis_name, "o3_mu")
        if is_cartesian:
            expected_labels = torch.arange(
                3,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected_labels):
                raise ValueError(
                    f"Cartesian component axis '{axis_name}' must use labels "
                    "[0, 1, 2] in x, y, z order."
                )
            metadata.append((False, 0, 1))
        elif is_spherical:
            ell = validate_integer("ell", int(key["o3_lambda" + suffix]), 0)
            sigma = int(key["o3_sigma" + suffix])
            _spherical_parity_factor(ell, sigma, is_improper=False)

            expected_labels = torch.arange(
                -ell,
                ell + 1,
                device=component.values.device,
                dtype=component.values.dtype,
            )
            if not torch.equal(component.values[:, 0], expected_labels):
                raise ValueError(
                    f"Spherical component axis '{axis_name}' for ell={ell} must use "
                    f"labels from {-ell} through {ell} in ascending order."
                )
            metadata.append((True, ell, sigma))
        else:
            raise ValueError(
                f"Found a component axis '{axis_name}', which is neither a Cartesian "
                "('xyz'/'xyz_1'/'xyz_2'/...) nor spherical ('o3_mu'/'o3_mu_1'/...) "
                "axis; it can not be transformed."
            )

    return metadata


def _max_o3_lambda_in_block(key: LabelsEntry, block: TensorBlock) -> int:
    """Return the largest angular momentum in one block's values or gradients.

    A block containing only scalar or Cartesian component axes returns ``-1``.
    """
    max_o3_lambda = -1
    metadata = _validate_component_axis_metadata(block.components, key)
    for is_spherical, ell, _sigma in metadata:
        if is_spherical and ell > max_o3_lambda:
            max_o3_lambda = ell

    for _gradient_name, gradient in block.gradients():
        gradient_metadata = _validate_component_axis_metadata(
            gradient.components,
            key,
        )
        for is_spherical, ell, _sigma in gradient_metadata:
            if is_spherical and ell > max_o3_lambda:
                max_o3_lambda = ell

    return max_o3_lambda


def max_o3_lambda_in_tensor(tensor: TensorMap) -> int:
    """Return the largest angular momentum in block values or attached gradients.

    A TensorMap containing only scalar or Cartesian component axes returns ``-1``.
    """
    max_o3_lambda = -1
    for key, block in tensor.items():
        block_max = _max_o3_lambda_in_block(key, block)
        if block_max > max_o3_lambda:
            max_o3_lambda = block_max

    return max_o3_lambda


def max_o3_lambda_in_system(system: System) -> int:
    """Return the largest angular momentum in any custom data of a system.

    A system whose custom data contains only scalar or Cartesian component
    axes returns ``-1``.
    """
    max_o3_lambda = -1
    for data_name in system.known_data():
        data = system.get_data(data_name)
        block_max = max_o3_lambda_in_tensor(data)
        if block_max > max_o3_lambda:
            max_o3_lambda = block_max

    return max_o3_lambda


def max_o3_lambda_in_any_system(systems: List[System]) -> int:
    """Return the largest angular momentum across a list of systems' custom data."""
    max_o3_lambda = -1
    for system in systems:
        block_max = max_o3_lambda_in_system(system)
        if block_max > max_o3_lambda:
            max_o3_lambda = block_max

    return max_o3_lambda


def _transform_systems_batched(
    systems: List[System],
    matrices: torch.Tensor,
    wigner_matrices: List[torch.Tensor],
    improper: torch.Tensor,
) -> List[System]:
    """Transform one system per operation in the batch.

    ``matrices[index]`` is the Cartesian operation applied to ``systems[index]``;
    ``wigner_matrices[ell][index]`` and ``improper[index]`` the corresponding
    spherical metadata. The caller ensures ``matrices.size(0) == len(systems)``.
    """
    transformed_systems: List[System] = []
    for index in range(matrices.size(0)):
        system = systems[index]
        if (
            matrices.dtype != system.positions.dtype
            or matrices.device != system.positions.device
        ):
            raise ValueError(
                "system and transformation matrices must have the same dtype and device"
            )

        positions = system.positions.unsqueeze(0) @ matrices[
            index : index + 1
        ].transpose(1, 2)
        cells = system.cell.unsqueeze(0) @ matrices[index : index + 1].transpose(1, 2)

        transformed_systems.append(
            System(
                positions=positions[0],
                types=system.types,
                cell=cells[0],
                pbc=system.pbc,
            )
        )

    for system_index, system in enumerate(systems):
        for options in system.known_neighbor_lists():
            neighbors = system.get_neighbor_list(options)
            # neighbor vectors are stored as (n_pairs, 3, 1); squeeze/unsqueeze
            # around the matmul. Detach the input graph before registering the
            # rotated values below.
            source_values = neighbors.values.detach().squeeze(-1)
            neighbor_values = source_values.unsqueeze(0) @ matrices[
                system_index : system_index + 1
            ].transpose(1, 2)
            rotated_neighbors = TensorBlock(
                values=neighbor_values[0].unsqueeze(-1),
                samples=neighbors.samples,
                components=neighbors.components,
                properties=neighbors.properties,
            )
            register_autograd_neighbors(
                transformed_systems[system_index],
                rotated_neighbors,
            )
            transformed_systems[system_index].add_neighbor_list(
                options,
                rotated_neighbors,
            )

    for system_index, system in enumerate(systems):
        for data_name in system.known_data():
            data = system.get_data(data_name)
            index_wigner: List[torch.Tensor] = []
            for D in wigner_matrices:
                index_wigner.append(D[system_index : system_index + 1])
            transformed_systems[system_index].add_data(
                data_name,
                _transform_tensormap_batched(
                    data,
                    matrices[system_index : system_index + 1],
                    index_wigner,
                    improper[system_index : system_index + 1],
                    None,
                ),
            )

    return transformed_systems


def _transformation_local_indices(
    samples: Labels,
    n_transformations: int,
    system_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map sample rows to local indices into the transformation batch.

    With ``system_ids``, rows are matched to operations by their ``"system"``
    label; without, the labels are used as batch indices directly.
    """
    if n_transformations <= 0:
        raise ValueError("n_transformations must be positive")
    if n_transformations == 1:
        return torch.zeros(
            len(samples),
            dtype=torch.long,
            device=samples.device,
        )
    if "system" not in samples.names:
        raise ValueError("multiple transformations require a 'system' sample dimension")

    labels = samples.column("system").to(dtype=torch.long)
    if system_ids is None:
        if bool(torch.any((labels < 0) | (labels >= n_transformations)).item()):
            raise ValueError("sample system indices exceed the transformation batch")
        return labels

    sorted_ids, sort_order = torch.sort(system_ids)
    positions = torch.searchsorted(sorted_ids, labels)
    positions = torch.clamp(positions, min=0, max=int(sorted_ids.numel()) - 1)
    matched = sorted_ids.index_select(0, positions) == labels
    if not bool(torch.all(matched).item()):
        if torch.jit.is_scripting():
            raise ValueError(
                "block samples contain system labels that are not in system_ids"
            )
        else:
            unknown_labels = torch.unique(labels[~matched])
            raise ValueError(
                f"Block samples contain system labels {unknown_labels.tolist()} "
                f"that are not in system_ids={system_ids.tolist()}. Every sample "
                "must be assigned to a system in the transformation."
            )
    return sort_order.index_select(0, positions)


def _transform_component_values_batched(
    values: torch.Tensor,
    components: list[Labels],
    key: LabelsEntry,
    local_indices: torch.Tensor,
    matrices: torch.Tensor,
    wigner_matrices: list[torch.Tensor],
    improper: torch.Tensor,
) -> torch.Tensor:
    """Transform the component axes of one values tensor.

    ``local_indices[i]`` selects the operation applied to row ``i``. A batch
    of one operation skips the per-row matrix gather entirely.
    """
    metadata = _validate_component_axis_metadata(components, key)
    if len(metadata) == 0:
        return values.clone()

    n_transformations = matrices.size(0)
    transformed = values
    spherical_parity = 1
    for component_index, (is_spherical, ell, sigma) in enumerate(metadata):
        if is_spherical:
            if ell >= len(wigner_matrices):
                raise ValueError(
                    f"ell={ell} exceeds "
                    f"max_angular_momentum={len(wigner_matrices) - 1}."
                )
            axis_matrices = wigner_matrices[ell]
            # the factor acquired by improper operations; applied per row below
            spherical_parity *= _spherical_parity_factor(ell, sigma, True)
        else:
            axis_matrices = matrices

        component_axis = component_index + 1
        moved = torch.movedim(transformed, component_axis, -1)
        moved_shape = moved.shape
        flattened = moved.flatten(start_dim=1, end_dim=-2)
        if n_transformations == 1:
            transformed = flattened @ axis_matrices[0].transpose(0, 1)
        else:
            matrices_for_rows = axis_matrices.index_select(0, local_indices)
            transformed = torch.bmm(
                flattened,
                matrices_for_rows.transpose(1, 2),
            )
        transformed = transformed.reshape(moved_shape)
        transformed = torch.movedim(transformed, -1, component_axis)

    if spherical_parity != 1 and bool(torch.any(improper).item()):
        if n_transformations == 1:
            transformed = transformed * float(spherical_parity)
        else:
            factors = torch.where(
                improper.index_select(0, local_indices),
                torch.tensor(
                    float(spherical_parity),
                    dtype=values.dtype,
                    device=values.device,
                ),
                torch.tensor(1.0, dtype=values.dtype, device=values.device),
            )
            factors_shape: list[int] = [-1]
            for _axis in range(values.dim() - 1):
                factors_shape.append(1)
            transformed = transformed * factors.view(factors_shape)
    return transformed


def _transform_block_batched(
    key: LabelsEntry,
    block: TensorBlock,
    matrices: torch.Tensor,
    wigner_matrices: list[torch.Tensor],
    improper: torch.Tensor,
    system_ids: Optional[torch.Tensor],
) -> TensorBlock:
    """Transform one block and its gradients with a batch of operations."""
    value_indices = _transformation_local_indices(
        block.samples,
        matrices.size(0),
        system_ids,
    )
    new_block = TensorBlock(
        values=_transform_component_values_batched(
            block.values,
            block.components,
            key,
            value_indices,
            matrices,
            wigner_matrices,
            improper,
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for gradient_name, gradient in block.gradients():
        parent_rows = gradient.samples.column("sample").to(dtype=torch.long)
        gradient_indices = value_indices.index_select(0, parent_rows)
        new_block.add_gradient(
            gradient_name,
            TensorBlock(
                values=_transform_component_values_batched(
                    gradient.values,
                    gradient.components,
                    key,
                    gradient_indices,
                    matrices,
                    wigner_matrices,
                    improper,
                ),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            ),
        )
    return new_block


def _transform_tensormap_batched(
    tensor: TensorMap,
    matrices: torch.Tensor,
    wigner_matrices: list[torch.Tensor],
    improper: torch.Tensor,
    system_ids: Optional[torch.Tensor],
) -> TensorMap:
    """Transform a TensorMap with a batch of O(3) operations.

    This is the single implementation behind every tensor-transformation entry
    point: ``matrices[i]`` is the Cartesian operation of local index ``i``,
    ``wigner_matrices[ell][i]`` the Wigner-D matrix of its proper rotational
    part, and ``improper[i]`` whether it includes the inversion. Row routing
    follows :py:meth:`O3Transformation.transform_tensormap`.
    """
    if (
        matrices.dim() != 3
        or matrices.size(0) == 0
        or matrices.size(1) != 3
        or matrices.size(2) != 3
    ):
        raise ValueError("matrices must have shape (N, 3, 3) with N > 0")
    if matrices.dtype != torch.float32 and matrices.dtype != torch.float64:
        raise TypeError("matrices must use float32 or float64")
    if len(tensor) != 0:
        reference_values = tensor.block(0).values
        if (
            matrices.dtype != reference_values.dtype
            or matrices.device != reference_values.device
        ):
            raise ValueError("tensor and matrices must have the same dtype and device")

    blocks: list[TensorBlock] = []
    for key, block in tensor.items():
        blocks.append(
            _transform_block_batched(
                key,
                block,
                matrices,
                wigner_matrices,
                improper,
                system_ids,
            )
        )

    return copy_tensormap_info(tensor, TensorMap(tensor.keys, blocks))
