"""
Rotate systems and tensor maps under O(3) transformations, routing rows of
multi-system tensors by their ``"system"`` sample label.

:py:class:`O3Transformation` holds a batch of one or more operations. The
tensor-transformation kernel in this module is TorchScript compatible, so a
scripted model can construct transformations from precomputed tensors inside
``forward`` and share one implementation with the eager API.
"""

from numbers import Integral
from typing import Optional

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
            raise TypeError(f"sigma must be an integer, got {type(sigma).__name__}.")
        integer_sigma = int(sigma)
    if integer_sigma not in (-1, 1):
        raise ValueError(f"sigma must be either -1 or +1, got {integer_sigma}.")

    if is_improper:
        return integer_sigma * int((-1) ** ell)

    return 1


def _determinants_3x3(matrices: torch.Tensor) -> torch.Tensor:
    """Return the determinants of a ``(N, 3, 3)`` batch of matrices.

    Written out explicitly because ``torch.linalg`` is not available in
    TorchScript.
    """
    a = matrices[:, 0, 0]
    b = matrices[:, 0, 1]
    c = matrices[:, 0, 2]
    d = matrices[:, 1, 0]
    e = matrices[:, 1, 1]
    f = matrices[:, 1, 2]
    g = matrices[:, 2, 0]
    h = matrices[:, 2, 1]
    i = matrices[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class O3Transformation:
    """
    A batch of one or more O(3) transformations, represented by ``(N, 3, 3)``
    rotation or improper-rotation matrices. A single ``(3, 3)`` matrix is
    stored as a batch of one.

    The constructor stores a copy of ``matrix`` and builds the Wigner-D
    matrices lazily on first use, which requires the eager ``wigners`` package.
    Scripted models construct transformations from precomputed tensors instead,
    through the private constructor arguments.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        max_angular_momentum: int,
        _improper: Optional[torch.Tensor] = None,
        _wigner_D: Optional[list[torch.Tensor]] = None,
    ):
        """
        :param matrix: ``(3, 3)`` or ``(N, 3, 3)`` rotation or
            improper-rotation matrices
        :param max_angular_momentum: non-negative maximum angular momentum for
            which Wigner-D matrices are available
        :param _improper: private trusted path used for internal batching:
            an ``(N,)`` boolean mask of the negative-determinant operations,
            paired with already-validated ``(N, 3, 3)`` matrices which are
            stored without copying or checks
        :param _wigner_D: private, only together with ``_improper``: one
            ``(N, 2*ell+1, 2*ell+1)`` stack of proper-part Wigner-D matrices
            per ``ell`` through ``max_angular_momentum``; ``None`` defers to
            the lazy eager build
        """
        if _improper is not None:
            matrices = matrix
            improper = _improper
        else:
            max_angular_momentum = validate_integer(
                "max_angular_momentum", max_angular_momentum, 0
            )

            if matrix.dim() == 2:
                matrices = matrix.unsqueeze(0)
            else:
                matrices = matrix
            if (
                matrices.dim() != 3
                or matrices.size(0) == 0
                or matrices.size(1) != 3
                or matrices.size(2) != 3
            ):
                if torch.jit.is_scripting():
                    raise ValueError(
                        "transformation matrices must have shape (3, 3) or "
                        "(N, 3, 3) with N > 0"
                    )
                else:
                    raise ValueError(
                        f"Transformation has shape {tuple(matrix.shape)}; "
                        "expected (3, 3) or (N, 3, 3) with N > 0."
                    )

            identity = torch.eye(3, device=matrices.device, dtype=matrices.dtype)
            if not torch.allclose(
                matrices @ matrices.transpose(1, 2),
                identity,
                atol=1e-5,
            ):
                raise ValueError(
                    "Transformation is not orthogonal (R @ R.T deviates from I)."
                )

            # Keep an independent copy so modifying the input tensor later cannot
            # make the matrices disagree with the parity and Wigner-D matrices.
            matrices = matrices.clone()
            improper = _determinants_3x3(matrices) < 0.0

        self._matrices = matrices
        self._max_angular_momentum = max_angular_momentum
        self._improper = improper
        self._wigner_D = _wigner_D

    @property
    def matrices(self) -> torch.Tensor:
        """The ``(N, 3, 3)`` batch of rotation or improper-rotation matrices."""
        return self._matrices

    @property
    def matrix(self) -> torch.Tensor:
        """The ``(3, 3)`` matrix of a single transformation.

        Raises for a batch of more than one operation; use :py:attr:`matrices`
        there.
        """
        if self._matrices.size(0) != 1:
            raise ValueError(
                f"this O3Transformation holds {self._matrices.size(0)} "
                "operations; use .matrices"
            )
        return self._matrices[0]

    @property
    def max_angular_momentum(self) -> int:
        """The maximum angular momentum with available Wigner-D matrices."""
        return self._max_angular_momentum

    @property
    def improper(self) -> torch.Tensor:
        """Boolean mask marking the improper operations in the batch."""
        return self._improper

    @property
    def is_improper(self) -> bool:
        """Whether the transformations are improper, with negative determinant.

        Raises for a batch mixing proper and improper operations; use
        :py:attr:`improper` there.
        """
        n_improper = int(self._improper.to(dtype=torch.long).sum().item())
        if n_improper == 0:
            return False
        if n_improper == self._improper.numel():
            return True
        raise ValueError(
            "this O3Transformation mixes proper and improper operations; use .improper"
        )

    @property
    @torch.jit.unused
    def dtype(self) -> torch.dtype:
        """The dtype of the transformation matrices."""
        return self._matrices.dtype

    @property
    @torch.jit.unused
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

    def _wigner_D_matrices(self) -> list[torch.Tensor]:
        """Return the per-``ell`` Wigner-D stacks, building them on first use."""
        wigner_D = self._wigner_D
        if wigner_D is None:
            wigner_D = self._build_wigner_D()
            self._wigner_D = wigner_D
        return wigner_D

    @torch.jit.unused
    def _build_wigner_D(self) -> list[torch.Tensor]:
        """Build the Wigner-D stacks with the eager numpy-based path."""
        packed = build_packed_wigner_matrices(
            self._matrices,
            self._max_angular_momentum,
        )
        n_matrices = self._matrices.size(0)
        return [
            wigner_matrices_for_lambda(packed, n_matrices, ell)
            for ell in range(self._max_angular_momentum + 1)
        ]

    def wigner_D_matrices(self, ell: int) -> torch.Tensor:
        """Return the proper-part Wigner-D matrices for ``ell``.

        For improper operations, the inversion-parity factor
        ``sigma * (-1) ** ell`` is applied separately when transforming
        spherical values.

        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :return: ``(N, 2*ell+1, 2*ell+1)`` stack of Wigner-D matrices
        """
        ell = self._validate_ell_range(ell)
        return self._wigner_D_matrices()[ell]

    def wigner_D_matrix(self, ell: int) -> torch.Tensor:
        """Return the proper-part Wigner-D matrix of a single transformation.

        Raises for a batch of more than one operation; use
        :py:meth:`wigner_D_matrices` there.

        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :return: (2*ell+1, 2*ell+1) Wigner-D matrix
        """
        if self._matrices.size(0) != 1:
            raise ValueError(
                f"this O3Transformation holds {self._matrices.size(0)} "
                "operations; use .wigner_D_matrices"
            )
        return self.wigner_D_matrices(ell)[0]

    def inverse(self) -> "O3Transformation":
        """Return the batch of inverse transformations.

        The inverse of an orthogonal matrix is its transpose, and the Wigner-D
        matrices of the inverse are the transposed Wigner-D matrices, so this
        returns transposed views of the existing storage without copying.
        """
        wigner_D = self._wigner_D
        inverse_wigner: Optional[list[torch.Tensor]] = None
        if wigner_D is not None:
            inverse_wigner = [D.transpose(1, 2) for D in wigner_D]
        return O3Transformation(
            self._matrices.transpose(1, 2),
            self._max_angular_momentum,
            _improper=self._improper,
            _wigner_D=inverse_wigner,
        )

    def with_inversion(self) -> "O3Transformation":
        """Return the batch composed with the inversion.

        Composing with the inversion negates the matrices and flips their
        parity, while the proper rotational part -- and with it the Wigner-D
        matrices -- is unchanged and shared with this batch.
        """
        return O3Transformation(
            -self._matrices,
            self._max_angular_momentum,
            _improper=torch.logical_not(self._improper),
            _wigner_D=self._wigner_D,
        )

    def transform_cartesian(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the transformations to Cartesian vectors.

        :param vectors: ``(..., 3)`` tensor of Cartesian vectors
        :return: transformed vectors, with the input shape for a single
            transformation or a leading batch axis (``(N, ..., 3)``) for a
            batch of more than one
        """
        if self._matrices.size(0) == 1:
            return vectors @ self._matrices[0].transpose(0, 1)

        flattened = vectors.reshape(1, -1, 3)
        transformed = flattened @ self._matrices.transpose(1, 2)
        output_shape: list[int] = [self._matrices.size(0)]
        for size in vectors.shape:
            output_shape.append(size)
        return transformed.reshape(output_shape)

    def transform_spherical(
        self, values: torch.Tensor, ell: int, sigma: int
    ) -> torch.Tensor:
        """Apply the transformations to spherical values.

        :param values: (..., 2*ell+1) tensor of spherical values
        :param ell: angular momentum in ``[0, max_angular_momentum]``
        :param sigma: ``+1`` for a proper spherical representation or ``-1`` for
            a pseudo one. Under an improper transformation, the representation
            acquires the factor ``sigma * (-1) ** ell``.
        :return: transformed values, with the input shape for a single
            transformation or a leading batch axis for a batch of more than one
        """
        ell = self._validate_ell_range(ell)
        # the parity factor acquired by the improper operations in the batch;
        # this also validates sigma
        parity = _spherical_parity_factor(ell, sigma, True)
        D = self.wigner_D_matrices(ell)

        if self._matrices.size(0) == 1:
            transformed = values @ D[0].transpose(0, 1)
            if parity != 1 and bool(torch.any(self._improper).item()):
                transformed = transformed * float(parity)
            return transformed

        dimension = 2 * ell + 1
        flattened = values.reshape(1, -1, dimension)
        transformed = flattened @ D.transpose(1, 2)
        if parity != 1 and bool(torch.any(self._improper).item()):
            factors = torch.where(
                self._improper,
                torch.tensor(float(parity), dtype=values.dtype, device=values.device),
                torch.tensor(1.0, dtype=values.dtype, device=values.device),
            )
            transformed = transformed * factors.view(-1, 1, 1)
        output_shape: list[int] = [self._matrices.size(0)]
        for size in values.shape:
            output_shape.append(size)
        return transformed.reshape(output_shape)

    def transform_systems(self, system: System) -> list[System]:
        """Apply every transformation in the batch to one System.

        Positions, cell vectors, neighbor-list displacements, and custom data
        following :ref:`o3-conventions` are transformed. Atomic types and
        periodic-boundary flags are preserved.

        :param system: input system, matching the transformation matrices in
            dtype and device
        :return: one transformed System per operation in the batch
        """
        matrices = self._matrices
        if (
            matrices.dtype != system.positions.dtype
            or matrices.device != system.positions.device
        ):
            raise ValueError(
                "system and transformation matrices must have the same dtype and device"
            )

        positions = system.positions.unsqueeze(0) @ matrices.transpose(1, 2)
        cells = system.cell.unsqueeze(0) @ matrices.transpose(1, 2)

        transformed_systems: list[System] = []
        for index in range(matrices.size(0)):
            transformed_systems.append(
                System(
                    positions=positions[index],
                    types=system.types,
                    cell=cells[index],
                    pbc=system.pbc,
                )
            )

        for options in system.known_neighbor_lists():
            neighbors = system.get_neighbor_list(options)
            # neighbor vectors are stored as (n_pairs, 3, 1); squeeze/unsqueeze
            # around the matmul. Detach the input graph before registering the
            # rotated values below.
            source_values = neighbors.values.detach().squeeze(-1)
            neighbor_values = source_values.unsqueeze(0) @ matrices.transpose(1, 2)
            for index in range(matrices.size(0)):
                rotated_neighbors = TensorBlock(
                    values=neighbor_values[index].unsqueeze(-1),
                    samples=neighbors.samples,
                    components=neighbors.components,
                    properties=neighbors.properties,
                )
                register_autograd_neighbors(
                    transformed_systems[index],
                    rotated_neighbors,
                )
                transformed_systems[index].add_neighbor_list(
                    options,
                    rotated_neighbors,
                )

        for data_name in system.known_data():
            data = system.get_data(data_name)
            wigner_matrices: list[torch.Tensor] = []
            if max_o3_lambda_in_tensor(data) >= 0:
                wigner_matrices = self._wigner_D_matrices()
            for index in range(matrices.size(0)):
                index_wigner: list[torch.Tensor] = []
                for D in wigner_matrices:
                    index_wigner.append(D[index : index + 1])
                transformed_systems[index].add_data(
                    data_name,
                    _transform_tensormap_batched(
                        data,
                        matrices[index : index + 1],
                        index_wigner,
                        self._improper[index : index + 1],
                        None,
                    ),
                )

        return transformed_systems

    def transform_tensormap(
        self,
        tensor: TensorMap,
        system_ids: Optional[torch.Tensor] = None,
    ) -> TensorMap:
        """Apply the transformations to a TensorMap and its gradients.

        Scalar, Cartesian, and spherical data are identified by their
        component-axis names, following :ref:`o3-conventions`. With a batch of
        more than one operation, the ``"system"`` sample label assigns each
        value row to an operation: when ``system_ids`` is ``None``, the labels
        index the batch directly, and otherwise rows labelled ``system_ids[i]``
        use operation ``i``. Gradient rows use the operation of the value row
        referenced by their ``"sample"`` label. With a single operation, the
        ``"system"`` label is optional and ignored.

        :param tensor: TensorMap to transform, matching the transformation
            matrices in dtype and device
        :param system_ids: optional one-dimensional tensor with one distinct
            ``"system"`` sample label per operation in the batch
        :return: transformed TensorMap with the same metadata and global
            information
        """
        if system_ids is not None:
            system_ids = system_ids.to(dtype=torch.long)
        wigner_matrices: list[torch.Tensor] = []
        if max_o3_lambda_in_tensor(tensor) >= 0:
            wigner_matrices = self._wigner_D_matrices()
        return _transform_tensormap_batched(
            tensor,
            self._matrices,
            wigner_matrices,
            self._improper,
            system_ids,
        )


def random_transformations(
    n: int,
    max_angular_momentum: int = 0,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_inversions: bool = False,
    generator: torch.Generator | None = None,
) -> list[O3Transformation]:
    """Sample ``n`` transformations uniformly from SO(3), or from O(3) when
    inversions are included.

    Rotations are sampled from the Haar measure on SO(3) via random unit quaternions.
    When ``include_inversions`` is ``True``, each matrix is independently negated with
    probability 0.5, giving a uniform distribution over the full O(3) group.

    :param n: non-negative number of transformations to generate
    :param max_angular_momentum: non-negative maximum angular momentum for
        Wigner-D matrices
    :param device: target device for the output tensors
    :param dtype: target dtype for the output tensors; must be
        :attr:`torch.float32` or :attr:`torch.float64`
    :param include_inversions: if ``True``, sample from O(3) instead of SO(3)
    :param generator: optional :class:`torch.Generator` for reproducible sampling; when
        ``None`` the global RNG is used
    :return: list of ``n`` single-operation :class:`O3Transformation` objects
    """
    n = validate_integer("n", n, 0)
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

    improper = torch.zeros(n, dtype=torch.bool, device=device)

    if include_inversions:
        signs = torch.randint(0, 2, (n,), device=device, generator=generator) * 2 - 1
        R = R * signs.to(dtype=dtype).reshape(n, 1, 1)
        improper = signs < 0

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

    return [
        O3Transformation(
            matrix.unsqueeze(0),
            max_angular_momentum,
            _improper=improper[index : index + 1],
        )
        for index, matrix in enumerate(R.unbind(0))
    ]


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
