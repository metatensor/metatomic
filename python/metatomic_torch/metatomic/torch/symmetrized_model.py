from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatrain.utils.augmentation import _apply_augmentations

from metatomic.torch import ModelOutput, System, register_autograd_neighbors


try:
    from scipy.integrate import lebedev_rule  # noqa: F401
    from scipy.spatial.transform import Rotation  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `scipy` package with `pip install scipy`."
    ) from e
try:
    import spherical  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `spherical` package with `pip install spherical`."
    ) from e
try:
    import quaternionic  # noqa: F401
except ImportError as e:
    raise ImportError(
        "To perform data augmentation on spherical targets, please "
        "install the `quaternionic` package with `pip install quaternionic`."
    ) from e


def _choose_quadrature(L_max: int) -> Tuple[int, int]:
    """
    Choose a Lebedev quadrature order and number of in-plane rotations to integrate
    spherical harmonics up to degree ``L_max``.

    :param L_max: maximum spherical harmonic degree
    :return: (lebedev_order, n_inplane_rotations)
    """
    available = [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        41,
        47,
        53,
        59,
        65,
        71,
        77,
        83,
        89,
        95,
        101,
        107,
        113,
        119,
        125,
        131,
    ]
    # pick smallest order >= L_max
    n = min(o for o in available if o >= L_max)
    # minimal gamma count
    K = L_max + 1
    return n, K


def get_euler_angles_quadrature(lebedev_order: int, n_rotations: int):
    """
    Get the Euler angles and weights for a Lebedev quadrature combined with in-plane
    rotations for SO(3) integration.

    :param lebedev_order: order of the Lebedev quadrature on the unit sphere
    :param n_rotations: number of in-plane rotations per Lebedev node
    :return: alpha, beta, gamma, w arrays of shape (M,), (M,), (K,), (M,)
        respectively, where M is the number of Lebedev nodes and K is the number of
        in-plane rotations.
    """

    # Lebedev nodes (X: (3, M))
    X, w = lebedev_rule(lebedev_order)  # w sums to 4*pi
    x, y, z = X
    alpha = np.arctan2(y, x)  # (M,)
    beta = np.arccos(np.clip(z, -1.0, 1.0))  # (M,)
    gamma = np.linspace(0.0, 2 * np.pi, n_rotations, endpoint=False)  # (K,)

    w_so3 = np.repeat(w / (4 * np.pi * n_rotations), repeats=gamma.size)  # (M*K,)

    A = np.repeat(alpha, gamma.size)  # (N,)
    B = np.repeat(beta, gamma.size)  # (N,)
    G = np.tile(gamma, alpha.size)  # (N,)

    return A, B, G, w_so3


def _rotations_from_angles(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray
) -> Rotation:
    """
    Compose rotations from ZYZ Euler angles.

    :param alpha: array of alpha angles (M,)
    :param beta: array of beta angles (M,)
    :param gamma: array of gamma angles (K,)
    :return: Rotation object containing all (M*K,) rotations
    """

    # Compose ZYZ rotations in SO(3)
    Rot = (
        Rotation.from_euler("z", alpha)
        * Rotation.from_euler("y", beta)
        * Rotation.from_euler("z", gamma)
    )

    return Rot


def _transform_system(system: System, transformation: torch.Tensor) -> System:
    transformed_system = System(
        positions=system.positions @ transformation.T,
        types=system.types,
        cell=system.cell @ transformation.T,
        pbc=system.pbc,
    )
    for options in system.known_neighbor_lists():
        neighbors = mts.detach_block(system.get_neighbor_list(options))

        neighbors.values[:] = (
            neighbors.values.squeeze(-1) @ transformation.T
        ).unsqueeze(-1)

        register_autograd_neighbors(system, neighbors)
        transformed_system.add_neighbor_list(options, neighbors)
    return transformed_system


def _complex_to_real_spherical_harmonics_transform(ell: int) -> np.ndarray:
    """
    Generate the transformation matrix from complex spherical harmonics
    to real spherical harmonics for a given l.
    Returns a transformation matrix of shape ((2l+1), (2l+1)).
    """
    if ell < 0 or not isinstance(ell, int):
        raise ValueError("l must be a non-negative integer.")

    # The size of the transformation matrix is (2l+1) x (2l+1)
    size = 2 * ell + 1
    T = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell  # Index in the matrix
        if m > 0:
            # Real part of Y_{l}^{m}
            T[m_index, ell + m] = 1 / np.sqrt(2) * (-1) ** m
            T[m_index, ell - m] = 1 / np.sqrt(2)
        elif m < 0:
            # Imaginary part of Y_{l}^{|m|}
            T[m_index, ell + abs(m)] = -1j / np.sqrt(2) * (-1) ** m
            T[m_index, ell - abs(m)] = 1j / np.sqrt(2)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            T[m_index, ell] = 1

    # Return the transformation matrix to convert complex to real spherical harmonics
    return T


def _compute_real_wigner_matrices(
    o3_lambda_max: int,
    angles: Tuple[np.ndarray, np.ndarray, np.ndarray],  # alpha, beta, gamma
) -> Dict[int, np.ndarray]:
    wigner = spherical.Wigner(o3_lambda_max)
    R = quaternionic.array.from_euler_angles(*angles)
    D = wigner.D(R)
    wigner_D_matrices = {}
    for ell in range(o3_lambda_max + 1):
        wigner_D_matrices[ell] = np.zeros(
            angles[0].shape + (2 * ell + 1, 2 * ell + 1), dtype=np.complex128
        )
        for mp in range(-ell, ell + 1):
            for m in range(-ell, ell + 1):
                # There is an unexplained conjugation factor in the definition given in
                # the quaternionic library.
                wigner_D_matrices[ell][..., mp + ell, m + ell] = (
                    D[..., wigner.Dindex(ell, mp, m)]
                ).conj()
        U = _complex_to_real_spherical_harmonics_transform(ell)
        wigner_D_matrices[ell] = np.einsum(
            "ij,...jk,kl->...il", U.conj(), wigner_D_matrices[ell], U.T
        )
        assert np.allclose(wigner_D_matrices[ell].imag, 0)
        wigner_D_matrices[ell] = [
            torch.from_numpy(D) for D in wigner_D_matrices[ell].real
        ]

    return wigner_D_matrices


class SymmetrizedModel(torch.nn.Module):
    def __init__(self, base_model, max_o3_lambda, batch_size: int = 32):
        super().__init__()
        self.base_model = base_model
        self.max_o3_lambda = max_o3_lambda
        self.batch_size = batch_size

        # Compute grid
        lebedev_order, n_inplane_rotations = _choose_quadrature(self.max_o3_lambda)
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane_rotations
        )
        self.so3_weights = torch.from_numpy(w_so3)

        # Active rotations
        self.so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        )
        self.n_so3_rotations = self.so3_rotations.size(0)

        # Compute inverse Wigner D representations
        angles_inverse_rotations = (np.pi - gamma, beta, np.pi - alpha)
        self.so3_inverse_rotations = torch.from_numpy(
            _rotations_from_angles(*angles_inverse_rotations).as_matrix()
        )
        self.wigner_D_inverse_rotations = _compute_real_wigner_matrices(
            self.max_o3_lambda, angles_inverse_rotations
        )

        # Compute characters
        # TODO

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Evaluate the model over the grid
        _, backtransformed_outputs = self._eval_over_grid(
            systems, outputs, selected_atoms
        )

        mean_std = self._compute_mean_and_variance(backtransformed_outputs)
        return mean_std

    def _compute_mean_and_variance(self, backtransformed_outputs):
        mean_std_outputs: Dict[str, TensorMap] = {}
        # Iterate over targets
        for target_name in backtransformed_outputs:
            mean_tensors: List[TensorMap] = []
            std_tensors: List[TensorMap] = []
            # Iterate over systems
            for i_sys in range(len(backtransformed_outputs[target_name])):
                tensor_so3 = backtransformed_outputs[target_name][i_sys][1]
                tensor_pso3 = backtransformed_outputs[target_name][i_sys][-1]

                mean_blocks: List[TensorBlock] = []
                std_blocks: List[TensorBlock] = []
                # Iterate over blocks
                for block_so3, block_pso3 in zip(tensor_so3, tensor_pso3, strict=True):
                    w = self.so3_weights.view(
                        self.n_so3_rotations, *[1] * (block_so3.values.ndim - 1)
                    )
                    mean_block = torch.sum(
                        (block_so3.values + block_pso3.values) * 0.5 * w, dim=0
                    )
                    second_moment_block = torch.sum(
                        (block_so3.values**2 + block_pso3.values**2) * 0.5 * w, dim=0
                    )
                    std_block = torch.sqrt(
                        torch.clamp(second_moment_block - mean_block**2, min=0.0)
                    )
                    mean_blocks.append(
                        TensorBlock(
                            samples=Labels("system", torch.tensor([[i_sys]])),
                            components=block_so3.components,
                            properties=block_so3.properties,
                            values=mean_block.unsqueeze(0),
                        )
                    )
                    std_blocks.append(
                        TensorBlock(
                            samples=Labels("system", torch.tensor([[i_sys]])),
                            components=block_so3.components,
                            properties=block_so3.properties,
                            values=std_block.unsqueeze(0),
                        )
                    )
                mean_tensors.append(TensorMap(tensor_so3.keys, mean_blocks))
                std_tensors.append(TensorMap(tensor_so3.keys, std_blocks))

            mean = mts.join(mean_tensors, "samples")
            std = mts.join(std_tensors, "samples")

            # Store results
            mean_std_outputs[target_name + "_mean"] = mean
            mean_std_outputs[target_name + "_std"] = std

        return mean_std_outputs

    def _eval_over_grid(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ):
        """
        Sample the model on the O(3) quadrature.

        :param systems: list of systems to evaluate
        :param model: atomistic model to evaluate
        :param device: device to use for computation
        :return: list of list of model outputs, shape (len(systems), N)
            where N is the number of quadrature points
        """

        device = systems[0].positions.device
        dtype = systems[0].positions.dtype

        transformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
            name: [{-1: None, 1: None} for _ in systems] for name in outputs
        }
        backtransformed_outputs: Dict[str, List[Dict[int, Optional[TensorMap]]]] = {
            name: [{-1: None, 1: None} for _ in systems] for name in outputs
        }
        for i_sys, system in enumerate(systems):
            for inversion in [-1, 1]:
                rotation_outputs: List[Dict[str, TensorMap]] = []
                for batch in range(0, len(self.so3_rotations), self.batch_size):
                    transformed_systems = [
                        _transform_system(
                            system, inversion * R.to(device=device, dtype=dtype)
                        )
                        for R in self.so3_rotations[batch : batch + self.batch_size]
                    ]
                    out = self.base_model(
                        transformed_systems,
                        outputs,
                        selected_atoms,
                    )
                    rotation_outputs.append(out)

                # Combine batch outputs
                for name in transformed_outputs:
                    combined: List[TensorMap] = [r[name] for r in rotation_outputs]
                    transformed_outputs[name][i_sys][inversion] = mts.join(
                        combined,
                        "samples",
                        add_dimension="batch_rotation",
                    )

        n_rot = self.so3_rotations.size(0)
        for name in transformed_outputs:
            for i_sys, system in enumerate(systems):
                for inversion in [-1, 1]:
                    tensor = transformed_outputs[name][i_sys][inversion]
                    _, backtransformed, _ = _apply_augmentations(
                        [system] * n_rot,
                        {name: tensor},
                        list(
                            (
                                self.so3_inverse_rotations.to(
                                    device=device, dtype=dtype
                                )
                                * inversion
                            ).unbind(0)
                        ),
                        self.wigner_D_inverse_rotations,
                    )
                    backtransformed_outputs[name][i_sys][inversion] = backtransformed[
                        name
                    ]

        return transformed_outputs, backtransformed_outputs
