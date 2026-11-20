import math
from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import (
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from .model import AtomisticModel


# Table I of Habershon, Markland, Manolopoulos, J. Chem. Phys. 131, 024501 (2009)
_PARAMETERS: Dict[str, float] = {
    "epsilon": 0.1852,  # kcal/mol, O-O Lennard-Jones well depth
    "sigma": 3.1589,  # Angstrom, O-O Lennard-Jones diameter
    "q_M": 1.1128,  # |e|, magnitude of the (negative) charge on the M site
    "gamma": 0.73612,  # r_M = gamma r_O + (1 - gamma) (r_H1 + r_H2) / 2
    "D_r": 116.09,  # kcal/mol, O-H stretch "Morse" depth
    "alpha_r": 2.287,  # 1/Angstrom, O-H stretch "Morse" range
    "r_eq": 0.9419,  # Angstrom, O-H equilibrium distance
    "k_theta": 87.85,  # kcal/mol/rad^2, H-O-H bending force constant
    "theta_eq": 107.4,  # degrees, H-O-H equilibrium angle
}

# e^2 / (4 pi eps_0) in kcal/mol * Angstrom, from the exact SI values of e, N_A, the
# thermochemical calorie (4.184 J) and eps_0 = 8.8541878128e-12 F/m (CODATA 2018)
_COULOMB_CONSTANT = 332.0637132991921


class QTIP4PF(torch.nn.Module):
    """
    Flexible q-TIP4P/F water model (`Habershon, Markland and Manolopoulos, J. Chem.
    Phys. 131, 024501 (2009) <https://doi.org/10.1063/1.3167790>`_) implemented in pure
    PyTorch, to be used when testing the integration of metatomic models with
    simulation engines on a physically meaningful system.

    Use :py:func:`qtip4pf_model` to get this model as an :py:class:`AtomisticModel`.

    Atoms must be ordered as O, H, H, O, H, H, ... with each consecutive triplet
    forming one water molecule. Molecules do not need to be whole across periodic
    boundaries. The energy is the sum of

    - a quartic expansion of a Morse potential for each O-H bond, and a harmonic
      H-O-H bend for each molecule:

      .. math::

          D_r \\left(
              \\alpha_r^2 \\Delta r^2 - \\alpha_r^3 \\Delta r^3
              + \\frac{7}{12} \\alpha_r^4 \\Delta r^4
          \\right)
          \\quad \\text{and} \\quad
          \\frac{1}{2} k_\\theta (\\theta - \\theta_\\text{eq})^2,

      with :math:`\\Delta r = r - r_\\text{eq}`;
    - a Lennard-Jones potential between oxygens, truncated at ``cutoff`` without
      shift or tail correction;
    - Coulomb interactions between the charges of different molecules, with
      :math:`q_M / 2` on each hydrogen and :math:`-q_M` on the M site, located at
      :math:`\\gamma \\, r_O + (1 - \\gamma) (r_{H_1} + r_{H_2}) / 2`.

    For fully periodic systems, electrostatics uses an Ewald sum with conducting
    boundary conditions. The real-space part includes all pairs of molecules whose O-O
    distance is below ``cutoff``, and the splitting parameter and reciprocal-space
    cutoff are chosen to make both truncation errors of order ``ewald_accuracy``
    (relative). For non-periodic systems, electrostatics is a direct sum over all
    pairs of molecules. Systems periodic along only one or two directions are not
    supported.

    The cost of the Ewald sum grows as the square of the number of atoms, this model
    is intended for small boxes (up to a few hundred molecules). Systems with the same
    number of atoms given to the model in a single call (e.g. the replicas of a path
    integral simulation) are evaluated together.

    This model only computes the total ``energy`` of each system, in kcal/mol and with
    positions in Angstrom. Forces and stress are computed by the engines with
    automatic differentiation. Per-atom energies and ``selected_atoms`` containing
    only a subset of the atoms are not supported.
    """

    def __init__(
        self,
        *,
        cutoff: float = 9.0,
        ewald_accuracy: float = 1e-6,
        parameters: Optional[Dict[str, float]] = None,
    ):
        """
        :param cutoff: cutoff (in Angstrom) for the Lennard-Jones interaction and for
            the real-space part of the Ewald sum, applied to O-O distances. The
            original model uses 9 Angstrom.
        :param ewald_accuracy: target relative accuracy of the Ewald sum
        :param parameters: new values for some of the parameters of the model, to
            compare with other implementations using rounded parameters. The keys can
            be ``"epsilon"`` and ``"D_r"`` (kcal/mol), ``"sigma"`` and ``"r_eq"``
            (Angstrom), ``"alpha_r"`` (1/Angstrom), ``"k_theta"`` (kcal/mol/rad^2),
            ``"theta_eq"`` (degrees), ``"q_M"`` (electron charge) and ``"gamma"``. The
            default values are the ones from Table I of the original publication.
        """
        super().__init__()

        if not (ewald_accuracy > 0.0 and ewald_accuracy < 1.0):
            raise ValueError(
                f"ewald_accuracy must be between 0 and 1, got {ewald_accuracy}"
            )

        # Real-space Coulomb includes all site pairs of molecules within `cutoff` (O-O
        # distance), and charged sites are ~1 A away from their oxygen: the smallest
        # excluded site-site distance is thus around `cutoff - 2 A`.
        if cutoff <= 4.0:
            raise ValueError(f"cutoff must be larger than 4 A, got {cutoff}")

        p = dict(_PARAMETERS)
        if parameters is not None:
            for key, value in parameters.items():
                if key not in p:
                    raise ValueError(f"unknown q-TIP4P/F parameter '{key}'")
                p[key] = float(value)

        self._epsilon = p["epsilon"]
        self._sigma = p["sigma"]
        self._gamma = p["gamma"]
        self._D_r = p["D_r"]
        self._alpha_r = p["alpha_r"]
        self._r_eq = p["r_eq"]
        self._k_theta = p["k_theta"]
        self._theta_eq = p["theta_eq"] * math.pi / 180.0
        self._k_e = _COULOMB_CONSTANT

        # Charges on (H1, H2, M), and their products. These are created here and not
        # in forward, since `torch.tensor(List[float])` goes through float32 in
        # TorchScript
        q_H = 0.5 * p["q_M"]
        q_M = -p["q_M"]
        charges = torch.tensor([q_H, q_H, q_M], dtype=torch.float64)
        self.register_buffer("_charges", charges)
        self.register_buffer("_qq", charges.reshape(3, 1) * charges.reshape(1, 3))
        # charge products for the (H1, H2), (H2, M), (H1, M) pairs
        self.register_buffer(
            "_intra_qq",
            torch.tensor([q_H * q_H, q_H * q_M, q_H * q_M], dtype=torch.float64),
        )
        self.register_buffer("_types", torch.tensor([8, 1, 1], dtype=torch.int32))

        self._cutoff = float(cutoff)
        # erfc(alpha r) ~ exp(-(alpha r)^2) ~ accuracy at r = cutoff - 2 A, and
        # exp(-k^2 / (4 alpha^2)) ~ accuracy at k = k_cut
        log_accuracy = math.sqrt(-math.log(ewald_accuracy))
        self._ewald_alpha = log_accuracy / (self._cutoff - 2.0)
        self._ewald_k_cut = 2.0 * self._ewald_alpha * log_accuracy
        # interaction of each charge with its own gaussian, per molecule
        self._ewald_self = (
            self._k_e * self._ewald_alpha / math.sqrt(math.pi) * (2.0 * q_H**2 + q_M**2)
        )

        self._nl_options = NeighborListOptions(
            cutoff=self._cutoff, full_list=False, strict=False
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        if outputs["energy"].sample_kind != "system":
            raise ValueError("q-TIP4P/F only provides the total energy of each system")

        if selected_atoms is not None:
            n_atoms = 0
            for system in systems:
                n_atoms += len(system)

            if len(selected_atoms) != n_atoms:
                raise ValueError(
                    "q-TIP4P/F can not compute the energy of a subset of atoms"
                )

        energies = self._compute_energies(systems)

        # Labels are created directly on the device, and marked as unique, to avoid
        # copies between host and device (which synchronize them)
        device = energies.device
        zero = torch.zeros((1, 1), dtype=torch.int32, device=device)
        system_indexes = torch.arange(len(systems), dtype=torch.int32, device=device)
        block = TensorBlock(
            values=energies.reshape(-1, 1),
            samples=Labels(
                ["system"], system_indexes.reshape(-1, 1), assume_unique=True
            ),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], zero, assume_unique=True),
        )
        return {"energy": TensorMap(Labels(["_"], zero, assume_unique=True), [block])}

    def _compute_energies(self, systems: List[System]) -> torch.Tensor:
        """
        Compute the energy of each system, in kcal/mol. Systems with the same number of
        atoms and the same periodicity are evaluated together.
        """
        n_systems = len(systems)
        for system in systems:
            if len(system) % 3 != 0:
                raise ValueError(
                    "q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ..."
                )

        # Everything that is needed on the host, gathered in a single transfer to
        # avoid multiple synchronizations when running on GPU
        types = torch.cat([system.types for system in systems])
        pbc = torch.stack([system.pbc for system in systems])
        cells = torch.stack([system.cell.detach() for system in systems])
        dtype = cells.dtype
        host: List[float] = torch.cat(
            [
                torch.all(types.reshape(-1, 3) == self._types).reshape(1).to(dtype),
                pbc.reshape(-1).to(dtype),
                torch.linalg.vector_norm(cells, dim=2).reshape(-1),
            ]
        ).tolist()
        if host[0] != 1.0:
            raise ValueError("q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ...")

        periodic: List[bool] = []
        n_max: List[List[int]] = []
        for i in range(n_systems):
            n_periodic = int(host[1 + 3 * i] + host[2 + 3 * i] + host[3 + 3 * i])
            if n_periodic != 0 and n_periodic != 3:
                raise ValueError(
                    "q-TIP4P/F only supports fully periodic or non-periodic systems"
                )
            periodic.append(n_periodic == 3)
            lengths = host[1 + 3 * n_systems + 3 * i : 4 + 3 * n_systems + 3 * i]
            n_max.append(_ewald_n_max(lengths, self._ewald_k_cut))

        all_same = True
        for i in range(n_systems):
            if len(systems[i]) != len(systems[0]) or periodic[i] != periodic[0]:
                all_same = False

        energies: List[torch.Tensor] = []
        if all_same:
            # the reciprocal-space grid must contain the k-vectors of all systems
            grid_n_max = [0, 0, 0]
            for i in range(n_systems):
                for d in range(3):
                    grid_n_max[d] = max(grid_n_max[d], n_max[i][d])

            # On CPU, evaluating many systems together is only faster as long as the
            # intermediate arrays stay small, so systems are evaluated in chunks of
            # about 1000 atoms
            chunk = n_systems
            if cells.device.type == "cpu":
                chunk = max(1, 1000 // max(1, len(systems[0])))

            for start in range(0, n_systems, chunk):
                energies.append(
                    self._batch_energies(
                        systems[start : start + chunk], periodic[0], grid_n_max
                    )
                )
        else:
            for i in range(n_systems):
                energies.append(
                    self._batch_energies([systems[i]], periodic[i], n_max[i])
                )

        return torch.cat(energies)

    def _batch_energies(
        self,
        systems: List[System],
        periodic: bool,
        n_max: List[int],
    ) -> torch.Tensor:
        """
        Compute the energies of systems with the same number of atoms, which are all
        periodic (``periodic=True``) or all non-periodic.
        """
        n_systems = len(systems)
        positions = torch.stack([system.positions for system in systems])
        cells = torch.stack([system.cell for system in systems])
        device = positions.device
        dtype = positions.dtype
        n_molecules = positions.shape[1] // 3

        # Tensors that require gradients are split with `unbind` and gathered with
        # `index_select` rather than sliced or indexed: on GPU, the cost is dominated
        # by the number of kernels, and the backward of slicing and indexing needs
        # several of them (zero-filled copies, sorting of indices) for each slice.

        ### intramolecular geometry
        atoms = positions.reshape(n_systems, n_molecules, 3, 3).unbind(2)
        oxygens = atoms[0]
        # O-H vectors, shape (n_systems, n_molecules, 2, 3)
        oh = torch.stack([atoms[1] - oxygens, atoms[2] - oxygens], dim=2)
        # only used for periodic systems
        inverse, volumes = cells, cells
        if periodic:
            inverse, volumes = _inverse_cell(cells)
            # minimum image, so that molecules can be split across the boundary
            images = torch.round(oh.reshape(n_systems, -1, 3) @ inverse) @ cells
            oh = oh - images.reshape(oh.shape)

        r_oh = torch.linalg.vector_norm(oh, dim=-1)
        oh_vectors = oh.unbind(2)
        d1 = oh_vectors[0]
        d2 = oh_vectors[1]
        cos_theta = torch.sum(d1 * d2, dim=-1)
        sin_theta = torch.linalg.vector_norm(torch.linalg.cross(d1, d2, dim=-1), dim=-1)
        theta = torch.atan2(sin_theta, cos_theta)

        a_dr = self._alpha_r * (r_oh - self._r_eq)
        a_dr2 = a_dr * a_dr
        stretch = a_dr2 * (1.0 - a_dr + (7.0 / 12.0) * a_dr2)
        energies = self._D_r * torch.sum(stretch, dim=(1, 2))
        bend = (theta - self._theta_eq) ** 2
        energies = energies + 0.5 * self._k_theta * torch.sum(bend, dim=1)

        # charged sites H1, H2, M relative to the oxygen of their molecule, with shape
        # (n_systems, n_molecules, 3, 3)
        d_M = (0.5 * (1.0 - self._gamma)) * (d1 + d2)
        charged = torch.stack([d1, d2, d_M], dim=2)
        qq = self._qq.to(dtype=dtype, device=device)

        # Positions of the oxygens, with shape (3, n_molecules_total), and of the
        # charged sites relative to the oxygens, with shape (3 * 3, n_molecules_total)
        # for (xyz, site), for all molecules in all systems. Molecules are the last
        # dimension to make element-wise operations over pairs efficient.
        oxygens_xyz = oxygens.reshape(-1, 3).T
        charged_xyz = charged.transpose(2, 3).reshape(-1, 9).T

        ### intermolecular interactions, with molecules indexed over all systems
        if periodic:
            # only keep O-O pairs from the neighbor lists. Their vectors are recomputed
            # from the positions instead of using the values of the neighbor lists, so
            # that gradients only flow through these pairs.
            samples: List[torch.Tensor] = []
            shifts: List[torch.Tensor] = []
            system_index: List[torch.Tensor] = []
            for i, system in enumerate(systems):
                values = system.get_neighbor_list(self._nl_options).samples.values
                samples.append(values)
                shifts.append(values[:, 2:].to(dtype) @ system.cell)
                system_index.append(
                    torch.full((values.shape[0],), i, dtype=torch.int32, device=device)
                )

            is_oxygen = systems[0].types == 8
            pairs = torch.cat(samples)
            oo_pairs = torch.nonzero(is_oxygen[pairs[:, 0]] & is_oxygen[pairs[:, 1]])
            oo_pairs = oo_pairs.reshape(-1)
            pairs = pairs[oo_pairs]
            pair_system = torch.cat(system_index)[oo_pairs]

            pair_energies = self._pair_energies(
                oxygens_xyz,
                charged_xyz,
                pair_system * n_molecules + pairs[:, 0] // 3,
                pair_system * n_molecules + pairs[:, 1] // 3,
                torch.index_select(torch.cat(shifts), 0, oo_pairs).T,
                qq,
                True,
            )
            energies = energies.index_add(0, pair_system, pair_energies)

            # reciprocal-space part of the Ewald sum
            sites = (oxygens.unsqueeze(2) + charged).reshape(n_systems, -1, 3)
            energies = energies + _ewald_reciprocal(
                sites @ inverse,
                self._charges.to(dtype=dtype, device=device).repeat(n_molecules),
                inverse,
                volumes,
                n_max,
                self._ewald_alpha,
                self._ewald_k_cut,
                self._k_e,
            )

            # self-interaction of each charge with its own gaussian
            energies = energies - self._ewald_self * n_molecules

            # remove the intramolecular interactions (H1-H2, H2-M, H1-M) that are
            # included in the reciprocal-space sum
            intra = torch.stack([d2 - d1, d_M - d2, d_M - d1], dim=2)
            intra_r = torch.linalg.vector_norm(intra, dim=-1)
            intra_qq = self._intra_qq.to(dtype=dtype, device=device)
            exclusion = (
                intra_qq * torch.special.erf(self._ewald_alpha * intra_r) / intra_r
            )
            energies = energies - self._k_e * torch.sum(exclusion, dim=(1, 2))
        else:
            # direct sum over all pairs of molecules, no electrostatic cutoff
            pairs = torch.triu_indices(
                n_molecules, n_molecules, offset=1, device=device
            )
            offsets = n_molecules * torch.arange(n_systems, device=device).reshape(
                -1, 1
            )
            pair_system = torch.arange(n_systems, device=device).repeat_interleave(
                pairs.shape[1]
            )
            pair_energies = self._pair_energies(
                oxygens_xyz,
                charged_xyz,
                (pairs[0] + offsets).reshape(-1),
                (pairs[1] + offsets).reshape(-1),
                torch.zeros((3, 1), dtype=dtype, device=device),
                qq,
                False,
            )
            energies = energies.index_add(0, pair_system, pair_energies)

        return energies

    def _pair_energies(
        self,
        oxygens: torch.Tensor,
        charged: torch.Tensor,
        molecule_i: torch.Tensor,
        molecule_j: torch.Tensor,
        shifts: torch.Tensor,
        qq: torch.Tensor,
        ewald: bool,
    ) -> torch.Tensor:
        """
        Compute the intermolecular energy for each pair of molecules (``molecule_i``,
        ``molecule_j``), with ``shifts`` (shape ``(3, n_pairs)``) translating molecule
        j to its periodic image interacting with molecule i. This contains the
        Lennard-Jones energy between oxygens within the cutoff, and either the
        real-space part of the Ewald sum (``ewald=True``) or the bare Coulomb
        interaction (``ewald=False``) between charged sites.

        ``oxygens`` (shape ``(3, n_molecules)``) contains the positions of the oxygens,
        and ``charged`` (shape ``(3 * 3, n_molecules)``) the positions of the charged
        sites relative to the oxygens.
        """
        n_pairs = molecule_i.shape[0]
        oxygens_i = torch.index_select(oxygens, 1, molecule_i)
        oxygens_j = torch.index_select(oxygens, 1, molecule_j)
        oo = oxygens_j - oxygens_i + shifts
        r2_oo = torch.sum(oo * oo, dim=0)
        sigma_r_6 = (self._sigma * self._sigma / r2_oo) ** 3
        lj = 4.0 * self._epsilon * (sigma_r_6 * sigma_r_6 - sigma_r_6)

        # vectors between the charged sites, with shape (3, 3, 3, n_pairs)
        charged_i = torch.index_select(charged, 1, molecule_i)
        charged_j = torch.index_select(charged, 1, molecule_j)
        vectors = (
            oo.reshape(3, 1, 1, n_pairs) + charged_j.reshape(3, 1, 3, n_pairs)
        ) - charged_i.reshape(3, 3, 1, n_pairs)
        r = torch.sqrt(torch.sum(vectors * vectors, dim=0))
        qq = qq.reshape(3, 3, 1)

        inside = r2_oo < self._cutoff * self._cutoff
        if ewald:
            coulomb = qq * torch.special.erfc(self._ewald_alpha * r) / r
            energies = lj + self._k_e * torch.sum(coulomb, dim=(0, 1))
            return torch.where(inside, energies, torch.zeros_like(energies))
        else:
            # Coulomb is not truncated for non-periodic systems
            coulomb = self._k_e * torch.sum(qq / r, dim=(0, 1))
            return coulomb + torch.where(inside, lj, torch.zeros_like(lj))


def qtip4pf_model(
    *,
    cutoff: float = 9.0,
    ewald_accuracy: float = 1e-6,
    parameters: Optional[Dict[str, float]] = None,
) -> AtomisticModel:
    """
    Get the :py:class:`QTIP4PF` water model as an :py:class:`AtomisticModel`, which
    can be exported to a file and then used by any simulation engine:

    .. code-block:: python

        from metatomic.torch.testing import qtip4pf_model

        model = qtip4pf_model()
        model.save("qtip4pf.pt")

    The exported file depends on the version of PyTorch, so it should be created in
    the environment used to run the simulation.

    :param cutoff: see :py:class:`QTIP4PF`
    :param ewald_accuracy: see :py:class:`QTIP4PF`
    :param parameters: see :py:class:`QTIP4PF`
    """
    model = QTIP4PF(cutoff=cutoff, ewald_accuracy=ewald_accuracy, parameters=parameters)
    model.eval()

    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(
                unit="kcal/mol",
                sample_kind="system",
                description="q-TIP4P/F potential energy",
            ),
        },
        atomic_types=[1, 8],
        # electrostatics is long-ranged
        interaction_range=float("inf"),
        length_unit="angstrom",
        supported_devices=["cpu", "cuda"],
        dtype="float64",
    )

    metadata = ModelMetadata(
        name="Test q-TIP4P/F",
        description=(
            "Flexible q-TIP4P/F water model with Ewald electrostatics, to be used "
            "when testing the integration of metatomic models with various simulation "
            "engines. Atoms must be ordered as O, H, H, O, H, H, ..."
        ),
        authors=["metatomic authors"],
        references={
            "model": ["https://doi.org/10.1063/1.3167790"],
            "implementation": ["https://github.com/metatensor/metatomic"],
        },
    )

    return AtomisticModel(model, metadata, capabilities)


def _inverse_cell(cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the inverse and volume of a batch of cells, with lattice vectors as rows"""
    # rows of the inverse transpose are (b x c, c x a, a x b) / det
    cross = torch.linalg.cross(cells.roll(-1, dims=1), cells.roll(-2, dims=1), dim=-1)
    # a . (b x c) = b . (c x a) = c . (a x b) = det
    det = torch.sum(cells * cross, dim=(1, 2)) / 3.0
    return cross.transpose(1, 2) / det.reshape(-1, 1, 1), torch.abs(det)


def _ewald_n_max(lengths: List[float], k_cut: float) -> List[int]:
    """
    Get the largest integers ``n_i`` such that ``|k| < k_cut`` is possible for
    k-vectors with components ``n_i`` in the basis of the reciprocal lattice, since
    ``n_i = k . a_i / 2 pi``. ``lengths`` contains the norms of the lattice vectors.
    """
    return [int(math.ceil(k_cut * length / (2.0 * math.pi))) for length in lengths]


def _ewald_reciprocal(
    fractional: torch.Tensor,
    charges: torch.Tensor,
    inverse: torch.Tensor,
    volumes: torch.Tensor,
    n_max: List[int],
    alpha: float,
    k_cut: float,
    k_e: float,
) -> torch.Tensor:
    """
    Compute the reciprocal-space Ewald energy

    ``k_e (2 pi / V) sum_{0 < |k| < k_cut} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2``

    with ``S(k) = sum_i q_i exp(i k r_i)``, for a batch of systems with the same
    number of charges. The inputs are the fractional coordinates ``s_i`` of the charges
    (shape ``(n_systems, n_charges, 3)``), the inverse of the cells, their volumes,
    and the largest components of the k-vectors along each reciprocal lattice vector.

    With ``k = 2 pi n @ inverse.T`` for integer ``n``, ``exp(i k r) = exp(2 pi i n_0
    s_0) exp(2 pi i n_1 s_1) exp(2 pi i n_2 s_2)``, and the structure factors on the
    full grid of ``n`` are computed with a single matrix multiplication. This uses
    real arithmetic, since complex operations are slow on CPU.
    """
    device = fractional.device
    dtype = fractional.dtype
    n_systems = fractional.shape[0]
    n_sites = fractional.shape[1]

    # values of n along each direction. Only n_0 >= 0 is needed since S(-k) = S(k)^*
    n_0 = torch.arange(0, n_max[0] + 1, dtype=dtype, device=device)
    n_1 = torch.arange(-n_max[1], n_max[1] + 1, dtype=dtype, device=device)
    n_2 = torch.arange(-n_max[2], n_max[2] + 1, dtype=dtype, device=device)
    size_01 = n_0.shape[0] * n_1.shape[0]
    size_2 = n_2.shape[0]

    # 2 pi n s for all sites and values of n, along each direction
    s = ((2.0 * math.pi) * fractional).unbind(2)
    angles_0 = s[0].unsqueeze(2) * n_0
    angles_1 = s[1].unsqueeze(2) * n_1
    angles_2 = s[2].unsqueeze(2) * n_2

    # exp(2 pi i (n_0 s_0 + n_1 s_1)), for all (n_0, n_1)
    cos_0 = torch.cos(angles_0).unsqueeze(3)
    sin_0 = torch.sin(angles_0).unsqueeze(3)
    cos_1 = torch.cos(angles_1).unsqueeze(2)
    sin_1 = torch.sin(angles_1).unsqueeze(2)
    cos_01 = (cos_0 * cos_1 - sin_0 * sin_1).reshape(n_systems, n_sites, size_01)
    sin_01 = (sin_0 * cos_1 + cos_0 * sin_1).reshape(n_systems, n_sites, size_01)

    # real and imaginary parts of the structure factors S(n_0, n_1, n_2), from the
    # products of the (cos_01, sin_01) and (q cos_2, q sin_2) blocks
    charges = charges.reshape(1, n_sites, 1)
    cos_2 = charges * torch.cos(angles_2)
    sin_2 = charges * torch.sin(angles_2)
    products = torch.cat([cos_01, sin_01], dim=2).transpose(1, 2) @ torch.cat(
        [cos_2, sin_2], dim=2
    )
    blocks = products.reshape(n_systems, 2, size_01, 2, size_2).unbind(1)
    cos_01_blocks = blocks[0].unbind(2)
    sin_01_blocks = blocks[1].unbind(2)
    s_real = cos_01_blocks[0] - sin_01_blocks[1]
    s_imag = cos_01_blocks[1] + sin_01_blocks[0]
    s2 = (s_real * s_real + s_imag * s_imag).reshape(n_systems, -1)

    # |k|^2 = (2 pi)^2 n^T G n with the metric G = inverse^T inverse, for all n on
    # the grid. This uses element-wise operations instead of matrix multiplications,
    # which could round differently for different numbers of systems. This way, the
    # k-vectors inside the cutoff are the same when evaluating systems one by one or
    # together.
    grid = torch.stack(torch.meshgrid([n_0, n_1, n_2], indexing="ij")).reshape(3, -1)
    nn = (grid.unsqueeze(1) * grid.unsqueeze(0)).reshape(1, 9, -1)
    metric = torch.sum(inverse.unsqueeze(3) * inverse.unsqueeze(2), dim=1)
    k2 = torch.sum(metric.reshape(n_systems, 9, 1) * nn, dim=1)
    k2 = (4.0 * math.pi * math.pi) * k2

    # n = 0 is excluded, and the n_0 = 0 plane contains both k and -k
    inside = torch.logical_and(k2 > 0.0, k2 < k_cut * k_cut)
    k2 = torch.where(inside, k2, torch.ones_like(k2))
    weights = torch.where(inside, torch.exp(-k2 / (4.0 * alpha * alpha)) / k2, 0.0)
    weights = torch.where(grid[0] > 0, weights, 0.5 * weights)

    return (4.0 * math.pi * k_e) / volumes * torch.sum(weights * s2, dim=1)
