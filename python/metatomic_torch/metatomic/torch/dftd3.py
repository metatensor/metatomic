from typing import Dict, List, Optional
from warnings import warn

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
    unit_conversion_factor,
)


_REQUIRED_D3_TABLES = ("rcov", "r4r2", "c6", "cn_ref")
_REQUIRED_DAMPING = ("a1", "a2", "s8")

# Standard Grimme D3 cutoffs from ``tad_dftd3.defaults`` (atomic units / Bohr).
# Used as defaults when the caller does not specify cutoffs explicitly; we
# convert from Bohr to the wrapped model's length unit at construction.
_D3_DISP_CUTOFF_BOHR = 50.0
_D3_CN_CUTOFF_BOHR = 25.0


class DFTD3(torch.nn.Module):
    """
    :py:class:`DFTD3` wraps an :py:class:`AtomisticModel` and adds a DFT-D3(BJ)
    dispersion correction to its energy output(s). The three-body correction
    term is **not** included.

    The wrapper can correct multiple energy variants at once, each with its
    own damping parameters. For every energy output key (e.g. ``"energy"`` or
    ``"energy/pbe"``) listed in ``damping_params``, the wrapper adds the D3
    energy as a differentiable tensor: ``E_corrected = E_base + E_D3``.

    The D3 energy is implemented in pure PyTorch and is naturally
    differentiable: ``torch.autograd`` flows from the corrected energy back to
    ``positions`` and ``cell`` through the neighbor list distances.

    The wrapper does **not** modify ``non_conservative_forces[/<variant>]``
    or ``non_conservative_stress[/<variant>]`` outputs the base model may
    expose. Those remain the base model's direct-predict outputs (no D3
    contribution). Use the autograd path for D3-corrected forces / stress.

    The D3 reference tables (``d3_params``) are shared across variants,
    matching the convention that the Grimme reference data is functional
    independent. Damping parameters (``a1``, ``a2``, ``s8``, ...) are
    provided per variant. All D3 tables and damping parameters must be
    expressed in the wrapped model's length and energy units.
    """

    _energy_keys: List[str]
    _energy_units: Dict[str, str]
    _a1: Dict[str, float]
    _a2: Dict[str, float]
    _s8: Dict[str, float]
    _s6: Dict[str, float]

    def __init__(
        self,
        model: AtomisticModel,
        d3_params: Dict[str, torch.Tensor],
        damping_params: Dict[str, Dict[str, float]],
        cutoff: Optional[float] = None,
        cn_cutoff: Optional[float] = None,
    ):
        """
        :param model: the :py:class:`AtomisticModel` to wrap
        :param d3_params: shared DFT-D3 reference tables with keys ``"rcov"``
            (shape ``(Z,)``), ``"r4r2"`` (shape ``(Z,)``), ``"c6"`` (shape
            ``(Z, Z, M, M)`` — typically ``M = 7``) and ``"cn_ref"`` (shape
            ``(Z, M)`` — per-element CN reference grid, with ``-1`` marking
            absent slots). Tables must use the wrapped model's length and
            energy units.
        :param damping_params: a mapping from an energy output key
            (e.g. ``"energy"``, ``"energy/pbe"``) to a mapping of damping
            parameters for that variant. Each damping map must provide
            ``a1``, ``a2`` and ``s8``; ``s6`` is optional (defaults to 1.0).
        :param cutoff: dispersion-pair cutoff in the wrapped model's length
            unit. If ``None``, defaults to the standard Grimme value of
            ``50 Bohr`` converted into the model's length unit.
        :param cn_cutoff: coordination-number cutoff in the wrapped model's
            length unit. If ``None``, defaults to ``25 Bohr`` converted into
            the model's length unit.
        The wrapped model's atomic types must be real atomic numbers; these
        are used directly to index the D3 parameter tables.
        """
        super().__init__()

        assert isinstance(model, AtomisticModel)

        for key in _REQUIRED_D3_TABLES:
            if key not in d3_params:
                raise KeyError("missing required D3 parameter table '" + key + "'")

        if len(damping_params) == 0:
            raise ValueError(
                "DFTD3 requires at least one energy variant in 'damping_params'"
            )

        capabilities = model.capabilities()
        if capabilities.length_unit == "":
            raise ValueError("DFTD3 requires the wrapped model to define a length unit")

        outputs = capabilities.outputs

        self._validate_d3_params(d3_params)
        rcov = d3_params["rcov"]
        r4r2 = d3_params["r4r2"]
        c6 = d3_params["c6"]
        cn_ref = d3_params["cn_ref"]

        bohr_to_model = float(unit_conversion_factor("bohr", capabilities.length_unit))
        if cutoff is None:
            cutoff = _D3_DISP_CUTOFF_BOHR * bohr_to_model
        if cn_cutoff is None:
            cn_cutoff = _D3_CN_CUTOFF_BOHR * bohr_to_model

        if cutoff <= 0.0:
            raise ValueError("DFTD3 cutoff must be positive, got " + str(cutoff))
        if cn_cutoff <= 0.0:
            raise ValueError("DFTD3 cn_cutoff must be positive, got " + str(cn_cutoff))

        max_atomic_type = max(capabilities.atomic_types)
        if max_atomic_type >= rcov.shape[0]:
            warn(
                "D3 tables do not cover all wrapped-model atomic types: "
                "maximum atomic type is "
                + str(max_atomic_type)
                + " but tables only support up to "
                + str(rcov.shape[0] - 1)
                + ". This will likely cause out-of-bounds errors in D3 table lookups. "
                + "Proceed at your own risk.",
                stacklevel=2,
            )

        for atomic_type in capabilities.atomic_types:
            if atomic_type <= 0:
                raise ValueError(
                    "DFTD3 requires model atomic types to be positive atomic "
                    "numbers, got " + str(atomic_type)
                )

        # The D3 reference tables and damping math run in the model's compute
        # dtype to avoid lossy float32 round-trips for float64 wrapped models.
        if capabilities.dtype == "float64":
            buffer_dtype = torch.float64
        else:
            buffer_dtype = torch.float32

        self.register_buffer("_rcov", rcov.detach().to(dtype=buffer_dtype))
        self.register_buffer("_r4r2", r4r2.detach().to(dtype=buffer_dtype))
        self.register_buffer("_c6", c6.detach().to(dtype=buffer_dtype))
        self.register_buffer("_cn_ref", cn_ref.detach().to(dtype=buffer_dtype))

        self._energy_keys = []
        self._energy_units = {}
        self._a1 = {}
        self._a2 = {}
        self._s8 = {}
        self._s6 = {}

        damping_method = "bj"
        for energy_key, params in damping_params.items():
            if energy_key != "energy" and not energy_key.startswith("energy/"):
                raise ValueError(
                    "DFTD3 damping_params key must be 'energy' or "
                    "'energy/<variant>', got '" + energy_key + "'"
                )
            if energy_key not in outputs:
                raise ValueError(
                    "DFTD3 cannot correct '"
                    + energy_key
                    + "': the wrapped model does not expose this output"
                )
            if outputs[energy_key].unit == "":
                raise ValueError(
                    "DFTD3 requires a defined unit for output '" + energy_key + "'"
                )
            for required in _REQUIRED_DAMPING:
                if required not in params:
                    raise KeyError(
                        "missing required damping parameter '"
                        + required
                        + "' for variant '"
                        + energy_key
                        + "'"
                    )
            method = str(params.get("damping", damping_method))
            if method != "bj":
                raise NotImplementedError(
                    "DFTD3 only implements Becke-Johnson damping; got '"
                    + method
                    + "' for variant '"
                    + energy_key
                    + "'"
                )

            self._energy_keys.append(energy_key)
            self._energy_units[energy_key] = outputs[energy_key].unit
            self._a1[energy_key] = float(params["a1"])
            self._a2[energy_key] = float(params["a2"])
            self._s8[energy_key] = float(params["s8"])
            self._s6[energy_key] = float(params.get("s6", 1.0))

        self._model = model.module
        self._cutoff = cutoff
        self._cn_cutoff = cn_cutoff
        self._neighbor_cutoff = max(cutoff, cn_cutoff)

        self._requested_neighbor_lists = model.requested_neighbor_lists()
        self._neighbor_list = NeighborListOptions(
            cutoff=self._neighbor_cutoff,
            full_list=False,
            strict=True,
            requestor="DFTD3",
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self._requested_neighbor_lists + [self._neighbor_list]

    @staticmethod
    def _validate_d3_params(d3_params: Dict[str, torch.Tensor]):
        rcov = d3_params["rcov"]
        r4r2 = d3_params["r4r2"]
        c6 = d3_params["c6"]
        cn_ref = d3_params["cn_ref"]

        for name, tensor in [
            ("rcov", rcov),
            ("r4r2", r4r2),
            ("c6", c6),
            ("cn_ref", cn_ref),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("D3 table '" + name + "' must be a torch.Tensor")
        if rcov.ndim != 1:
            raise ValueError("'rcov' must be 1D, got shape " + str(tuple(rcov.shape)))
        if r4r2.ndim != 1:
            raise ValueError("'r4r2' must be 1D, got shape " + str(tuple(r4r2.shape)))
        if c6.ndim != 4:
            raise ValueError("'c6' must be 4D, got shape " + str(tuple(c6.shape)))
        if cn_ref.ndim != 2:
            raise ValueError(
                "'cn_ref' must be 2D, got shape " + str(tuple(cn_ref.shape))
            )
        if c6.shape[0] != c6.shape[1]:
            raise ValueError(
                "'c6' must be square in its first two axes, got shape "
                + str(tuple(c6.shape))
            )
        if c6.shape[2] != c6.shape[3]:
            raise ValueError(
                "'c6' must be square in its last two axes, got shape "
                + str(tuple(c6.shape))
            )
        if cn_ref.shape[0] != c6.shape[0]:
            raise ValueError(
                "'cn_ref' first axis must match 'c6' first axis, got "
                + str(cn_ref.shape[0])
                + " vs "
                + str(c6.shape[0])
            )
        if cn_ref.shape[1] != c6.shape[2]:
            raise ValueError(
                "'cn_ref' second axis must match 'c6' last axis, got "
                + str(cn_ref.shape[1])
                + " vs "
                + str(c6.shape[2])
            )
        if rcov.shape[0] < c6.shape[0] or r4r2.shape[0] < c6.shape[0]:
            raise ValueError(
                "'rcov' and 'r4r2' must cover at least 'c6' first axis "
                "length ("
                + str(c6.shape[0])
                + "), got "
                + str(rcov.shape[0])
                + " and "
                + str(r4r2.shape[0])
            )

    @staticmethod
    def wrap(
        model: AtomisticModel,
        d3_params: Dict[str, torch.Tensor],
        damping_params: Dict[str, Dict[str, float]],
        cutoff: Optional[float] = None,
        cn_cutoff: Optional[float] = None,
    ) -> AtomisticModel:
        """Wrap ``model`` with a differentiable DFT-D3(BJ) energy correction.

        The returned :py:class:`AtomisticModel` has the same outputs as the
        input, but each energy variant listed in ``damping_params`` is
        corrected by ``E_D3``. The correction is differentiable so the
        standard autograd path produces D3-corrected conservative forces
        and stress.
        """
        wrapper = DFTD3(
            model=model.eval(),
            d3_params=d3_params,
            damping_params=damping_params,
            cutoff=cutoff,
            cn_cutoff=cn_cutoff,
        )

        capabilities = model.capabilities()
        supported_devices = [
            device
            for device in capabilities.supported_devices
            if device in ["cpu", "cuda"]
        ]
        if len(supported_devices) == 0:
            raise ValueError(
                "DFTD3 only supports CPU and CUDA devices, but the wrapped "
                "model declares " + str(capabilities.supported_devices)
            )

        new_capabilities = ModelCapabilities(
            outputs=capabilities.outputs,
            atomic_types=capabilities.atomic_types,
            interaction_range=max(
                capabilities.interaction_range, wrapper._neighbor_cutoff
            ),
            length_unit=capabilities.length_unit,
            supported_devices=supported_devices,
            dtype=capabilities.dtype,
        )

        return AtomisticModel(
            wrapper.eval(), model.metadata(), capabilities=new_capabilities
        )

    def _compute_cn(
        self,
        atomic_numbers: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the D3 coordination number for every atom.

        The shared half neighbor list visits each pair once; the CN
        contribution is symmetric, so we add it to both atoms. If the shared
        list was built with a larger dispersion cutoff, filter back to the CN
        cutoff before evaluating the counting function.
        """
        if self._cn_cutoff < self._neighbor_cutoff:
            mask = dist <= self._cn_cutoff
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            dist = dist[mask]

        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]

        rcov_i = self._rcov[z_i]
        rcov_j = self._rcov[z_j]
        r_cov_pair = rcov_i + rcov_j

        # CN coordination counting steepness, hardcoded to match tad_dftd3.
        # Ref: k_1 in Section 2.E of https://doi.org/10.1063/1.3382344
        # In the ref above, you can also find a parameter k_2 = 3/4, but in the latest
        # tad-dftd3 codebase, this is set to 1, see https://github.com/tad-mctc/tad-mctc/blob/0d3bb31018520fb8a85bc79c000d4aae01f51235/src/tad_mctc/ncoord/count.py#L51-L72
        k_cn: float = 16.0

        # Counting function: 1 / (1 + exp(-k * (r_cov / r - 1)))
        # Safe clamp, see https://github.com/tad-mctc/tad-mctc/blob/0d3bb31018520fb8a85bc79c000d4aae01f51235/src/tad_mctc/storch/elemental.py#L34-L78
        dist_safe = torch.clamp(dist, min=1e-10)
        exponent = -k_cn * (r_cov_pair / dist_safe - 1.0)
        exponent = torch.clamp(exponent, max=100.0)
        term = 1.0 / (1.0 + torch.exp(exponent))

        n_atoms = atomic_numbers.shape[0]
        cn = torch.zeros(n_atoms, dtype=term.dtype, device=term.device)
        cn = cn.index_add(0, idx_i, term)
        cn = cn.index_add(0, idx_j, term)
        return cn

    def _compute_weights(
        self, atomic_numbers: torch.Tensor, cn: torch.Tensor
    ) -> torch.Tensor:
        """Per-atom Gaussian weights over the (Z, M) reference CN grid.

        For each atom A, ``w_A^k = exp(-k_w * (CN_A - CN_ref_A^k)^2)``,
        with absent reference slots (``CN_ref < 0``) zeroed out, then
        normalized so that ``sum_k w_A^k = 1``. If every weight underflows
        (CN far from any reference), fall back to a one-hot at the largest
        reference CN — this matches the behavior of ``tad_dftd3``.
        """
        # Gaussian weighting steepness on (CN - CN_ref)^2, hardcoded to
        # match tad_dftd3.
        # Ref: k_3 in Section 2.E of https://doi.org/10.1063/1.3382344
        k_weight: float = 4.0

        ref_cn_i = self._cn_ref[atomic_numbers]  # (n_atoms, M)
        # CN references can be zero, e.g. Noble gases,
        # negative values (-1) mean invalid references
        mask = ref_cn_i >= 0.0

        # Match tad-dftd3's numerics, but use a log-sum-exp normalization
        # instead of dividing by tiny raw Gaussian sums. In float32,
        # transition-metal CN values far from all reference points can produce
        # norms around 1e-30; a direct division creates unstable backward
        # intermediates even if the forward value is finite.
        diff = (cn.unsqueeze(1) - ref_cn_i).to(dtype=torch.float64)
        log_w = -k_weight * diff.pow(2)
        neg_inf = torch.full_like(log_w, -torch.inf)
        log_w = torch.where(mask, log_w, neg_inf)

        max_log_w, _ = log_w.max(dim=1, keepdim=True)
        w = torch.exp(log_w - max_log_w)
        w = torch.where(mask, w, torch.zeros_like(w))

        norm = w.sum(dim=1, keepdim=True)
        w_normalized = w / norm
        return w_normalized.to(dtype=cn.dtype)

    def _compute_c6_pairs(
        self,
        atomic_numbers: torch.Tensor,
        weights: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """Effective ``C6_AB`` for each dispersion pair via the (M, M)
        reference grid contracted against the per-atom CN weights.

        Zero C6 reference entries are missing D3 reference points, not physical
        zero-C6 environments, so exclude them from the pair-specific
        normalization denominator. I haven't found a clear statement of this in the D3
        literature, but I checked the values of C6, and the smallest nonzero entry is
        0.9311.
        """
        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]

        w_i = weights[idx_i]  # (P, M)
        w_j = weights[idx_j]
        c6_ref_pairs = self._c6[z_i, z_j]  # (P, M, M)

        weighted_c6 = torch.bmm(w_i.unsqueeze(1), c6_ref_pairs).squeeze(1)
        numerator = (weighted_c6 * w_j).sum(dim=1)

        valid_reference = (c6_ref_pairs != 0.0).to(dtype=w_i.dtype)
        valid_weight = torch.bmm(w_i.unsqueeze(1), valid_reference).squeeze(1)
        denominator = (valid_weight * w_j).sum(dim=1)

        small = torch.full_like(denominator, 1e-20)
        safe_denominator = torch.where(
            denominator > small, denominator, torch.ones_like(denominator)
        )
        zero = torch.zeros((), dtype=numerator.dtype, device=numerator.device)
        c6_pairs = torch.where(denominator > small, numerator / safe_denominator, zero)
        return c6_pairs

    def _compute_pair_energy(
        self,
        atomic_numbers: torch.Tensor,
        c6_pairs: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        dist: torch.Tensor,
        a1: float,
        a2: float,
        s6: float,
        s8: float,
    ) -> torch.Tensor:
        """Becke-Johnson damped pair energy summed over half-list pairs."""
        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]
        r4r2_i = self._r4r2[z_i]
        r4r2_j = self._r4r2[z_j]

        # C8 = C6 * 3 * Q_A * Q_B, with Q stored as r4r2 (length unit).
        # This means R0 = sqrt(C8 / C6) simplifies to sqrt(3 * Q_A * Q_B).
        qq = 3.0 * r4r2_i * r4r2_j
        c8_pairs = c6_pairs * qq
        r0 = torch.sqrt(qq)

        cutoff_r = a1 * r0 + a2

        dist2 = dist * dist
        dist4 = dist2 * dist2
        dist6 = dist4 * dist2
        dist8 = dist6 * dist2

        cutoff2 = cutoff_r * cutoff_r
        cutoff4 = cutoff2 * cutoff2
        cutoff6 = cutoff4 * cutoff2
        cutoff8 = cutoff6 * cutoff2

        denom6 = dist6 + cutoff6
        denom8 = dist8 + cutoff8

        e6 = -(c6_pairs / denom6)
        e8 = -(c8_pairs / denom8)

        energy_pairs = s6 * e6 + s8 * e8
        return energy_pairs.sum()

    def _d3_energy(self, system: System, energy_key: str) -> torch.Tensor:
        """Differentiable scalar D3 energy for ``system`` with the damping
        registered under ``energy_key``."""
        atomic_numbers = system.types.to(torch.int64)

        nl = system.get_neighbor_list(self._neighbor_list)
        sample_values = nl.samples.values.to(torch.int64)
        idx_i = sample_values[:, 0]
        idx_j = sample_values[:, 1]
        dist = torch.linalg.vector_norm(nl.values, dim=1).squeeze(-1)

        cn = self._compute_cn(atomic_numbers, idx_i, idx_j, dist)
        weights = self._compute_weights(atomic_numbers, cn)

        if self._cutoff < self._neighbor_cutoff:
            mask = dist <= self._cutoff
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            dist = dist[mask]

        c6_pairs = self._compute_c6_pairs(atomic_numbers, weights, idx_i, idx_j)

        return self._compute_pair_energy(
            atomic_numbers,
            c6_pairs,
            idx_i,
            idx_j,
            dist,
            a1=self._a1[energy_key],
            a2=self._a2[energy_key],
            s6=self._s6[energy_key],
            s8=self._s8[energy_key],
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        # Determine which of our variants the user requested.
        need_variants: List[str] = []
        for energy_key in self._energy_keys:
            if energy_key not in outputs:
                continue
            if outputs[energy_key].sample_kind != "system":
                raise NotImplementedError(
                    "DFTD3 does not support per-atom corrected energies"
                )
            need_variants.append(energy_key)

        if selected_atoms is not None and len(need_variants) > 0:
            raise NotImplementedError(
                "DFTD3 does not support 'selected_atoms' for corrected outputs"
            )

        # Always forward every requested output to the base model. Non-D3
        # outputs pass through unchanged; D3-corrected energies get the
        # correction added below.
        if len(outputs) == 0:
            results = torch.jit.annotate(Dict[str, TensorMap], {})
        else:
            results = self._model(systems, outputs, selected_atoms)

        if len(need_variants) == 0:
            return results

        for energy_key in need_variants:
            d3_energies: List[torch.Tensor] = []
            for system in systems:
                d3_energies.append(self._d3_energy(system, energy_key))

            energy_result = results[energy_key]
            block = energy_result.block()
            if len(d3_energies) > 0:
                correction = torch.stack(d3_energies, dim=0).reshape(-1, 1)
                corrected_values = block.values + correction.to(
                    dtype=block.values.dtype, device=block.values.device
                )
            else:
                corrected_values = block.values

            corrected_block = TensorBlock(
                values=corrected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            results[energy_key] = TensorMap(energy_result.keys, [corrected_block])

        return results
