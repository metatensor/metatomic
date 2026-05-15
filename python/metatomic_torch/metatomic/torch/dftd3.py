import warnings
from importlib.resources import files
from typing import Dict, List, Optional, Tuple

import numpy as np
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
_D3_PARAMETERS_NPZ = "dftd3_parameters.npz"


def load_dftd3_parameters(
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load the packaged Grimme D3 reference tables.

    The returned tensors are in the atomic-unit convention expected by
    :class:`DFTD3`: ``rcov`` and ``r4r2`` in Bohr, ``c6`` in
    Hartree * Bohr^6, and dimensionless ``cn_ref`` values. The table layout is
    the current pure-PyTorch wrapper layout: ``c6`` has shape ``(Z, Z, M, M)``
    and ``cn_ref`` has shape ``(Z, M)``.
    """

    path = files("metatomic.torch").joinpath("data", _D3_PARAMETERS_NPZ)
    with path.open("rb") as fd:
        with np.load(fd) as data:
            params = {
                key: torch.from_numpy(data[key].copy()) for key in _REQUIRED_D3_TABLES
            }

    if dtype is not None:
        params = {key: value.to(dtype=dtype) for key, value in params.items()}
    return params


class DFTD3(torch.nn.Module):
    """
    :py:class:`DFTD3` wraps an :py:class:`AtomisticModel` and adds a DFT-D3(BJ)
    dispersion correction to its energy output(s). The three-body correction
    term is **not** included.

    The wrapper can correct multiple output variants at once, each with its
    own damping parameters. For every energy output key (e.g. ``"energy"`` or
    ``"energy/pbe"``) listed in ``damping_params``, the wrapper adds the D3
    energy as a differentiable tensor: ``E_corrected = E_base + E_D3``.

    The D3 energy is implemented in pure PyTorch and is naturally
    differentiable: ``torch.autograd`` flows from the corrected energy back to
    ``positions`` and ``cell`` through the neighbor list distances.

    ``damping_params`` can also contain direct output keys such as
    ``"non_conservative_force/<variant>"`` and
    ``"non_conservative_stress/<variant>"``. Direct force/stress outputs are
    corrected only when their output keys are explicitly listed in
    ``damping_params``; energy damping parameters are not inferred for these
    outputs. The D3 contribution to direct force/stress outputs is obtained by
    a local autograd derivative of the same D3 energy expression with respect
    to the neighbor-vector values.

    ``selected_atoms`` is supported with the usual domain-decomposition
    convention: the D3 environment is computed with all atoms in each
    :class:`System`, while pair energies are split equally between the two pair
    endpoints and only the shares belonging to selected atoms are added.

    ``excluded_atom_types`` can be used to disable D3 pair energies involving
    specific atom types, while keeping all atoms in the D3 coordination-number
    environment. This is useful for systems where D3 should not be applied to
    pairs involving selected species, such as common cations.

    .. warning::

        The D3 correction to ``non_conservative_force[/<variant>]`` and
        ``non_conservative_stress[/<variant>]`` requires gradient tracking
        inside ``forward``. These direct corrected outputs can not be computed
        under ``torch.no_grad()`` or ``torch.inference_mode()``.

    The D3 reference tables (``d3_params``) are shared across variants,
    matching the convention that the Grimme reference data is functional
    independent. Damping parameters (``a1``, ``a2``, ``s8``, ...) are
    provided per variant. All D3 tables and damping parameters **must** be
    passed in **atomic units**. The wrapper converts the final D3 energy into the
    wrapped model's energy unit of the corresponding output.
    """

    _energy_keys: List[str]
    _energy_units: Dict[str, str]
    _a1: Dict[str, float]
    _a2: Dict[str, float]
    _s8: Dict[str, float]
    _s6: Dict[str, float]
    _force_keys: List[str]
    _stress_keys: List[str]
    _force_damping_keys: Dict[str, str]
    _stress_damping_keys: Dict[str, str]
    _force_units: Dict[str, str]
    _stress_units: Dict[str, str]

    def __init__(
        self,
        model: AtomisticModel,
        damping_params: Dict[str, Dict[str, float]],
        d3_params: Optional[Dict[str, torch.Tensor]] = None,
        cutoff: Optional[float] = None,
        cn_cutoff: Optional[float] = None,
        excluded_atom_types: Optional[List[int]] = None,
    ):
        """
        :param model: the :py:class:`AtomisticModel` to wrap
        :param damping_params: a mapping from an output key to a mapping of damping
            parameters for that output. Keys can be ``"energy[/<variant>]"``,
            ``"non_conservative_force[/<variant>]"`` or
            ``"non_conservative_stress[/<variant>]"``. Each damping map must provide
            ``a1``, ``a2`` and ``s8``; ``s6`` is optional (defaults to 1.0).
        :param d3_params: shared DFT-D3 reference tables with keys ``"rcov"``
            (shape ``(Z,)``), ``"r4r2"`` (shape ``(Z,)``), ``"c6"`` (shape
            ``(Z, Z, M, M)`` — typically ``M = 5``) and ``"cn_ref"`` (shape
            ``(Z, M)`` — per-element CN reference grid, with ``-1`` marking
            absent slots). Tables must be in D3 atomic units. If ``None``,
            the packaged Grimme D3 reference tables are used.
        :param cutoff: dispersion-pair cutoff in the wrapped model's length
            unit. If ``None``, defaults to the standard Grimme value of
            ``50 Bohr`` converted into the model's length unit.
        :param cn_cutoff: coordination-number cutoff in the wrapped model's
            length unit. If ``None``, defaults to ``25 Bohr`` converted into
            the model's length unit.
        :param excluded_atom_types: optional atom types for which D3 pair
            energies should be disabled. Any pair where either endpoint has one
            of these types contributes zero D3 energy. Coordination numbers are
            still computed with all atoms.
        The wrapped model's atomic types must be real atomic numbers; these
        are used directly to index the D3 parameter tables.
        """
        super().__init__()

        assert isinstance(model, AtomisticModel)

        if d3_params is None:
            d3_params = load_dftd3_parameters()
        for key in _REQUIRED_D3_TABLES:
            if key not in d3_params:
                raise KeyError(f"missing required D3 parameter table '{key}'")

        if len(damping_params) == 0:
            raise ValueError(
                "DFTD3 requires at least one corrected output in 'damping_params'"
            )

        capabilities = model.capabilities()
        if capabilities.length_unit == "":
            raise ValueError("DFTD3 requires the wrapped model to define a length unit")
        self._length_unit = capabilities.length_unit

        outputs = capabilities.outputs

        self._validate_d3_params(d3_params)
        rcov = d3_params["rcov"]
        r4r2 = d3_params["r4r2"]
        c6 = d3_params["c6"]
        cn_ref = d3_params["cn_ref"]

        bohr_to_model = float(unit_conversion_factor("bohr", self._length_unit))
        if cutoff is None:
            cutoff = _D3_DISP_CUTOFF_BOHR * bohr_to_model
        if cn_cutoff is None:
            cn_cutoff = _D3_CN_CUTOFF_BOHR * bohr_to_model

        if cutoff <= 0.0:
            raise ValueError(f"DFTD3 cutoff must be positive, got {cutoff}")
        if cn_cutoff <= 0.0:
            raise ValueError(f"DFTD3 cn_cutoff must be positive, got {cn_cutoff}")

        max_atomic_type = max(capabilities.atomic_types)
        if max_atomic_type >= rcov.shape[0]:
            warnings.warn(
                "D3 tables do not cover all wrapped-model atomic types: "
                f"maximum atomic type is {max_atomic_type} but tables only "
                f"support up to {rcov.shape[0] - 1}. This will likely cause "
                "out-of-bounds errors in D3 table lookups. Proceed at your own risk.",
                stacklevel=2,
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
        if excluded_atom_types is None:
            excluded_atom_types = []
        self.register_buffer(
            "_excluded_atom_types",
            torch.tensor(excluded_atom_types, dtype=torch.int64),
        )

        self._energy_keys = []
        self._energy_units = {}
        self._a1 = {}
        self._a2 = {}
        self._s8 = {}
        self._s6 = {}
        self._force_keys = []
        self._stress_keys = []
        self._force_damping_keys = {}
        self._stress_damping_keys = {}
        self._force_units = {}
        self._stress_units = {}

        # Register the D3 corrections for the outputs explicitly listed in
        # ``damping_params``.
        damping_method = "bj"
        for output_key, params in damping_params.items():
            is_energy = output_key == "energy" or output_key.startswith("energy/")
            is_force = output_key == "non_conservative_force" or output_key.startswith(
                "non_conservative_force/"
            )
            is_stress = (
                output_key == "non_conservative_stress"
                or output_key.startswith("non_conservative_stress/")
            )
            if not (is_energy or is_force or is_stress):
                raise ValueError(
                    "DFTD3 damping_params key must be 'energy[/<variant>]', "
                    "'non_conservative_force[/<variant>]' or "
                    f"'non_conservative_stress[/<variant>]', got '{output_key}'"
                )
            if output_key not in outputs:
                raise ValueError(
                    f"DFTD3 cannot correct '{output_key}': the wrapped model "
                    "does not expose this output"
                )
            if is_energy and outputs[output_key].sample_kind not in ["system", "atom"]:
                raise ValueError(
                    f"DFTD3 requires output '{output_key}' to have "
                    "sample_kind='system' or sample_kind='atom'"
                )
            if outputs[output_key].unit == "":
                raise ValueError(
                    f"DFTD3 requires a defined unit for output '{output_key}'"
                )
            for required in _REQUIRED_DAMPING:
                if required not in params:
                    raise KeyError(
                        f"missing required damping parameter '{required}' "
                        f"for output '{output_key}'"
                    )
            method = str(params.get("damping", damping_method))
            if method != "bj":
                raise NotImplementedError(
                    "DFTD3 only implements Becke-Johnson damping; "
                    f"got '{method}' for output '{output_key}'"
                )

            self._a1[output_key] = float(params["a1"])
            self._a2[output_key] = float(params["a2"])
            self._s8[output_key] = float(params["s8"])
            self._s6[output_key] = float(params.get("s6", 1.0))

            if is_energy:
                self._energy_keys.append(output_key)
                self._energy_units[output_key] = outputs[output_key].unit
            elif is_force:
                self._register_non_conservative_output(
                    output_key,
                    outputs[output_key],
                    "atom",
                    self._force_keys,
                    self._force_damping_keys,
                    self._force_units,
                    output_key,
                )
            else:
                self._register_non_conservative_output(
                    output_key,
                    outputs[output_key],
                    "system",
                    self._stress_keys,
                    self._stress_damping_keys,
                    self._stress_units,
                    output_key,
                )

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
    def _register_non_conservative_output(
        output_key: str,
        output: ModelOutput,
        sample_kind: str,
        output_keys: List[str],
        damping_keys: Dict[str, str],
        units: Dict[str, str],
        damping_key: str,
    ):
        if output.sample_kind != sample_kind:
            raise ValueError(
                f"DFTD3 requires output '{output_key}' to have "
                f"sample_kind='{sample_kind}'"
            )
        if output.unit == "":
            raise ValueError(f"DFTD3 requires a defined unit for output '{output_key}'")

        output_keys.append(output_key)
        damping_keys[output_key] = damping_key
        units[output_key] = output.unit

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
                raise TypeError(f"D3 table '{name}' must be a torch.Tensor")
        if rcov.ndim != 1:
            raise ValueError(f"'rcov' must be 1D, got shape {tuple(rcov.shape)}")
        if r4r2.ndim != 1:
            raise ValueError(f"'r4r2' must be 1D, got shape {tuple(r4r2.shape)}")
        if c6.ndim != 4:
            raise ValueError(f"'c6' must be 4D, got shape {tuple(c6.shape)}")
        if cn_ref.ndim != 2:
            raise ValueError(f"'cn_ref' must be 2D, got shape {tuple(cn_ref.shape)}")
        if c6.shape[0] != c6.shape[1]:
            raise ValueError(
                f"'c6' must be square in its first two axes, got shape "
                f"{tuple(c6.shape)}"
            )
        if c6.shape[2] != c6.shape[3]:
            raise ValueError(
                f"'c6' must be square in its last two axes, got shape {tuple(c6.shape)}"
            )
        if cn_ref.shape[0] != c6.shape[0]:
            raise ValueError(
                f"'cn_ref' first axis must match 'c6' first axis, got "
                f"{cn_ref.shape[0]} vs {c6.shape[0]} vs {c6.shape[0]}"
            )
        if cn_ref.shape[1] != c6.shape[2]:
            raise ValueError(
                f"'cn_ref' second axis must match 'c6' last axis, got "
                f"{cn_ref.shape[1]} vs {c6.shape[2]} vs {c6.shape[2]}"
            )
        if rcov.shape[0] < c6.shape[0] or r4r2.shape[0] < c6.shape[0]:
            raise ValueError(
                f"'rcov' and 'r4r2' must cover at least 'c6' first axis "
                f"length ({c6.shape[0]}), got {rcov.shape[0]} and {r4r2.shape[0]}"
            )

    @staticmethod
    def wrap(
        model: AtomisticModel,
        damping_params: Dict[str, Dict[str, float]],
        d3_params: Optional[Dict[str, torch.Tensor]] = None,
        cutoff: Optional[float] = None,
        cn_cutoff: Optional[float] = None,
        excluded_atom_types: Optional[List[int]] = None,
    ) -> AtomisticModel:
        """Wrap ``model`` with a differentiable DFT-D3(BJ) energy correction.

        The returned :py:class:`AtomisticModel` has the same outputs as the
        input, but each output listed in ``damping_params`` is corrected by
        the corresponding D3 contribution. The correction is differentiable so the
        standard autograd path produces D3-corrected conservative forces
        and stress.
        """
        wrapper = DFTD3(
            model=model.eval(),
            damping_params=damping_params,
            d3_params=d3_params,
            cutoff=cutoff,
            cn_cutoff=cn_cutoff,
            excluded_atom_types=excluded_atom_types,
        )

        capabilities = model.capabilities()
        supported_devices = [device for device in capabilities.supported_devices]
        if len(supported_devices) == 0:
            raise ValueError(
                "DFTD3 only supports CPU and CUDA devices, but the wrapped "
                f"model declares {capabilities.supported_devices}"
            )

        # ``AtomisticModel.capabilities()`` includes compatibility aliases for
        # deprecated output names. Keep only the names declared by the input
        # model; ``AtomisticModel`` will add aliases again if needed.
        declared_outputs = {}
        for name in model._model_capabilities_outputs_names:
            output = capabilities.outputs[name]
            if name in wrapper._energy_keys and output.sample_kind == "atom":
                output = ModelOutput(
                    unit=output.unit,
                    sample_kind="system",
                    explicit_gradients=output.explicit_gradients,
                    description=output.description,
                )
            declared_outputs[name] = output

        new_capabilities = ModelCapabilities(
            outputs=declared_outputs,
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
            mask = dist <= self._cn_cutoff * unit_conversion_factor(
                self._length_unit, "bohr"
            )
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

    def _remove_excluded_pairs(
        self,
        atomic_numbers: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._excluded_atom_types.numel() == 0:
            return idx_i, idx_j, dist

        excluded_atom_types = self._excluded_atom_types.to(device=atomic_numbers.device)
        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]
        excluded_i = (z_i.unsqueeze(1) == excluded_atom_types.unsqueeze(0)).any(dim=1)
        excluded_j = (z_j.unsqueeze(1) == excluded_atom_types.unsqueeze(0)).any(dim=1)
        keep_pair = ~(excluded_i | excluded_j)
        return idx_i[keep_pair], idx_j[keep_pair], dist[keep_pair]

    def _compute_pair_energies(
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
        """Becke-Johnson damped energy for every half-list pair."""
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
        return energy_pairs

    @staticmethod
    def _selected_atoms_for_system(
        selected_atoms: Optional[Labels], system_index: int
    ) -> Optional[torch.Tensor]:
        if selected_atoms is None:
            return None

        selected_values = selected_atoms.values.to(torch.int64)
        mask = selected_values[:, 0] == system_index
        return selected_values[mask, 1]

    @staticmethod
    def _system_sample_indices(block: TensorBlock) -> torch.Tensor:
        return block.samples.values[:, 0].to(torch.int64)

    @staticmethod
    def _atom_sample_indices(block: TensorBlock, systems: List[System]) -> torch.Tensor:
        sample_values = block.samples.values.to(torch.int64)
        offsets = torch.jit.annotate(List[int], [])
        offset = 0
        for system in systems:
            offsets.append(offset)
            offset += system.positions.shape[0]

        offset_tensor = torch.tensor(
            offsets, dtype=torch.int64, device=sample_values.device
        )
        return offset_tensor[sample_values[:, 0]] + sample_values[:, 1]

    @staticmethod
    def _energy_block_as_system(block: TensorBlock, n_systems: int) -> TensorBlock:
        """
        Called when the energy block is per-atom; sum back to per-system values so we
        can add the D3 correction.
        """
        if block.samples.names == ["system"]:
            return block
        if block.samples.names != ["system", "atom"]:
            raise ValueError(
                "DFTD3 can only correct energy blocks with 'system' or "
                "'system', 'atom' samples"
            )

        sample_values = block.samples.values.to(torch.int64)
        system_indices = sample_values[:, 0]
        values = torch.zeros(
            n_systems,
            block.values.shape[1],
            dtype=block.values.dtype,
            device=block.values.device,
        )
        values = values.index_add(0, system_indices, block.values)
        system_samples = Labels(
            "system",
            torch.arange(
                n_systems, dtype=torch.int64, device=block.values.device
            ).reshape(-1, 1),
        )
        return TensorBlock(
            values=values,
            samples=system_samples,
            components=block.components,
            properties=block.properties,
        )

    def _neighbor_pairs(
        self, system: System
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nl = system.get_neighbor_list(self._neighbor_list)
        sample_values = nl.samples.values.to(torch.int64)
        idx_i = sample_values[:, 0]
        idx_j = sample_values[:, 1]
        return idx_i, idx_j, nl.values

    def _d3_energy_from_neighbor_values(
        self,
        system: System,
        damping_key: str,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        neighbor_values: torch.Tensor,
        selected_atom_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Differentiable scalar D3 energy for ``system`` with the damping
        registered under ``damping_key``. ``neighbor_values`` are in the wrapped
        model's length unit."""
        atomic_numbers = system.types.to(torch.int64)

        dist = torch.linalg.vector_norm(neighbor_values, dim=1).squeeze(
            -1
        ) * unit_conversion_factor(self._length_unit, "bohr")

        # TODO: if we ever support workflows that request multiple distinct
        # damping keys at once, these damping-independent pair terms
        # (CN/weights/C6) could be computed once and reused.
        cn = self._compute_cn(atomic_numbers, idx_i, idx_j, dist)
        weights = self._compute_weights(atomic_numbers, cn)

        if self._cutoff < self._neighbor_cutoff:
            mask = dist <= self._cutoff * unit_conversion_factor(
                self._length_unit, "bohr"
            )
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            dist = dist[mask]

        idx_i, idx_j, dist = self._remove_excluded_pairs(
            atomic_numbers, idx_i, idx_j, dist
        )
        c6_pairs = self._compute_c6_pairs(atomic_numbers, weights, idx_i, idx_j)

        energy_pairs = self._compute_pair_energies(
            atomic_numbers,
            c6_pairs,
            idx_i,
            idx_j,
            dist,
            a1=self._a1[damping_key],
            a2=self._a2[damping_key],
            s6=self._s6[damping_key],
            s8=self._s8[damping_key],
        )
        if selected_atom_indices is None:
            return energy_pairs.sum()

        atom_weights = torch.zeros(
            atomic_numbers.shape[0],
            dtype=energy_pairs.dtype,
            device=energy_pairs.device,
        )
        atom_weights = atom_weights.index_fill(0, selected_atom_indices, 1.0)
        pair_weights = 0.5 * (atom_weights[idx_i] + atom_weights[idx_j])
        return (energy_pairs * pair_weights).sum()

    def _d3_energy(
        self,
        system: System,
        damping_key: str,
        selected_atom_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        idx_i, idx_j, neighbor_values = self._neighbor_pairs(system)
        return self._d3_energy_from_neighbor_values(
            system, damping_key, idx_i, idx_j, neighbor_values, selected_atom_indices
        )

    def _d3_direct_derivatives(
        self,
        system: System,
        damping_key: str,
        selected_atom_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """D3 direct force and stress in ``Hartree / length`` and
        ``Hartree / length^3``, where ``length`` is the wrapped model's length
        unit."""
        idx_i, idx_j, neighbor_values = self._neighbor_pairs(system)
        n_atoms = system.positions.shape[0]
        forces = torch.zeros(
            n_atoms,
            3,
            dtype=neighbor_values.dtype,
            device=neighbor_values.device,
        )
        stress = torch.zeros(
            3,
            3,
            dtype=neighbor_values.dtype,
            device=neighbor_values.device,
        )

        if neighbor_values.numel() == 0:
            return forces, stress

        pair_values = neighbor_values.detach().clone()
        pair_values.requires_grad_(True)
        energy = self._d3_energy_from_neighbor_values(
            system, damping_key, idx_i, idx_j, pair_values, selected_atom_indices
        )

        if not energy.requires_grad:
            raise RuntimeError(
                "DFTD3 non-conservative force/stress correction requires "
                "gradient tracking to be enabled"
            )

        gradients = torch.autograd.grad([energy], [pair_values])
        dE_dr = gradients[0]
        if dE_dr is None:
            raise RuntimeError("failed to compute DFTD3 neighbor-vector gradients")

        pair_vectors = pair_values.squeeze(-1)
        dE_dr_vectors = dE_dr.squeeze(-1)

        forces = forces.index_add(0, idx_i, dE_dr_vectors)
        forces = forces.index_add(0, idx_j, -dE_dr_vectors)

        volume = torch.abs(torch.linalg.det(system.cell))
        if volume > 0.0:
            stress = torch.einsum("pi,pj->ij", pair_vectors, dE_dr_vectors) / volume
        return forces, stress

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

        need_force_keys: List[str] = []
        need_stress_keys: List[str] = []
        need_non_conservative_damping_keys: List[str] = []
        for force_key in self._force_keys:
            if force_key in outputs:
                if outputs[force_key].sample_kind != "atom":
                    raise NotImplementedError(
                        "DFTD3 only supports atom-sample non-conservative forces"
                    )
                need_force_keys.append(force_key)
                damping_key = self._force_damping_keys[force_key]
                if damping_key not in need_non_conservative_damping_keys:
                    need_non_conservative_damping_keys.append(damping_key)

        for stress_key in self._stress_keys:
            if stress_key in outputs:
                if outputs[stress_key].sample_kind != "system":
                    raise NotImplementedError(
                        "DFTD3 only supports system-sample non-conservative stress"
                    )
                need_stress_keys.append(stress_key)
                damping_key = self._stress_damping_keys[stress_key]
                if damping_key not in need_non_conservative_damping_keys:
                    need_non_conservative_damping_keys.append(damping_key)

        # Always forward every requested output to the base model. Non-D3
        # outputs pass through unchanged; D3-corrected energies get the
        # correction added below.
        if len(outputs) == 0:
            results = torch.jit.annotate(Dict[str, TensorMap], {})
        else:
            results = self._model(systems, outputs, selected_atoms)

        if len(need_variants) == 0 and len(need_non_conservative_damping_keys) == 0:
            return results

        # First compute the D3 correction for energy variants, which will automatically
        # correct the corresponding conservative forces and stress via autograd.
        for energy_key in need_variants:
            d3_energies: List[torch.Tensor] = []
            for system_i, system in enumerate(systems):
                selected = self._selected_atoms_for_system(selected_atoms, system_i)
                d3_energies.append(self._d3_energy(system, energy_key, selected))

            energy_result = results[energy_key]
            block = self._energy_block_as_system(energy_result.block(), len(systems))
            if len(d3_energies) > 0:
                correction_by_system = torch.stack(d3_energies, dim=0)
                correction = correction_by_system.index_select(
                    0, self._system_sample_indices(block)
                ).reshape(-1, 1)
                corrected_values = block.values + correction.to(
                    dtype=block.values.dtype, device=block.values.device
                ) * unit_conversion_factor("hartree", self._energy_units[energy_key])
            else:
                corrected_values = block.values

            corrected_block = TensorBlock(
                values=corrected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            results[energy_key] = TensorMap(energy_result.keys, [corrected_block])

        # Calculate the corrections for non-conservative forces and stresses
        non_conservative_forces = torch.jit.annotate(Dict[str, List[torch.Tensor]], {})
        non_conservative_stresses = torch.jit.annotate(
            Dict[str, List[torch.Tensor]], {}
        )
        for damping_key in need_non_conservative_damping_keys:
            d3_forces: List[torch.Tensor] = []
            d3_stresses: List[torch.Tensor] = []
            for system_i, system in enumerate(systems):
                selected = self._selected_atoms_for_system(selected_atoms, system_i)
                force, stress = self._d3_direct_derivatives(
                    system, damping_key, selected
                )
                d3_forces.append(force)
                d3_stresses.append(stress)
            non_conservative_forces[damping_key] = d3_forces
            non_conservative_stresses[damping_key] = d3_stresses

        for force_key in need_force_keys:
            damping_key = self._force_damping_keys[force_key]
            force_result = results[force_key]
            block = force_result.block()
            correction_by_atom = torch.cat(non_conservative_forces[damping_key], dim=0)
            correction = correction_by_atom.index_select(
                0, self._atom_sample_indices(block, systems)
            ).reshape(-1, 3, 1)
            force_unit = f"hartree/{self._length_unit}"
            corrected_values = block.values + correction.to(
                dtype=block.values.dtype, device=block.values.device
            ) * unit_conversion_factor(force_unit, self._force_units[force_key])
            corrected_block = TensorBlock(
                values=corrected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            results[force_key] = TensorMap(force_result.keys, [corrected_block])

        for stress_key in need_stress_keys:
            damping_key = self._stress_damping_keys[stress_key]
            stress_result = results[stress_key]
            block = stress_result.block()
            correction_by_system = torch.stack(
                non_conservative_stresses[damping_key], dim=0
            )
            correction = correction_by_system.index_select(
                0, self._system_sample_indices(block)
            ).unsqueeze(-1)
            stress_unit = f"hartree/{self._length_unit}^3"
            corrected_values = block.values + correction.to(
                dtype=block.values.dtype, device=block.values.device
            ) * unit_conversion_factor(stress_unit, self._stress_units[stress_key])
            corrected_block = TensorBlock(
                values=corrected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            results[stress_key] = TensorMap(stress_result.keys, [corrected_block])

        return results
