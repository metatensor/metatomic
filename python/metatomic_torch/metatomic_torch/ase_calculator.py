import warnings


def __getattr__(name: str):
    if name == "MetatomicCalculator":
        from metatomic_ase import MetatomicCalculator

        warnings.warn(
            "Importing MetatomicCalculator from metatomic.torch.ase_calculator is "
            "deprecated and will be removed in a future release. Please import "
            "from metatomic_ase instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return MetatomicCalculator
    elif name == "SymmetrizedCalculator":
        from metatomic_ase import SymmetrizedCalculator

        warnings.warn(
            "Importing SymmetrizedCalculator from metatomic.torch.ase_calculator is "
            "deprecated and will be removed in a future release. Please import "
            "from metatomic_ase instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return SymmetrizedCalculator
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
