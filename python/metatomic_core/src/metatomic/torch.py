import sys


try:
    import metatomic_torch
except ImportError as e:
    raise ImportError(
        "metatomic-torch is required to use the metatomic.torch module. "
        "Please install it with `pip install metatomic-torch` or using "
        "your favorite Python package manager."
    ) from e

# metatomic.torch is registered as an alias in metatomic_torch's __init__.py
assert sys.modules["metatomic.torch"] is metatomic_torch
