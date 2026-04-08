import re

import pytest

# C++ TORCH_WARN_DEPRECATION writes to stderr; filter the known quantity warning
_QUANTITY_DEPRECATION_RE = re.compile(
    r"\[W\d+ [\d.:]+\s+model\.cpp:\d+\] Warning: ModelOutput\.quantity is "
    r"deprecated.*\(function set_quantity\)\n"
)


@pytest.fixture(autouse=True)
def fail_test_with_output(capfd):
    yield
    captured = capfd.readouterr()
    # the code should not print anything to stdout or stderr
    assert captured.out == ""
    err = _QUANTITY_DEPRECATION_RE.sub("", captured.err)
    assert err == ""
