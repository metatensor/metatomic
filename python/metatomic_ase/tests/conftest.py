import re

import pytest


@pytest.fixture(autouse=True)
def fail_test_with_output(capfd):
    yield
    captured = capfd.readouterr()
    # the code should not print anything to stdout or stderr
    # except for expected deprecation warnings
    assert captured.out == ""
    # Filter out expected deprecation warnings
    stderr_lines = captured.err.splitlines()
    unexpected_errors = [
        line
        for line in stderr_lines
        if not re.search(
            r"(ModelOutput\.quantity is deprecated|"
            r"compute_requested_neighbors_from_options.*is deprecated)",
            line,
        )
    ]
    assert "".join(unexpected_errors) == "", f"Unexpected stderr output: {captured.err}"
