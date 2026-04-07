import pytest


@pytest.fixture(autouse=True)
def fail_test_with_output(capfd):
    yield
    captured = capfd.readouterr()
    # the code should not print anything to stdout or stderr
    assert captured.out == ""
    assert captured.err == ""
