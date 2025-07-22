"""Tests for the top level of the Python package contained within this project"""

import pytest

import illumifix


def test_version_is_available() -> None:
    try:
        illumifix.__version__
    except AttributeError:
        pytest.fail("__version__ is not available as an attribute on the package.")
