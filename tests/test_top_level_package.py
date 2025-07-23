"""Tests for the top level of the Python package contained within this project"""

import pytest

from ome_zarr.format import FormatV02

import illumifix
from illumifix import PROJECT_NAME, ZARR_FORMAT


def test_version_is_available() -> None:
    try:
        illumifix.__version__  # noqa: B018
    except AttributeError:
        pytest.fail("__version__ is not available as an attribute on the package.")


def test_project_name_is_illumifix() -> None:
    assert PROJECT_NAME == "illumifix"


def test_zarr_format_is_as_expected() -> None:
    assert ZARR_FORMAT == FormatV02()
