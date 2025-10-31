"""Tests for the application of illumination correction scalings/weights """

from pathlib import Path

import hypothesis as hyp
from hypothesis import given
import hypothesis.extra.numpy as hyp_npy
import numpy as np
import pytest
import zarr

from scripts.apply_illumination_correction import write_group


def test_write_group__creates_the_expected_root(tmp_path: Path) -> None:
    subfolder: str = "P0001.zarr"
    data: np.ndarray = np.arange(6).reshape((2, 3))
    target: Path = tmp_path / subfolder
    assert not target.exists()
    write_group(data=data, path=target)
    assert target.is_dir()


@pytest.mark.skip("not implemented")
def test_write_group__output_parse_recovers_input(tmp_path: Path) -> None:
    pass
