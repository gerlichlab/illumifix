"""Tests for the behavior of the program with which illumination correction weights/scalings are computed and written to disk."""

from pathlib import Path

import hypothesis as hyp
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.compute_illumination_correction import workflow


def test_workflow__requires_version_name(tmp_path: Path) -> None:
    with pytest.raises(TypeError) as err_ctx:
        workflow(base_paths=[], output_path=tmp_path, overwrite=True)
    assert (
        str(err_ctx.value) == "workflow() missing 1 required keyword-only argument: 'version_name'"
    )


@given(version_name=st.text())
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture,))
def test_workflow__crashes_if_output_path_exists_and_overwrite_is_not_allowed(
    tmp_path, version_name
) -> None:
    with pytest.raises(FileExistsError) as err_ctx:
        workflow(base_paths=[], output_path=tmp_path, version_name=version_name, overwrite=False)
    assert str(err_ctx.value) == f"Output path already exists: {tmp_path}"
