"""Tests for ZARR-related utilities"""

import copy
import functools
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import zarr
from expression import Result, fst, result, snd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyp_npy
from hypothesis.strategies import SearchStrategy
from ome_zarr.axes import KNOWN_AXES as OME_AXIS_TYPES

from illumifix.zarr_tools import (
    ArrayWithDimensions,
    AxisMapping,
    CanonicalImageDimensions,
    ChannelMeta,
    Channels,
    DimensionsForIlluminationCorrectionScaling,
    ZarrAxis,
    compute_corrected_channels,
    parse_single_array_and_dimensions_from_zarr_group,
)

DEFAULT_DATA_NAME = "test_zarr_data"
DEFAULT_DATA_TYPE = "u2"
IMAGE_DATA_TYPE = np.uint16
WEIGHTS_DATA_TYPE = np.float64


def _build_dataset(dims: tuple[int, ...], dtype: str = DEFAULT_DATA_TYPE) -> np.ndarray:
    return np.arange(
        functools.reduce(lambda acc, n: acc * n, dims, 1),
        dtype=dtype,
    ).reshape(dims)


LEGIT_TARGET_TYPES = [CanonicalImageDimensions, DimensionsForIlluminationCorrectionScaling]
NON_TIME_DIMENSIONS = (4, 3, 2, 2)
BASE_DATA_DIMENSIONS = (5, *NON_TIME_DIMENSIONS)
BASE_DATA = _build_dataset(BASE_DATA_DIMENSIONS)


def _infer_dataset_creation_arguments(data: np.ndarray) -> Mapping[str, object]:
    return {
        "path": f"/{DEFAULT_DATA_NAME}",
        "shape": data.shape,
        "dtype": data.dtype,
        "chunks": (1,) + data.shape[1:],
    }


def prepare_axis_metadata(
    metadata: list[Mapping[str, str]],
) -> Mapping[str, list[Mapping[str, list[Mapping[str, str]]]]]:
    return {"multiscales": [{"axes": metadata}]}


def build_zarr_array(
    *,
    root: zarr.Group,
    data: np.ndarray,
    name: str = DEFAULT_DATA_NAME,
    dtype: Optional[str] = None,
) -> zarr.Array:
    arr = root.create_dataset(
        name=name,
        shape=data.shape,
        dtype=dtype or data.dtype,
        chunks=(1,) + data.shape[1:],
    )
    arr[:] = data
    return arr


def prepare_zarr_group(
    *,
    path: Path,
    data: np.ndarray,
    axis_metadata: list[Mapping[str, str]],
    name: str = DEFAULT_DATA_NAME,
    dtype: str = DEFAULT_DATA_TYPE,
) -> zarr.Group:
    root = zarr.group(store=zarr.DirectoryStore(path=path))
    root.attrs.update(prepare_axis_metadata(axis_metadata))
    build_zarr_array(
        root=root, data=data, name=name, dtype=dtype
    )  # NB: purely for side effect of writing the data
    return root


def _create_space_axis_map(name: str) -> Mapping[str, str]:
    return {"name": name, "type": "space", "unit": "micrometer"}


SPACE_AXES = [_create_space_axis_map(ax) for ax in ["z", "y", "x"]]


@pytest.fixture
def zarr_path(tmp_path) -> Path:
    return tmp_path / "test.zarr"


@pytest.fixture
def zarr_group(zarr_path) -> zarr.Group:
    return prepare_zarr_group(
        path=zarr_path,
        data=BASE_DATA,
        axis_metadata=[
            {"name": "t", "type": "time", "unit": "minute"},
            {"name": "c", "type": "channel"},
            *SPACE_AXES,
        ],
    )


def _wrap_parse_error_message(msg: str) -> Result[ArrayWithDimensions, list[str]]:
    return Result.Error(msg)


@pytest.mark.parametrize("axis", ZarrAxis)
def test_zarr_axis__corresponds_to_ome_zarr(axis: ZarrAxis):
    assert axis.value == OME_AXIS_TYPES[axis.name]


@pytest.mark.parametrize("axis", ZarrAxis)
def test_zarr_axis__value_is_typename(axis: ZarrAxis):
    assert axis.value == axis.typename


@pytest.mark.parametrize("target_type", LEGIT_TARGET_TYPES)
def test_group_argument_type_other_than_zarr_group_gives_attribute_error(target_type):
    observed = parse_single_array_and_dimensions_from_zarr_group(
        group=BASE_DATA, target_type=target_type
    )
    expected = _wrap_parse_error_message("Alleged zarr.Group instance lacks .attrs member")
    assert observed == expected


@pytest.mark.parametrize("target_type", [object, AxisMapping])
def test_target_type_argument_type_other_than_axis_mapping_subtype_gives_type_error(
    zarr_group, target_type
):
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert (
                        msg
                        == f"{target_type.__name__} is not a valid target type for parsing ZARR group dimensions"
                    )
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


@pytest.mark.parametrize("target_type", LEGIT_TARGET_TYPES)
def test_zarr_group_must_have_only_one_array(zarr_group, target_type):
    build_zarr_array(
        root=zarr_group, data=BASE_DATA, name="duplicate"
    )  # NB: purely for side effect of writing the data
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Not a single array, but 2 present in ZARR group"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


@pytest.mark.parametrize("target_type", LEGIT_TARGET_TYPES)
def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales__not_directly_under_multiscales(
    zarr_group, target_type
):
    zarr_group.attrs["multiscales"] = zarr_group.attrs["multiscales"][0]["axes"]
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Cannot access axes from attrs; error (KeyError): 'axes'"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


@pytest.mark.parametrize("target_type", LEGIT_TARGET_TYPES)
def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales__not_just_mapped_to_axes(
    zarr_group, target_type
):
    replacement = zarr_group.attrs["multiscales"][0]["axes"]
    del zarr_group.attrs["multiscales"]
    zarr_group.attrs["multiscales"] = {"axes": replacement}
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Cannot access axes from attrs; error (KeyError): 0"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


@pytest.mark.parametrize("target_type", LEGIT_TARGET_TYPES)
def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales_and_mapped_to_axes(
    zarr_group, target_type
):
    replacement = zarr_group.attrs["multiscales"][0]["axes"]
    del zarr_group.attrs["multiscales"]
    zarr_group.attrs["multiscales"] = {0: replacement}
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert (
                        msg
                        == "Cannot access axes from attrs; error (TypeError): list indices must be integers or slices, not str"
                    )
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


def test_parse_golden_path_5D(zarr_group: zarr.Array):
    target_type = CanonicalImageDimensions
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="ok", ok=(data, dims)):
            kwargs = copy.deepcopy(_infer_dataset_creation_arguments(BASE_DATA))
            expected: zarr.Array = zarr.array(BASE_DATA, **kwargs)

            assert dims == target_type(t=5, c=4, z=3, y=2, x=2)
            assert np.all(data[:] == expected[:])
            assert data.name == expected.name
        case unexpected:
            pytest.fail(f"Expected a Result.Ok but got {unexpected}")


def test_parse_golden_path_3D(zarr_path: Path):
    n_ch: int = 4
    y_sz: int = 6
    x_sz: int = 5
    dataset = _build_dataset((n_ch, y_sz, x_sz))
    kwargs = copy.deepcopy(_infer_dataset_creation_arguments(dataset))
    expected: zarr.Array = zarr.array(dataset, **kwargs)

    zarr_group: zarr.Group = prepare_zarr_group(
        path=zarr_path,
        data=dataset,
        axis_metadata=[{"name": "c", "type": "channel"}]
        + [_create_space_axis_map(a) for a in ["y", "x"]],
    )
    target_type = DimensionsForIlluminationCorrectionScaling
    match parse_single_array_and_dimensions_from_zarr_group(
        group=zarr_group, target_type=target_type
    ):
        case result.Result(tag="ok", ok=(data, dims)):
            assert dims == target_type(c=n_ch, y=y_sz, x=x_sz)
            assert np.all(data[:] == expected[:])
            assert data.name == expected.name
        case unexpected:
            pytest.fail(f"Expected a Result.Ok but got {unexpected}")


@pytest.mark.parametrize(
    ("kwargs", "exp_err_msg"),
    [
        ({"c": -1, "y": 4, "x": 3}, "'c' must be > 0: -1"),
        ({"c": 1, "y": -2, "x": 3}, "'y' must be > 0: -2"),
        ({"c": 2, "y": 4, "x": -4}, "'x' must be > 0: -4"),
        ({"c": 0, "y": 4, "x": 3}, "'c' must be > 0: 0"),
        ({"c": 1, "y": 0, "x": 3}, "'y' must be > 0: 0"),
        ({"c": 2, "y": 4, "x": 0}, "'x' must be > 0: 0"),
    ],
)
def test_dimensions_for_scaling__gives_expected_error_when_given_a_nonpositive_argument(
    kwargs, exp_err_msg
):
    with pytest.raises(ValueError) as err_ctx:  # noqa: PT011
        DimensionsForIlluminationCorrectionScaling(**kwargs)
    obs_err_msg: str = str(err_ctx.value)
    assert exp_err_msg == obs_err_msg


@pytest.mark.parametrize(
    ("kwargs", "exp_err_msg"),
    [
        ({"t": -2, "c": 2, "z": 3, "y": 4, "x": 3}, "'t' must be > 0: -2"),
        ({"t": 2, "c": -1, "z": 3, "y": 4, "x": 3}, "'c' must be > 0: -1"),
        ({"t": 2, "c": 1, "z": -3, "y": 4, "x": 3}, "'z' must be > 0: -3"),
        ({"t": 2, "c": 1, "z": 3, "y": -2, "x": 3}, "'y' must be > 0: -2"),
        ({"t": 2, "c": 2, "z": 3, "y": 4, "x": -1}, "'x' must be > 0: -1"),
        ({"t": 0, "c": 2, "z": 3, "y": 4, "x": 3}, "'t' must be > 0: 0"),
        ({"t": 2, "c": 0, "z": 3, "y": 4, "x": 3}, "'c' must be > 0: 0"),
        ({"t": 2, "c": 1, "z": 0, "y": 4, "x": 3}, "'z' must be > 0: 0"),
        ({"t": 2, "c": 1, "z": 3, "y": 0, "x": 3}, "'y' must be > 0: 0"),
        ({"t": 2, "c": 2, "z": 3, "y": 4, "x": 0}, "'x' must be > 0: 0"),
    ],
)
def test_image_dimensions__gives_expected_error_when_given_a_nonpositive_argument(
    kwargs, exp_err_msg
):
    with pytest.raises(ValueError) as err_ctx:  # noqa: PT011
        CanonicalImageDimensions(**kwargs)
    obs_err_msg: str = str(err_ctx.value)
    assert exp_err_msg == obs_err_msg


@st.composite
def gen_weights_dimensions(draw) -> DimensionsForIlluminationCorrectionScaling:
    c: int = draw(st.integers(min_value=1, max_value=4))
    y: int = draw(st.integers(min_value=2, max_value=6))
    x: int = draw(st.integers(min_value=2, max_value=6))
    return DimensionsForIlluminationCorrectionScaling(c=c, y=y, x=x)


@st.composite
def gen_image_dimensions(draw) -> CanonicalImageDimensions:
    t: int = draw(st.integers(min_value=1, max_value=5))
    c: int = draw(st.integers(min_value=1, max_value=4))
    z: int = draw(st.integers(min_value=2, max_value=5))
    y: int = draw(st.integers(min_value=2, max_value=6))
    x: int = draw(st.integers(min_value=2, max_value=6))
    return CanonicalImageDimensions(t=t, c=c, z=z, y=y, x=x)


def gen_random_rank_array(data_type: type, **kwargs) -> SearchStrategy[np.ndarray]:
    return (
        st.integers(min_value=1, max_value=5)
        .flatmap(
            lambda rank: st.lists(
                st.integers(min_value=1, max_value=5), min_size=rank, max_size=rank
            )
        )
        .map(tuple)
        .flatmap(lambda shape: hyp_npy.arrays(dtype=data_type, shape=shape, **kwargs))
    )


def gen_image(dim: CanonicalImageDimensions) -> SearchStrategy[np.ndarray]:
    return hyp_npy.arrays(
        dtype=IMAGE_DATA_TYPE,
        shape=(dim.t, dim.c, dim.z, dim.y, dim.x),
    )


def gen_weights(dim: DimensionsForIlluminationCorrectionScaling) -> SearchStrategy[np.ndarray]:
    return gen_reasonable_weights((dim.c, dim.y, dim.x))


def gen_small_image_and_dimensions() -> tuple[np.ndarray, CanonicalImageDimensions]:
    return gen_image_dimensions().flatmap(lambda dim: gen_image(dim).map(lambda img: (img, dim)))


def gen_small_weights_and_dimensions() -> (
    tuple[np.ndarray, DimensionsForIlluminationCorrectionScaling]
):
    return gen_weights_dimensions().flatmap(
        lambda dim: gen_weights(dim).map(lambda wts: (wts, dim))
    )


def gen_reasonable_weights(shape, **kwargs) -> SearchStrategy[np.ndarray]:
    params = {
        "min_value": np.finfo(
            WEIGHTS_DATA_TYPE
        ).tiny,  # Weights must be positive, and prevent divisition by 0.
        "max_value": 5.0,  # The upweighting of a pixel will rarely, if ever, exceed this factor.
        "allow_nan": False,  # Prevent odd behavior when equality and proximity testing.
        "allow_infinity": False,  # Prevent odd behavior when equality and proximity testing.
    }
    params.update(kwargs)
    return hyp_npy.arrays(shape=shape, dtype=WEIGHTS_DATA_TYPE, elements=st.floats(**params))


@st.composite
def gen_legal_channel_correction_inputs(
    draw,
) -> tuple[
    np.ndarray, CanonicalImageDimensions, np.ndarray, DimensionsForIlluminationCorrectionScaling
]:
    img: np.ndarray
    dim_img: CanonicalImageDimensions
    img, dim_img = draw(gen_small_image_and_dimensions())
    dim_wts: DimensionsForIlluminationCorrectionScaling = (
        DimensionsForIlluminationCorrectionScaling(
            c=dim_img.c,
            y=dim_img.y,
            x=dim_img.x,
        )
    )
    wts: np.ndarray = draw(gen_reasonable_weights((dim_wts.c, dim_wts.y, dim_wts.x)))
    return img, dim_img, wts, dim_wts


@given(
    wts_dim_channel=gen_small_weights_and_dimensions().flatmap(
        lambda pair: st.integers(min_value=0, max_value=pair[1].c - 1).map(
            lambda ch_req: (*pair, ch_req)
        )
    )
)
def test_channel_extraction_for_weights__gives_expected_value_when_given_legal_input(
    wts_dim_channel,
):
    wts: np.ndarray
    dim: DimensionsForIlluminationCorrectionScaling
    ch: int
    wts, dim, ch = wts_dim_channel
    match dim.get_channel_data(channel=ch, array=wts):
        case result.Result(tag="ok", ok=data):
            assert np.allclose(data, wts[ch, :, :])
        case result.Result(tag="error", error=e):
            pytest.fail(f"Expected a Result.Ok, but got a Result.Error: {e}")
        case unknown:
            pytest.fail(f"Expected a Result, but got a {type(unknown).__name__}")


@given(inputs=gen_legal_channel_correction_inputs())
def test_compute_corrected_channels__gives_back_a_list_of_length_equal_to_number_of_channels(
    inputs,
):
    img, dim_img, wts, dim_wts = inputs
    match compute_corrected_channels(
        image=img, image_dimensions=dim_img, weights=wts, weight_dimensions=dim_wts
    ):
        case result.Result(tag="ok", ok=array_per_channel):
            assert len(array_per_channel) == dim_wts.c
        case result.Result(tag="error", error=messages):
            pytest.fail(f"Expected an Ok, but got an Error: {messages}")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value, but got a {type(unknown).__name__}")


@given(inputs=gen_legal_channel_correction_inputs())
def test_compute_corrected_channels__gives_back_a_list_in_which_each_element_has_correct_shape(
    inputs,
):
    img, dim_img, wts, dim_wts = inputs
    match compute_corrected_channels(
        image=img, image_dimensions=dim_img, weights=wts, weight_dimensions=dim_wts
    ):
        case result.Result(tag="ok", ok=array_per_channel):
            # The full time series of voxels should be extracted for each channel, and voxel size must remain the
            # same, so therefore each channel's array should be of rank 4, with the first dimension (time) equaling
            # the original full image's time dimension, and then the same voxel size as previously.
            exp_shape: tuple[int, int, int, int] = (dim_img.t, dim_img.z, dim_img.y, dim_img.x)
            for ch, arr in enumerate(array_per_channel):
                assert (
                    arr.shape == exp_shape
                ), f"For channel {ch}, expected shape of {exp_shape} but got {arr.shape}"
        case result.Result(tag="error", error=messages):
            pytest.fail(f"Expected an Ok, but got an Error: {messages}")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value, but got a {type(unknown).__name__}")


@given(inputs=gen_legal_channel_correction_inputs())
# NB: Here, we restrict the domain of pixel values to what's reasonable, to avoid equality testing issues with clipping.
def test_compute_corrected_channels__gives_the_correct_answer_when_given_legal_inputs(inputs):
    img, dim_img, wts, dim_wts = inputs
    max_val: int = np.iinfo(IMAGE_DATA_TYPE).max
    get_exp_elm: Callable[[float], int] = lambda x: min(round(x), max_val)
    match compute_corrected_channels(
        image=img, image_dimensions=dim_img, weights=wts, weight_dimensions=dim_wts
    ):
        case result.Result(tag="ok", ok=array_per_channel):
            for ch, scaled_array in enumerate(array_per_channel):
                expected_array: np.ndarray = np.vectorize(get_exp_elm)(
                    img[:, ch, :, :, :] * wts[ch, :, :]
                )
                assert np.allclose(scaled_array, expected_array)
        case result.Result(tag="error", error=messages):
            pytest.fail(f"Expected an Ok, but got an Error: {messages}")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value, but got a {type(unknown).__name__}")


@given(
    wts_and_dim=gen_small_weights_and_dimensions(),
    img_and_dim=gen_image_dimensions().flatmap(
        lambda dim: gen_random_rank_array(data_type=IMAGE_DATA_TYPE)
        .filter(lambda img: img.ndim != dim.rank)
        .map(lambda img: (img, dim))
    ),
)
def test_compute_corrected_channels__requires_correct_image_rank(img_and_dim, wts_and_dim):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    img, img_dim = img_and_dim
    wts: np.ndarray
    wts_dim: DimensionsForIlluminationCorrectionScaling
    wts, wts_dim = wts_and_dim
    match compute_corrected_channels(
        image=img, image_dimensions=img_dim, weights=wts, weight_dimensions=wts_dim
    ):
        case result.Result(tag="error", error=messages):
            assert messages == [
                f"Image is of rank {img.ndim}, but dimensions are of rank {img_dim.rank}"
            ]
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    wts_and_dim=gen_weights_dimensions().flatmap(
        lambda dim: gen_random_rank_array(data_type=WEIGHTS_DATA_TYPE)
        .filter(lambda wts: wts.ndim != dim.rank)
        .map(lambda wts: (wts, dim))
    ),
    img_and_dim=gen_small_image_and_dimensions(),
)
def test_compute_corrected_channels__requires_correct_weights_rank(img_and_dim, wts_and_dim):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    img, img_dim = img_and_dim
    wts: np.ndarray
    wts_dim: DimensionsForIlluminationCorrectionScaling
    wts, wts_dim = wts_and_dim
    match compute_corrected_channels(
        image=img, image_dimensions=img_dim, weights=wts, weight_dimensions=wts_dim
    ):
        case result.Result(tag="error", error=messages):
            assert messages == [
                f"Weights are of rank {wts.ndim}, but dimensions are of rank {wts_dim.rank}"
            ]
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    inputs=st.tuples(gen_image_dimensions(), gen_weights_dimensions())
    .filter(lambda pair: pair[0].c != pair[1].c)
    .flatmap(
        lambda dims_pair: st.tuples(gen_image(dims_pair[0]), gen_weights(dims_pair[1])).map(
            lambda arrs_pair: (arrs_pair[0], dims_pair[0], arrs_pair[1], dims_pair[1])
        )
    )
)
def test_compute_corrected_channels__correctly_enforces_agreement_between_number_of_channels(
    inputs,
):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    wts: np.ndarray
    wts_dim: DimensionsForIlluminationCorrectionScaling
    img, img_dim, wts, wts_dim = inputs
    match compute_corrected_channels(
        image=img, image_dimensions=img_dim, weights=wts, weight_dimensions=wts_dim
    ):
        case result.Result(tag="error", error=messages):
            assert messages == [
                f"Image's dimensions indicate {img_dim.c} channel(s) while weights' dimensions indicate {wts_dim.c} channel(s)"
            ]
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    img_dim_ch=gen_small_image_and_dimensions().flatmap(
        lambda img_dim: st.integers(min_value=0, max_value=snd(img_dim).c - 1).map(
            lambda ch: (*img_dim, ch)
        )
    )
)
def test_get_channel_data__always_reduces_rank_by_exactly_one(img_dim_ch):
    img: np.ndarray
    dim: CanonicalImageDimensions
    img, dim, ch = img_dim_ch
    match dim.get_channel_data(channel=ch, array=img):
        case result.Result(tag="ok", ok=obs):
            assert obs.ndim == img.ndim - 1
        case result.Result(tag="error", error=e):
            pytest.fail(f"Expected a Result.Ok but got a Result.Error: {e}")
        case unknown:
            pytest.fail(
                f"Expected a Result-wrapped value but got a {type(unknown).__name__}: {unknown}"
            )


gen_wavelength: SearchStrategy[None | float] = st.one_of(
    st.none(), st.floats(min_value=380, max_value=700)
)
gen_channel_index: SearchStrategy[int] = st.integers(min_value=0)


def gen_not_type(excl: type | tuple[type, ...]) -> SearchStrategy[object]:
    return st.from_type(type).flatmap(st.from_type).filter(lambda x: not isinstance(x, excl))


@st.composite
def gen_channel_meta(
    draw,
    *,
    get_gen_name: Callable[[], SearchStrategy[str]] = st.text,
    get_gen_index: Callable[[], SearchStrategy[int]] = lambda: gen_channel_index,
    get_gen_wavelength_opt: Callable[[], SearchStrategy[Optional[float]]] = lambda: st.one_of(
        st.none(), st.floats(min_value=380, max_value=700)
    ),
) -> ChannelMeta:
    name: str = draw(get_gen_name())
    index: int = draw(get_gen_index())
    emission_opt: Optional[float] = draw(get_gen_wavelength_opt())
    excitation_opt: Optional[float] = draw(get_gen_wavelength_opt())
    return ChannelMeta(
        name=name, index=index, emissionLambdaNm=emission_opt, excitationLambdaNm=excitation_opt
    )


@st.composite
def gen_valid_channels_arguments(draw) -> tuple[int, tuple[ChannelMeta, ...]]:
    ch_names: list[str] = draw(st.lists(st.text(), unique=True, min_size=1))
    n_ch: int = len(ch_names)
    emissions: list[None | float] = draw(st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch))
    excitations: list[None | float] = draw(st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch))
    metas: list[ChannelMeta] = [
        ChannelMeta(name=name, index=i, emissionLambdaNm=emi, excitationLambdaNm=exc)
        for i, (name, emi, exc) in enumerate(zip(ch_names, emissions, excitations, strict=True))
    ]
    return n_ch, tuple(metas)


@given(
    args=st.tuples(
        gen_not_type(str),
        gen_channel_index,
        gen_wavelength,
        gen_wavelength,
    )
)
def test_channel_meta_name__must_be_string(
    args: tuple[object, int, None | float, None | float],
) -> None:
    not_name: object
    index: int
    excitation: None | float
    emission: None | float
    not_name, index, excitation, emission = args
    with pytest.raises(TypeError):
        ChannelMeta(
            name=not_name, index=index, excitationLambdaNm=excitation, emissionLambdaNm=emission
        )
    # NB: no check here on the error message since it's seemingly very difficult to check substrings for this one.


@given(
    args_and_error=st.tuples(
        st.text(),
        st.one_of(
            st.integers(max_value=-1).map(lambda i: (i, ValueError(f"'index' must be >= 0: {i}"))),
            gen_not_type(int).map(
                lambda obj: (obj, TypeError(""))
            ),  # NB: no real check here since the messages can be weird.
        ),
        gen_wavelength,
        gen_wavelength,
    )
)
def test_channel_meta_index__must_be_nonnegative_integer(
    args_and_error: tuple[str, tuple[object, TypeError | ValueError], None | float, None | float],
) -> None:
    name: str
    not_index: object
    exp_err: TypeError | ValueError
    excitation: None | float
    emission: None | float
    name, (not_index, exp_err), excitation, emission = args_and_error
    with pytest.raises(type(exp_err)) as err_ctx:
        ChannelMeta(
            name=name, index=not_index, excitationLambdaNm=excitation, emissionLambdaNm=emission
        )
    assert str(err_ctx.value).startswith(str(exp_err))


@given(
    args_and_error=st.tuples(
        st.text(),
        gen_channel_index,
        st.one_of(
            gen_not_type((float, type(None))).map(
                lambda obj: (
                    obj,
                    TypeError(
                        f"emissionLambdaNm is neither null nor float, but {type(obj).__name__}"
                    ),
                )
            ),
            st.floats(max_value=0).map(
                lambda y: (y, ValueError(f"emissionLambdaNm must be strictly positive; got {y}"))
            ),
        ),
        gen_wavelength,
    )
)
def test_channel_meta_emission_wavelength__must_be_null_or_positive_float(
    args_and_error: tuple[str, int, tuple[object, TypeError | ValueError], None | float],
) -> None:
    name: str
    index: int
    not_emission: object
    excitation: None | float
    exp_err: TypeError | ValueError
    name, index, (not_emission, exp_err), excitation = args_and_error
    with pytest.raises(type(exp_err)) as err_ctx:
        ChannelMeta(
            name=name, index=index, emissionLambdaNm=not_emission, excitationLambdaNm=excitation
        )
    assert str(err_ctx.value) == str(exp_err)


@given(
    args_and_error=st.tuples(
        st.text(),
        gen_channel_index,
        gen_wavelength,
        st.one_of(
            gen_not_type((float, type(None))).map(
                lambda obj: (
                    obj,
                    TypeError(
                        f"excitationLambdaNm is neither null nor float, but {type(obj).__name__}"
                    ),
                )
            ),
            st.floats(max_value=0).map(
                lambda y: (y, ValueError(f"excitationLambdaNm must be strictly positive; got {y}"))
            ),
        ),
    )
)
def test_channel_meta_excitation_wavelength__must_be_null_or_positive_float(
    args_and_error: tuple[str, int, None | float, tuple[object, TypeError | ValueError]],
) -> None:
    name: str
    index: int
    emission: object
    not_excitation: None | float
    exp_err: TypeError | ValueError
    name, index, emission, (not_excitation, exp_err) = args_and_error
    with pytest.raises(type(exp_err)) as err_ctx:
        ChannelMeta(
            name=name, index=index, emissionLambdaNm=emission, excitationLambdaNm=not_excitation
        )
    assert str(err_ctx.value) == str(exp_err)


@given(
    args_and_error=st.tuples(
        st.one_of(
            gen_not_type(int).map(
                lambda obj: (obj, "")
            ),  # NB: no real check here since the messages can be weird.
            st.integers(max_value=0).map(
                lambda n_ch: (
                    n_ch,
                    f"Cannot build Channels instance; error (ValueError): 'count' must be > 0: {n_ch}",
                )
            ),
        ),
        st.lists(gen_channel_meta()).map(tuple),
    )
)
def test_channels_channel_count__must_be_positive_integer(
    args_and_error: tuple[tuple[object, str], Iterable[ChannelMeta]],
) -> None:
    not_count: object
    values: Iterable[ChannelMeta]
    expected_prefix: str
    (not_count, expected_prefix), values = args_and_error
    match Channels.from_count_and_metas(not_count, values):
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got Result.Ok")
        case result.Result(tag="error", error=obs_msg):
            assert obs_msg.startswith(expected_prefix)
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    args_and_error=st.lists(gen_channel_meta())
    .map(tuple)
    .flatmap(
        lambda metas: st.integers(min_value=max(len(metas), 0))
        .filter(lambda count: count != len(metas))
        .map(lambda count: (count, metas))
    )
    .map(
        lambda count_metas_pair: (
            *count_metas_pair,
            f"Got {len(snd(count_metas_pair))} channel(s) (from attribute values) for a Channels instance claiming to have {fst(count_metas_pair)} channels",
        )
    )
)
def test_channels_channel_count__must_match_channels_length(
    args_and_error: tuple[int, tuple[ChannelMeta, ...], str],
):
    n_ch: int
    metas: tuple[ChannelMeta, ...]
    exp_err_msg: str
    n_ch, metas, exp_err_msg = args_and_error
    with pytest.raises(ValueError) as err_ctx:  # noqa: PT011
        Channels(count=n_ch, values=metas)
    assert str(err_ctx.value) == exp_err_msg


@given(
    args_and_error=st.lists(gen_channel_meta(), min_size=1)
    .map(tuple)
    .map(lambda metas: (len(metas), metas))
    .flatmap(
        lambda count_metas_pair: st.lists(st.from_type(type).flatmap(st.from_type), min_size=1)
        .map(tuple)
        .flatmap(
            lambda non_metas: st.permutations(snd(count_metas_pair) + non_metas).map(
                lambda shuffled: (
                    fst(count_metas_pair),
                    tuple(shuffled),
                    f"{len(non_metas)} non-channel value(s) for attribute values",
                )
            )
        )
    )
)
def test_channels_values__must_each_be_channel(
    args_and_error: tuple[int, tuple[ChannelMeta, ...], str],
):
    n_ch: int
    not_all_metas: tuple[ChannelMeta, ...]
    exp_err_msg: str
    n_ch, not_all_metas, exp_err_msg = args_and_error
    with pytest.raises(TypeError) as err_ctx:
        Channels(count=n_ch, values=not_all_metas)
    assert str(err_ctx.value) == exp_err_msg


@given(st.data())
def test_channels_values__must_have_correct_indices(data):
    n_ch: int = data.draw(st.integers(min_value=1, max_value=10))
    names: list[str] = data.draw(st.lists(st.text(), min_size=n_ch, max_size=n_ch, unique=True))
    not_indices: list[int] = data.draw(
        st.lists(st.integers(min_value=0), min_size=n_ch, max_size=n_ch).filter(
            lambda indices: set(indices) != set(range(n_ch))
        )
    )
    emissions: list[None | float] = data.draw(
        st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch)
    )
    excitations: list[None | float] = data.draw(
        st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch)
    )
    metas: list[ChannelMeta] = [
        ChannelMeta(name=name, index=index, emissionLambdaNm=emi, excitationLambdaNm=exc)
        for name, index, emi, exc in zip(names, not_indices, emissions, excitations, strict=True)
    ]
    with pytest.raises(ValueError) as err_ctx:  # noqa: PT011
        Channels(count=n_ch, values=tuple(metas))
    assert str(err_ctx.value).startswith("For attribute values, indices don't match expectation")


@given(st.data())
def test_channels_values__must_have_unique_names(data):
    n_ch: int = data.draw(st.integers(min_value=2, max_value=10))
    not_names: list[str] = data.draw(st.lists(st.text(), min_size=n_ch - 1, max_size=n_ch - 1))
    repeat: str = data.draw(st.sampled_from(not_names))
    emissions: list[None | float] = data.draw(
        st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch)
    )
    excitations: list[None | float] = data.draw(
        st.lists(gen_wavelength, min_size=n_ch, max_size=n_ch)
    )
    metas: list[ChannelMeta] = [
        ChannelMeta(name=name, index=index, emissionLambdaNm=emi, excitationLambdaNm=exc)
        for name, index, emi, exc in zip(
            list(random.sample([*not_names, repeat], n_ch)),
            range(n_ch),
            emissions,
            excitations,
            strict=True,
        )
    ]
    with pytest.raises(ValueError) as err_ctx:  # noqa: PT011
        Channels(count=n_ch, values=tuple(metas))
    assert str(err_ctx.value).startswith("For attribute values, there are repeated channel name(s)")


@given(args=gen_valid_channels_arguments())
def test_channels_values__must_be_tuple(args: tuple[int, tuple[ChannelMeta, ...]]):
    n_ch: int
    chs: tuple[ChannelMeta, ...]
    n_ch, chs = args
    with pytest.raises(TypeError):
        Channels(count=n_ch, values=list(chs))
    # NB: no check here on the error message since it's seemingly very difficult to check substrings for this one.
