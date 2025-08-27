"""Tests for ZARR-related utilities"""

import copy
import functools
import json
import random
from collections.abc import Callable, Collection, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar

import attrs
import numpy as np
import pytest
import zarr  # type: ignore[import-untyped]
from expression import Result, fst, result, snd
from gertils import ExtantFolder
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyp_npy
from hypothesis.strategies import SearchStrategy
from ome_zarr.axes import KNOWN_AXES as OME_AXIS_TYPES  # type: ignore[import-untyped]
from ome_zarr.io import ZarrLocation  # type: ignore[import-untyped]

from illumifix.zarr_tools import (
    CHANNEL_COUNT_KEY,
    ArrayWithDimensions,
    CanonicalImageDimensions,
    ChannelKey,
    ChannelMeta,
    Channels,
    OmeZarrAxis,
    WaveLenOpt,
    compute_corrected_channels,
    create_zgroup_file,
    parse_channels_from_flattened_mapping_with_count,
    parse_channels_from_zarr,
    parse_single_array_and_dimensions_from_zarr_group,
)

DEFAULT_DATA_NAME = "test_zarr_data"
DEFAULT_DATA_TYPE = "u2"
IMAGE_DATA_TYPE = np.uint16
MAX_NUM_CHANNELS: int = 4
WEIGHTS_DATA_TYPE = np.float64


def _build_dataset(dims: tuple[int, ...], dtype: str = DEFAULT_DATA_TYPE) -> np.ndarray:
    return np.arange(
        functools.reduce(lambda acc, n: acc * n, dims, 1),
        dtype=dtype,
    ).reshape(dims)


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
def zarr_path(tmp_path: Path) -> Path:
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


def _wrap_parse_error_message(msg: str) -> Result[ArrayWithDimensions, str]:
    return Result.Error(msg)


def test_zarr_axis__has_length_five():
    assert len(OmeZarrAxis) == 5  # noqa: PLR2004


@pytest.mark.parametrize("axis", OmeZarrAxis)
def test_zarr_axis__name_is_lowercase(axis: OmeZarrAxis):
    assert axis.name == axis.name.lower()


@pytest.mark.parametrize("axis", OmeZarrAxis)
def test_zarr_axis__typename_corresponds_to_ome_zarr(axis: OmeZarrAxis):
    assert axis.typename == OME_AXIS_TYPES[axis.name]


def test_group_argument_type_other_than_zarr_group_gives_attribute_error():
    observed = parse_single_array_and_dimensions_from_zarr_group(group=BASE_DATA)
    expected = _wrap_parse_error_message("Alleged zarr.Group instance lacks .attrs member")
    assert observed == expected


def test_zarr_group_must_have_only_one_array(zarr_group):
    build_zarr_array(
        root=zarr_group, data=BASE_DATA, name="duplicate"
    )  # NB: purely for side effect of writing the data
    match parse_single_array_and_dimensions_from_zarr_group(group=zarr_group):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Not a single array, but 2 present in ZARR group"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales__not_directly_under_multiscales(
    zarr_group,
):
    zarr_group.attrs["multiscales"] = zarr_group.attrs["multiscales"][0]["axes"]
    match parse_single_array_and_dimensions_from_zarr_group(group=zarr_group):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Cannot access axes from attrs; error (KeyError): 'axes'"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales__not_just_mapped_to_axes(
    zarr_group,
):
    replacement = zarr_group.attrs["multiscales"][0]["axes"]
    del zarr_group.attrs["multiscales"]
    zarr_group.attrs["multiscales"] = {"axes": replacement}
    match parse_single_array_and_dimensions_from_zarr_group(group=zarr_group):
        case result.Result(tag="error", error=msg):
            match msg:
                case str():
                    assert msg == "Cannot access axes from attrs; error (KeyError): 0"
                case _:
                    pytest.fail(f"Expected a string error message but got {type(msg).__name__}")
        case unexpected:
            pytest.fail(f"Expected a Result.Error but got {unexpected}")


def test_zarr_group_must_have_axes_entry_in_first_element_of_multiscales_and_mapped_to_axes(
    zarr_group,
):
    replacement = zarr_group.attrs["multiscales"][0]["axes"]
    del zarr_group.attrs["multiscales"]
    zarr_group.attrs["multiscales"] = {0: replacement}
    match parse_single_array_and_dimensions_from_zarr_group(group=zarr_group):
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


def test_parse_golden_path(zarr_group: zarr.Array):
    match parse_single_array_and_dimensions_from_zarr_group(group=zarr_group):
        case result.Result(tag="ok", ok=(data, dims)):
            kwargs = copy.deepcopy(_infer_dataset_creation_arguments(BASE_DATA))
            expected: zarr.Array = zarr.array(BASE_DATA, **kwargs)

            assert dims == CanonicalImageDimensions(t=5, c=4, z=3, y=2, x=2)
            assert np.all(data[:] == expected[:])
            assert data.name == expected.name
        case unexpected:
            pytest.fail(f"Expected a Result.Ok but got {unexpected}")


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
def gen_weights_dimensions(draw) -> CanonicalImageDimensions:
    c: int = draw(st.integers(min_value=1, max_value=MAX_NUM_CHANNELS))
    y: int = draw(st.integers(min_value=2, max_value=6))
    x: int = draw(st.integers(min_value=2, max_value=6))
    return CanonicalImageDimensions(t=1, c=c, z=1, y=y, x=x)


@st.composite
def gen_image_dimensions(draw) -> CanonicalImageDimensions:
    t: int = draw(st.integers(min_value=1, max_value=5))
    c: int = draw(st.integers(min_value=1, max_value=MAX_NUM_CHANNELS))
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


def gen_weights(dim: CanonicalImageDimensions) -> SearchStrategy[np.ndarray]:
    return gen_reasonable_weights((1, dim.c, 1, dim.y, dim.x))


def gen_small_image_and_dimensions() -> SearchStrategy[tuple[np.ndarray, CanonicalImageDimensions]]:
    return gen_image_dimensions().flatmap(lambda dim: gen_image(dim).map(lambda img: (img, dim)))


def gen_small_weights_and_dimensions() -> (
    SearchStrategy[tuple[np.ndarray, CanonicalImageDimensions]]
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
    return hyp_npy.arrays(shape=shape, dtype=WEIGHTS_DATA_TYPE, elements=st.floats(**params))  # type: ignore[arg-type]


def _build_channels(chs: Collection[ChannelMeta]) -> Channels:
    return Channels(
        count=len(chs), values=tuple(attrs.evolve(c, index=i) for i, c in enumerate(chs))
    )


def gen_fixed_length_channels(size: int) -> SearchStrategy[list[ChannelMeta]]:
    names: SearchStrategy[list[str]] = st.sets(st.text(), min_size=size, max_size=size).map(list)
    emissions: SearchStrategy[list[WaveLenOpt]] = st.lists(
        gen_wavelength, min_size=size, max_size=size
    )
    excitations: SearchStrategy[list[WaveLenOpt]] = st.lists(
        gen_wavelength, min_size=size, max_size=size
    )
    return st.tuples(names, emissions, excitations).map(
        lambda t: [
            ChannelMeta(name=name, index=i, emissionLambdaNm=emit, excitationLambdaNm=excite)
            for i, (name, emit, excite) in enumerate(zip(t[0], t[1], t[2], strict=True))
        ]
    )


@st.composite
def gen_legal_channel_correction_inputs(
    draw,
) -> tuple[
    np.ndarray, CanonicalImageDimensions, Channels, np.ndarray, CanonicalImageDimensions, Channels
]:
    img: np.ndarray
    dim_img: CanonicalImageDimensions
    img, dim_img = draw(gen_small_image_and_dimensions())
    weights_ch_count: int = draw(st.integers(min_value=dim_img.c, max_value=MAX_NUM_CHANNELS))
    dim_wts: CanonicalImageDimensions = CanonicalImageDimensions(
        t=1,
        c=weights_ch_count,
        z=1,
        y=dim_img.y,
        x=dim_img.x,
    )
    chs_wts: list[ChannelMeta] = draw(gen_fixed_length_channels(weights_ch_count))
    raw_chs_img: set[ChannelMeta] = draw(
        st.sets(st.sampled_from(chs_wts), min_size=dim_img.c, max_size=dim_img.c)
    )
    wts: np.ndarray = draw(gen_reasonable_weights((1, dim_wts.c, 1, dim_wts.y, dim_wts.x)))
    return img, dim_img, _build_channels(raw_chs_img), wts, dim_wts, _build_channels(chs_wts)


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
    dim: CanonicalImageDimensions
    ch: int
    wts, dim, ch = wts_dim_channel
    match dim.get_channel_data(channel=ch, array=wts):
        case result.Result(tag="ok", ok=data):
            assert np.allclose(data, wts[:, ch, :, :, :])
        case result.Result(tag="error", error=e):
            pytest.fail(f"Expected a Result.Ok, but got a Result.Error: {e}")
        case unknown:
            pytest.fail(f"Expected a Result, but got a {type(unknown).__name__}")


@given(inputs=gen_legal_channel_correction_inputs())
def test_compute_corrected_channels__gives_back_a_list_of_length_equal_to_number_of_channels(
    inputs,
):
    img, dim_img, chs_img, wts, dim_wts, chs_wts = inputs
    match compute_corrected_channels(
        image=img,
        image_dimensions=dim_img,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=dim_wts,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="ok", ok=array_per_channel):
            assert len(array_per_channel) == dim_img.c
        case result.Result(tag="error", error=messages):
            pytest.fail(f"Expected an Ok, but got an Error: {messages}")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value, but got a {type(unknown).__name__}")


@given(inputs=gen_legal_channel_correction_inputs())
def test_compute_corrected_channels__gives_back_a_list_in_which_each_element_has_correct_shape(
    inputs,
):
    img, dim_img, chs_img, wts, dim_wts, chs_wts = inputs
    match compute_corrected_channels(
        image=img,
        image_dimensions=dim_img,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=dim_wts,
        weights_channels=chs_wts,
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
    img: np.ndarray
    dim_img: CanonicalImageDimensions
    chs_img: Channels
    wts: np.ndarray
    dim_wts: CanonicalImageDimensions
    chs_wts: Channels
    img, dim_img, chs_img, wts, dim_wts, chs_wts = inputs
    max_val: int = np.iinfo(IMAGE_DATA_TYPE).max
    get_exp_elm: Callable[[float], int] = lambda x: min(round(x), max_val)
    weights_channels_indices: dict[ChannelKey, int] = {
        ch.get_lookup_key(): i for i, ch in enumerate(chs_wts.values)
    }
    match compute_corrected_channels(
        image=img,
        image_dimensions=dim_img,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=dim_wts,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="ok", ok=array_per_channel):
            assert len(array_per_channel) == chs_img.count
            assert len(array_per_channel) == img.shape[1]
            assert len(array_per_channel) == dim_img.c
            for i, ch in enumerate(chs_img.values):
                sub_img = img[:, i, :, :, :]
                sub_wts = wts[:, weights_channels_indices[ch.get_lookup_key()], :, :, :]
                expected_array: np.ndarray = np.vectorize(get_exp_elm)(sub_img * sub_wts)
                scaled_array = array_per_channel[i]
                assert np.allclose(scaled_array, expected_array)
        case result.Result(tag="error", error=messages):
            pytest.fail(f"Expected an Ok, but got an Error: {messages}")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value, but got a {type(unknown).__name__}")


_A = TypeVar("_A")


def _gen_channels_from_tuple_with_dimensions(
    pair: tuple[_A, CanonicalImageDimensions],
) -> SearchStrategy[tuple[_A, CanonicalImageDimensions, Channels]]:
    return gen_fixed_length_channels(snd(pair).c).map(lambda chs: (*pair, _build_channels(chs)))


@given(
    wts_and_dim_and_chs=gen_small_weights_and_dimensions().flatmap(
        _gen_channels_from_tuple_with_dimensions
    ),
    img_and_dim_and_chs=gen_image_dimensions()
    .flatmap(
        lambda dim: gen_random_rank_array(data_type=IMAGE_DATA_TYPE)
        .filter(lambda img: img.ndim != dim.rank)
        .map(lambda img: (img, dim))
    )
    .flatmap(_gen_channels_from_tuple_with_dimensions),
)
def test_compute_corrected_channels__requires_correct_image_rank(
    img_and_dim_and_chs, wts_and_dim_and_chs
):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    chs_img: Channels
    img, img_dim, chs_img = img_and_dim_and_chs
    wts: np.ndarray
    wts_dim: CanonicalImageDimensions
    chs_wts: Channels
    wts, wts_dim, chs_wts = wts_and_dim_and_chs
    match compute_corrected_channels(
        image=img,
        image_dimensions=img_dim,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=wts_dim,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="error", error=messages):
            assert (
                f"Image is of rank {img.ndim}, but dimensions are of rank {img_dim.rank}"
                in messages
            )
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    wts_and_dim_and_chs=gen_weights_dimensions()
    .flatmap(
        lambda dim: gen_random_rank_array(data_type=WEIGHTS_DATA_TYPE)
        .filter(lambda wts: wts.ndim != dim.rank)
        .map(lambda wts: (wts, dim))
    )
    .flatmap(_gen_channels_from_tuple_with_dimensions),
    img_and_dim_and_chs=gen_small_image_and_dimensions().flatmap(
        _gen_channels_from_tuple_with_dimensions
    ),
)
def test_compute_corrected_channels__requires_correct_weights_rank(
    img_and_dim_and_chs, wts_and_dim_and_chs
):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    chs_img: Channels
    img, img_dim, chs_img = img_and_dim_and_chs
    wts: np.ndarray
    wts_dim: CanonicalImageDimensions
    chs_wts: Channels
    wts, wts_dim, chs_wts = wts_and_dim_and_chs
    match compute_corrected_channels(
        image=img,
        image_dimensions=img_dim,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=wts_dim,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="error", error=messages):
            assert (
                f"Weights are of rank {wts.ndim}, but dimensions are of rank {wts_dim.rank}"
                in messages
            )
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    wts_and_dim_and_chs=gen_small_weights_and_dimensions().flatmap(
        _gen_channels_from_tuple_with_dimensions
    ),
    img_and_dim_and_chs=gen_small_image_and_dimensions().flatmap(
        lambda pair: st.integers(min_value=1, max_value=MAX_NUM_CHANNELS)
        .filter(lambda n_ch: n_ch != snd(pair).c)
        .flatmap(gen_fixed_length_channels)
        .map(lambda chs: (*pair, _build_channels(chs)))
    ),
)
def test_compute_corrected_channels__requires_match_between_image_channels_and_corresponding_axis_length(
    img_and_dim_and_chs, wts_and_dim_and_chs
):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    chs_img: Channels
    img, img_dim, chs_img = img_and_dim_and_chs
    wts: np.ndarray
    wts_dim: CanonicalImageDimensions
    chs_wts: Channels
    wts, wts_dim, chs_wts = wts_and_dim_and_chs
    match compute_corrected_channels(
        image=img,
        image_dimensions=img_dim,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=wts_dim,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="error", error=messages):
            assert (
                f"Image channels count is {chs_img.count} but dimensions allege {img_dim.c}"
                in messages
            )
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected a Result.Error but got a Result.Ok")
        case unknown:
            pytest.fail(f"Expected a Result-wrapped value but got a {type(unknown).__name__}")


@given(
    wts_and_dim_and_chs=gen_small_weights_and_dimensions().flatmap(
        lambda pair: st.integers(min_value=1, max_value=MAX_NUM_CHANNELS)
        .filter(lambda n_ch: n_ch != snd(pair).c)
        .flatmap(gen_fixed_length_channels)
        .map(lambda chs: (*pair, _build_channels(chs)))
    ),
    img_and_dim_and_chs=gen_small_image_and_dimensions().flatmap(
        _gen_channels_from_tuple_with_dimensions
    ),
)
def test_compute_corrected_channels__requires_match_between_weights_channels_and_corresponding_axis_length(
    img_and_dim_and_chs, wts_and_dim_and_chs
):
    img: np.ndarray
    img_dim: CanonicalImageDimensions
    chs_img: Channels
    img, img_dim, chs_img = img_and_dim_and_chs
    wts: np.ndarray
    wts_dim: CanonicalImageDimensions
    chs_wts: Channels
    wts, wts_dim, chs_wts = wts_and_dim_and_chs
    match compute_corrected_channels(
        image=img,
        image_dimensions=img_dim,
        image_channels=chs_img,
        weights=wts,
        weights_dimensions=wts_dim,
        weights_channels=chs_wts,
    ):
        case result.Result(tag="error", error=messages):
            assert (
                f"Weights channels count is {chs_wts.count} but dimensions allege {wts_dim.c}"
                in messages
            )
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


gen_wavelength: SearchStrategy[WaveLenOpt] = st.one_of(
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
    get_gen_wavelength_opt: Callable[[], SearchStrategy[WaveLenOpt]] = lambda: st.one_of(
        st.none(), st.floats(min_value=380, max_value=700)
    ),
) -> ChannelMeta:
    name: str = draw(get_gen_name())
    index: int = draw(get_gen_index())
    emission_opt: WaveLenOpt = draw(get_gen_wavelength_opt())
    excitation_opt: WaveLenOpt = draw(get_gen_wavelength_opt())
    return ChannelMeta(
        name=name, index=index, emissionLambdaNm=emission_opt, excitationLambdaNm=excitation_opt
    )


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
            name=not_name,  # type: ignore[arg-type]
            index=index,
            excitationLambdaNm=excitation,
            emissionLambdaNm=emission,
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
            name=name,
            index=not_index,  # type: ignore[arg-type]
            excitationLambdaNm=excitation,
            emissionLambdaNm=emission,
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
            name=name,
            index=index,
            emissionLambdaNm=not_emission,  # type: ignore[arg-type]
            excitationLambdaNm=excitation,
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
    name, index, emission, (not_excitation, exp_err) = args_and_error  # type: ignore[assignment]
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
    match Channels.from_count_and_metas(not_count, values):  # type: ignore[arg-type]
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


_ParseInput = TypeVar("_ParseInput", Mapping[str, object], ExtantFolder, Path, ZarrLocation)


def _prep_parse_channels_zarr(
    zarr_root: Path, metadata: Mapping[str, object], *, nest_metadata_under_key: bool = True
) -> Path:
    metadata_path: Path = zarr_root / ".zattrs"
    with metadata_path.open(mode="w") as metadata_file:
        json.dump(
            {"metadata": metadata} if nest_metadata_under_key else metadata, metadata_file, indent=2
        )
    create_zgroup_file(root=zarr_root)
    return zarr_root


parse_channels_from_zarr_with_flattened_mapping_with_count = partial(
    parse_channels_from_zarr,
    parse_channels=parse_channels_from_flattened_mapping_with_count,
)


def get_parse_channels__parse_prep_pairs(*, nest_metadata_under_key: bool = True):
    prep_data = partial(_prep_parse_channels_zarr, nest_metadata_under_key=nest_metadata_under_key)
    return [
        (
            parse_channels_from_flattened_mapping_with_count,
            lambda _, meta: {"metadata": meta} if nest_metadata_under_key else meta,
        ),
        (
            parse_channels_from_zarr_with_flattened_mapping_with_count,
            prep_data,
        ),
        (
            parse_channels_from_zarr_with_flattened_mapping_with_count,
            lambda tmp, meta: ExtantFolder(prep_data(tmp, meta)),
        ),
        (
            parse_channels_from_zarr_with_flattened_mapping_with_count,
            lambda tmp, meta: ZarrLocation(prep_data(tmp, meta)),
        ),
    ]


CANONICAL_CHANNEL_DATA: list[Mapping[str, None | float | str]] = [
    {"emissionLambdaNm": 700.5, "excitationLambdaNm": None, "name": "Far Red"},
    {"emissionLambdaNm": 535.0, "excitationLambdaNm": 488.0, "name": "GFP"},
    {"emissionLambdaNm": 450.5, "excitationLambdaNm": 365.0, "name": "DAPI"},
    {"emissionLambdaNm": 610.5, "excitationLambdaNm": 560.0, "name": "Red"},
]


CANONICAL_METADATA: Mapping[str, object] = {
    "axis_sizes": {"C": 4, "X": 2048, "Y": 2044, "Z": 38},
    CHANNEL_COUNT_KEY: 4,
    **{f"channel_{i}": ch_md for i, ch_md in enumerate(CANONICAL_CHANNEL_DATA)},
    "microscope": {
        "immersionRefractiveIndex": 1.515,
        "modalityFlags": ["fluorescence", "camera"],
        "objectiveMagnification": 60.0,
        "objectiveName": "Plan Apo λ 60x Oil",
        "objectiveNumericalAperture": 1.4,
        "zoomMagnification": 1.0,
    },
}


@pytest.mark.parametrize(("parse", "prepare"), get_parse_channels__parse_prep_pairs())
def test_parse_channels__good_example_data(
    tmp_path: Path,
    parse: Callable[[_ParseInput], Result[Channels, list[str]]],
    prepare: Callable[[Path, Mapping[str, object]], _ParseInput],
) -> None:
    exp: Channels = Channels(
        count=4,
        values=tuple(
            ChannelMeta(index=i, **ch_md)  # type: ignore[arg-type]
            for i, ch_md in enumerate(CANONICAL_CHANNEL_DATA)
        ),
    )
    input_data: _ParseInput = prepare(tmp_path, CANONICAL_METADATA)
    match parse(input_data):
        case result.Result(tag="error", error=messages):
            pytest.fail(
                f"Expected parse success but got a failure, with {len(messages)} message(s): {'; '.join(messages)}"
            )
        case result.Result(tag="ok", ok=obs):
            assert obs == exp
        case unknown:
            pytest.fail(
                f"Expected a result.Result-wrapped value but got a {type(unknown).__name__}"
            )


@pytest.mark.parametrize(("parse", "prepare"), get_parse_channels__parse_prep_pairs())
def test_parse_channels__must_have_channel_count(
    tmp_path: Path,
    parse: Callable[[_ParseInput], Result[Channels, list[str]]],
    prepare: Callable[[Path, Mapping[str, object]], _ParseInput],
) -> None:
    input_data: _ParseInput = prepare(
        tmp_path, {k: v for k, v in CANONICAL_METADATA.items() if k != CHANNEL_COUNT_KEY}
    )
    match parse(input_data):
        case result.Result(tag="error", error=messages):
            assert messages == [f"Missing the key ({CHANNEL_COUNT_KEY}) for number of channels"]
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected parse failure but got a success")
        case unknown:
            pytest.fail(
                f"Expected a result.Result-wrapped value but got a {type(unknown).__name__}"
            )


def test_parse_channels_from_zarr__prohibits_path_as_raw_string(tmp_path: Path) -> None:
    match parse_channels_from_zarr_with_flattened_mapping_with_count(str(tmp_path)):
        case result.Result(tag="error", error=messages):
            assert messages == ["Cannot parse channels from value of type str"]
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected parse failure but got a success")
        case unknown:
            pytest.fail(
                f"Expected a result.Result-wrapped value but got a {type(unknown).__name__}"
            )


@pytest.mark.parametrize(
    ("parse", "prepare"),
    [
        (parse, prep)
        for parse, prep in get_parse_channels__parse_prep_pairs(nest_metadata_under_key=False)
        if parse is parse_channels_from_zarr_with_flattened_mapping_with_count
    ],
)
def test_parse_channels__fails_when_metadata_are_not_under_proper_key(
    tmp_path: Path,
    parse: Callable[[_ParseInput], Result[Channels, list[str]]],
    prepare: Callable[[Path, Mapping[str, object]], _ParseInput],
) -> None:
    input_data: _ParseInput = prepare(tmp_path, CANONICAL_METADATA)
    match parse(input_data):
        case result.Result(tag="error", error=messages):
            assert len(messages) == 1
            assert messages[0].startswith("Missing metadata key in given mapping")
        case result.Result(tag="ok", ok=_):
            pytest.fail("Expected parse failure but got a success")
        case unknown:
            pytest.fail(
                f"Expected a result.Result-wrapped value but got a {type(unknown).__name__}"
            )
