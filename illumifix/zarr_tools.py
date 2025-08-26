"""Helpers for working with data stored in ZARR"""

import copy
from enum import Enum
from math import floor
import re
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol, TypeAlias

import attrs
import numpy as np
from ome_zarr.io import ZarrLocation
import zarr
import zarr.attrs
from expression import Option, Result, fst, result, snd
from expression.collections import TypedArray
from gertils import ExtantFolder

from illumifix.expression_utilities import sequence_accumulate_errors, traverse_accumulate_errors


CHANNEL_COUNT_KEY: Literal["channelCount"] = "channelCount"

ArrayLike: TypeAlias = np.ndarray | zarr.Array
ChannelIndex: TypeAlias = int
ChannelKey: TypeAlias = tuple["ChannelName", "WaveLenOpt", "WaveLenOpt"]
ChannelName: TypeAlias = str
Wavelength: TypeAlias = float
WaveLenOpt: TypeAlias = Optional[Wavelength]


@attrs.define(frozen=True)
class _ZarrAxis:
    name: str
    typename: str


class ZarrAxis(Enum):
    T = _ZarrAxis("t", "time")
    C = _ZarrAxis("c", "channel")
    Z = _ZarrAxis("z", "space")
    Y = _ZarrAxis("y", "space")
    X = _ZarrAxis("x", "space")
    
    @property
    def name(self) -> str:
        return self.value.name
    
    @property
    def typename(self) -> str:
        return self.value.typename


_CHECK_POSITIVE_INT = [attrs.validators.instance_of(int), attrs.validators.gt(0)]

_ATTR_AXIS_PAIRS: OrderedDict[str, ZarrAxis] = OrderedDict((ax.name, ax) for ax in ZarrAxis)


def _check_null_or_positive_float(_, attr: attrs.Attribute, value: object) -> None:
    match value:
        case None:
            return
        case float():
            if value <= 0:
                raise ValueError(f"{attr.name} must be strictly positive; got {value}")
        case _:
            raise TypeError(f"{attr.name} is neither null nor float, but {type(value).__name__}")


class JsonEncoderForChannelMeta(json.JSONEncoder):
    """Support JSON encoding of ChannelMeta instances."""

    def default(self, o):
        return attrs.asdict(o) if isinstance(o, ChannelMeta) else super().default(o)


@attrs.define(kw_only=True, frozen=True, eq=True)
class ChannelMeta:
    name = attrs.field(validator=attrs.validators.instance_of(str))  # type: ChannelName
    index = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)])  # type: ChannelIndex
    emissionLambdaNm = attrs.field(validator=_check_null_or_positive_float)  # type: WaveLenOpt
    excitationLambdaNm = attrs.field(validator=_check_null_or_positive_float)  # type: WaveLenOpt

    def get_lookup_key(self) -> ChannelKey:
        return self.name, self.emissionLambdaNm, self.excitationLambdaNm


def _all_are_channels(_: "Channels", attr: attrs.Attribute, values: tuple[object]) -> None:
    """Check that all values are channel metadata, throwing a TypeError otherwise."""
    match [val for val in values if not isinstance(val, ChannelMeta)]:
        case []:
            return
        case non_channels:
            raise TypeError(f"{len(non_channels)} non-channel value(s) for attribute {attr.name}")


def _channel_indices_are_legal(
    inst: "Channels", attr: attrs.Attribute, value: tuple[ChannelMeta, ...]
) -> None:
    """Check that the channel count is as expected and indices are [0, ..., N - 1], throwing a ValueError otherwise."""
    n_ch: int = len(value)
    if n_ch != inst.count:
        raise ValueError(
            f"Got {n_ch} channel(s) (from attribute {attr.name}) for a {type(inst).__name__} instance claiming to have {inst.count} channels"
        )
    match [
        (obs, exp)
        for exp, obs in zip(range(n_ch), sorted(map(lambda c: c.index, value)), strict=True)
        if obs != exp
    ]:
        case []:
            return
        case mismatched:
            raise ValueError(
                f"For attribute {attr.name}, indices don't match expectation in {len(mismatched)} case(s): {mismatched}"
            )


def _channel_names_are_unique(
    _: "Channels", attr: attrs.Attribute, value: tuple[ChannelMeta, ...]
) -> None:
    """Check that the channels declared together are unique based on name."""
    match list(
        (name, count) for name, count in Counter(c.name for c in value).items() if count != 1
    ):
        case []:
            return
        case repeated:
            raise ValueError(
                f"For attribute {attr.name}, there are repeated channel name(s): {repeated}"
            )


@attrs.define(kw_only=True, frozen=True, eq=True)
class Channels:
    """A representation of the collection of channels for a particular image."""

    count = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)])  # type: int
    values = attrs.field(
        validator=[
            attrs.validators.instance_of(tuple),
            _all_are_channels,
            _channel_indices_are_legal,
            _channel_names_are_unique,
        ]
    )  # type: tuple[ChannelMeta, ...]

    @classmethod
    def from_count_and_metas(
        cls, count: int, channels: Iterable[ChannelMeta]
    ) -> Result["Channels", str]:
        try:
            inst = cls(count=count, values=tuple(channels))
        except Exception as e:
            return Result.Error(
                f"Cannot build {cls.__name__} instance; error ({type(e).__name__}): {e}"
            )
        return Result.Ok(inst)


def _parse_channel_key(key: str) -> Option[int]:
    return Option.of_optional(re.search(r"^channel_(\d+)$", key)).map(
        lambda match: int(match.group(1))
    )


def _build_channel_meta(index: int, data: dict[str, object]) -> Result[ChannelMeta, str]:
    kwargs = copy.copy(data)
    kwargs["index"] = index
    try:
        ch_md: ChannelMeta = ChannelMeta(**kwargs)
    except Exception as e:
        return Result.Error(f"Error ({type(e).__name__})")
    return Result.Ok(ch_md)


def _parse_channel_value(*, index: int, value: object) -> Result[ChannelMeta, str]:
    match value:
        case dict():
            return _build_channel_meta(index=index, data=value)
        case _:
            return Result.Error(
                f"Wrong type ({type(value).__name__}, not dict) for channel data object"
            )


def parse_channel_meta_maybe(key: str, value: object) -> Option[Result[ChannelMeta, list[str]]]:
    return _parse_channel_key(key).map(lambda index: _parse_channel_value(index=index, value=value))


def parse_channels_from_flattened_mapping_with_count(
    zarr_root_attrs: Mapping[str, object],
) -> Result[Channels, list[str]]:
    return (
        Option.of_optional(zarr_root_attrs.get("metadata"))
        .to_result(
            [
                f"Missing metadata key in given mapping; {len(zarr_root_attrs.keys())} key(s): {', '.join(zarr_root_attrs.keys())}"
            ]
        )
        .bind(
            lambda metadata: Option.of_optional(metadata.get(CHANNEL_COUNT_KEY))
            .to_result([f"Missing the key ({CHANNEL_COUNT_KEY}) for number of channels"])
            .map(lambda n_ch: (n_ch, metadata))
        )
        .bind(
            lambda n_ch__metadata: sequence_accumulate_errors(
                TypedArray.of_seq(snd(n_ch__metadata).items()).choose(
                    lambda kv: parse_channel_meta_maybe(*kv)
                )
            ).bind(
                lambda metas: Channels.from_count_and_metas(fst(n_ch__metadata), metas).map_error(
                    lambda msg: [msg]
                )
            )
        )
    )


def parse_channels_from_mapping_with_channels_list(
    metadata: Mapping[str, object],
) -> Result[Channels, list[str]]:
    key: Literal["channels"] = "channels"

    def safe_read_one(maybe_ch_meta: object) -> Result[ChannelMeta, str]:
        match maybe_ch_meta:
            case data if isinstance(data, Mapping):
                try:
                    return Result.Ok(ChannelMeta(**data))
                except (TypeError, ValueError) as e:
                    return Result.Error(
                        f"Cannot parse as data as channel meta; error ({type(e).__name__}): {e}"
                    )
            case _:
                return Result.Error(
                    f"Data to parse as channel meta isn't mapping, but {type(maybe_ch_meta).__name__}"
                )

    def safe_read_multiple(maybe_channels_list: object) -> Result[list[ChannelMeta], list[str]]:
        match maybe_channels_list:
            case items if isinstance(items, Iterable):
                return traverse_accumulate_errors(safe_read_one)(items)
            case _:
                return Result.Error(
                    [
                        f"Potential list of channels isn't iterable, but {type(maybe_channels_list).__name__}"
                    ]
                )

    return (
        Option.of_optional(metadata.get(key))
        .to_result([f"Missing the key ({key}) for the list of channels"])
        .bind(safe_read_multiple)
        .map(lambda chs: Channels(count=len(chs), values=tuple(chs)))
    )


def parse_channels_from_zarr(
    loc: Path | ExtantFolder | ZarrLocation,
    *,
    parse_channels: Callable[[Mapping[str, object]], Result[Channels, list[str]]],
) -> Result[Channels, list[str]]:
    zarr_path: ZarrLocation
    match loc:
        case Path():
            zarr_path = ZarrLocation(loc)
        case ExtantFolder():
            zarr_path = ZarrLocation(loc.path)
        case ZarrLocation():
            zarr_path = loc
        case unknown:
            return Result.Error(
                [f"Cannot parse channels from value of type {type(unknown).__name__}"]
            )
    return parse_channels(zarr_path.root_attrs)


def create_zgroup_file(*, root: Path) -> Path:
    """Create the minimal content for the ZARR group file (format), in the correct place."""
    outpath = root / ".zgroup"
    with outpath.open(mode="x") as outfile:
        json.dump({"zarr_format": 2}, outfile, indent=4)
    return outpath


def extract_single_channel_single_z_data(
    *,
    ch: ChannelIndex,
    img: np.ndarray,
    dim: "CanonicalImageDimensions",
) -> Result[np.ndarray, list[str]]:
    z: int = floor(dim.z / 2)  # Target the central z-slice.
    return dim.get_axes_data(img, ((ZarrAxis.C, ch), (ZarrAxis.Z, z)))


class AxisMapping(Protocol):
    def get_length(self, axis: ZarrAxis) -> int:
        return getattr(self, axis.name)

    @property
    def rank(self) -> int:
        return len(_get_fields(self))

    def to_axis_map(self) -> OrderedDict[ZarrAxis, int]:
        return OrderedDict(
            (_ATTR_AXIS_PAIRS[attr], getattr(self, attr)) for attr in _iter_attr_names(self)
        )

    def get_axes_data(
        self, array: ArrayLike, axis_value_pairs: Iterable[tuple[ZarrAxis, int]]
    ) -> Result[np.ndarray, list[str]]:
        indexer: list[int | slice] = []
        requests: Mapping[str, int] = {ax.name: value for ax, value in axis_value_pairs}
        errors: list[str] = []
        for attr_name in _iter_attr_names(self):
            try:
                i = requests[attr_name]
            except KeyError:
                i = slice(0, None)
            else:
                upper_bound: int = getattr(self, attr_name)
                if not 0 <= i < upper_bound:
                    errors.append(
                        f"Value (for {attr_name}) out of bounds (not in [0, {upper_bound})): {i}"
                    )
            indexer.append(i)
        return Result.Error(errors) if errors else Result.Ok(array[tuple(indexer)])

    def get_axis_data(
        self, i: int, *, axis: ZarrAxis, array: ArrayLike
    ) -> Result[np.ndarray, str]:
        if array.ndim != self.rank:
            return Result.Error(f"Array is rank {array.ndim}, but dimensions are rank {self.rank}")
        ax_abbr: str = axis.name
        upper_bound: int = getattr(self, ax_abbr)
        if not 0 <= i < upper_bound:
            return Result.Error(
                f"Value (for {ax_abbr}) out of bounds (not in [0, {upper_bound})): {i}"
            )
        indexer: tuple[int | slice, ...] = tuple(
            i if attr_name == ax_abbr else slice(0, None) for attr_name in _iter_attr_names(self)
        )
        return Result.Ok(array[indexer])

    def get_channel_data(
        self, *, channel: int, array: ArrayLike
    ) -> Result[np.ndarray, str]:
        return self.get_axis_data(channel, axis=ZarrAxis.C, array=array)


@attrs.define(frozen=True, kw_only=True, eq=True)
class CanonicalImageDimensions(AxisMapping):
    t = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    c = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    z = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    y = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    x = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int

    def get_z_data(
        self, *, z_slice: int, array: ArrayLike
    ) -> Result[np.ndarray, str]:
        return self.get_axis_data(z_slice, axis=ZarrAxis.Z, array=array)


ArrayWithDimensions: TypeAlias = tuple[zarr.Array, CanonicalImageDimensions]


def get_single_array_from_zarr_group(group: zarr.Group) -> Result[zarr.Array, str]:
    match list(group.array_keys()):
        case [arr_name]:
            return Result.Ok(group.get(arr_name))
        case names:
            return Result.Error(f"Not a single array, but {len(names)} present in ZARR group")


def parse_single_array_and_dimensions_from_zarr_group(
    group: zarr.Group,
) -> Result[ArrayWithDimensions, str]:
    try:
        attrs: zarr.attrs.Attributes = group.attrs
    except AttributeError:
        return Result.Error("Alleged zarr.Group instance lacks .attrs member")
    try:
        axes: list[Mapping[str, str]] = attrs["multiscales"][0]["axes"]
    except (IndexError, KeyError, TypeError) as e:
        return Result.Error(f"Cannot access axes from attrs; error ({type(e).__name__}): {e}")
    try:
        axis_names: list[str] = [ax["name"] for ax in axes]
    except KeyError as e:
        return Result.Error(f"Cannot access axis name for each element of axes; error: {e}")
    match get_single_array_from_zarr_group(group):
        case result.Result(tag="ok", ok=arr):
            dims: list[int] = list(arr.shape)
            if len(dims) != len(axis_names):
                return Result.Error(
                    f"{len(axis_names)} axis name(s) but {len(dims)} dimensions; these must match"
                )
            try:
                return Result.Ok(
                    (arr, CanonicalImageDimensions(**dict(zip(axis_names, dims, strict=True))))
                )
            except TypeError as e:
                return Result.Error(f"Could not build image dimensions; error ({type(e).__name__}): {e}")
        case res:
            return res


def _get_fields(axis_map: CanonicalImageDimensions) -> Iterable[attrs.Attribute]:
    return attrs.fields(axis_map if isinstance(axis_map, type) else axis_map.__class__)


def _iter_attr_names(cls: type) -> Iterable[str]:
    # NB: would be nice to refine the argument type here, upper bounding as an attrs class.
    for attr in _get_fields(cls):
        yield attr.name


def compute_corrected_channels(
    *,
    image: zarr.Array,
    image_dimensions: CanonicalImageDimensions,
    image_channels: Channels,
    weights: zarr.Array,
    weights_dimensions: CanonicalImageDimensions,
    weights_channels: Channels,
) -> Result[list[np.ndarray], list[str]]:
    errors: list[str] = []
    if image.ndim != image_dimensions.rank:
        errors.append(
            f"Image is of rank {image.ndim}, but dimensions are of rank {image_dimensions.rank}"
        )
    if weights.ndim != weights_dimensions.rank:
        errors.append(
            f"Weights are of rank {weights.ndim}, but dimensions are of rank {weights_dimensions.rank}"
        )
    if image_channels.count != image_dimensions.c:
        errors.append(f"Image channels count is {image_channels.count} but dimensions allege {image_dimensions.c}")
    if weights_channels.count != weights_dimensions.c:
        errors.append(f"Weights channels count is {weights_channels.count} but dimensions allege {weights_dimensions.c}")
    if errors:
        return Result.Error(errors)

    channel_indices_in_weights: dict[tuple[ChannelName, WaveLenOpt, WaveLenOpt], int] = {
        ch.get_lookup_key(): i 
        for i, ch in enumerate(weights_channels.values)
    }
    def index_channel_in_weights(ch: ChannelMeta) -> Result[int, str]:
        return Option.of_optional(channel_indices_in_weights.get(ch.get_lookup_key())).to_result(f"Channel not defined in weights: {ch}")

    match traverse_accumulate_errors(lambda t: index_channel_in_weights(snd(t)).map(lambda i: (fst(t), i)))(enumerate(image_channels.values)):
        case result.Result(tag="ok", ok=index_pairs):
            errors: list[str] = []
            by_ch: list[Result[np.ndarray, list[str]]] = []
            for ch_img_index, ch_wts_index in index_pairs:
                match sequence_accumulate_errors(
                    (
                        image_dimensions.get_channel_data(channel=ch_img_index, array=image),
                        weights_dimensions.get_channel_data(channel=ch_wts_index, array=weights),
                    )
                ):
                    case result.Result(tag="ok", ok=(img, wts)):
                        by_ch.append(
                            np.vectorize(round)(
                                np.clip(img * wts, a_min=0, a_max=np.iinfo(img.dtype).max)
                            )
                        )
                    case result.Result(tag="error", error=messages):
                        errors.append(
                            f"For image channel {ch_img_index} and weights channel {ch_wts_index}, {len(errors)} error(s) extracting data: {', '.join(messages)}"
                        )
            return Result.Error(errors) if errors else Result.Ok(by_ch)
        case result.Result(tag="error", error=messages):
            return Result.Error(messages)
        case unknown:
            raise TypeError(f"Expected an expression.Result-wrapped value but got a {type(unknown).__name__}")
