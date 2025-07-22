"""Helpers for working with data stored in ZARR"""

import copy
import re
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Protocol, TypeAlias, TypeVar

import attrs
import numpy as np
import zarr
import zarr.attrs
from expression import Option, Result, result
from expression.collections import TypedArray

from illumifix.expression_utilities import sequence_accumulate_errors


class ZarrAxis(Enum):
    T = "time"
    C = "channel"
    Z = "space"
    Y = "space"
    X = "space"

    @property
    def name(self) -> str:
        return super().name.lower()

    @property
    def typename(self) -> str:
        return self.value


_CHECK_POSITIVE_INT = [attrs.validators.instance_of(int), attrs.validators.gt(0)]

_ATTR_AXIS_PAIRS: OrderedDict[str, ZarrAxis] = OrderedDict((ax.name, ax) for ax in ZarrAxis)

Channel: TypeAlias = int


def _check_null_or_positive_float(_, attr: attrs.Attribute, value: object) -> None:
    match value:
        case None:
            return
        case float():
            if value <= 0:
                raise ValueError(f"{attr.name} must be strictly positive; got {value}")
        case _:
            raise TypeError(f"{attr.name} is neither null nor float, but {type(value).__name__}")


@attrs.define(kw_only=True, frozen=True, eq=True)
class ChannelMeta:
    name = attrs.field(validator=attrs.validators.instance_of(str))  # type: str
    index = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)])  # type: int
    emissionLambdaNm = attrs.field(validator=_check_null_or_positive_float)  # type: Optional[float]
    excitationLambdaNm = attrs.field(validator=_check_null_or_positive_float)  # type: Optional[float]


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
    values: tuple[ChannelMeta, ...] = attrs.field(
        validator=[
            attrs.validators.instance_of(tuple),
            _all_are_channels,
            _channel_indices_are_legal,
            _channel_names_are_unique,
        ]
    )

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
    return _parse_channel_key(key).map(lambda index: _parse_channel_value(index=index, data=value))


def parse_channels(metadata: Mapping[str, object]) -> Result[Channels, list[str]]:
    n_ch_key: str = "channelCount"

    return (
        Option.of_optional(metadata.get(n_ch_key))
        .to_result(f"Missing the key ({n_ch_key}) for number of channels")
        .bind(
            lambda n_ch: sequence_accumulate_errors(
                TypedArray.of_seq(metadata.items()).choose(lambda kv: parse_channel_meta_maybe(*kv))
            ).bind(lambda metas: Channels.from_count_and_metas(n_ch, metas).map(lambda msg: [msg]))
        )
    )


class AxisMapping(Protocol):
    @property
    def rank(self) -> int:
        return len(_get_fields(self))

    def to_axis_map(self) -> OrderedDict[ZarrAxis, int]:
        return OrderedDict(
            (_ATTR_AXIS_PAIRS[attr], getattr(self, attr)) for attr in _iter_attr_names(self)
        )

    @classmethod
    def from_simple_mapping(
        cls, m: Mapping[str, int]
    ) -> "DimensionsForIlluminationCorrectionScaling":
        kwargs = copy.copy(m)
        for k in set(m.keys()) - set(_iter_attr_names(cls)):
            try:
                v = kwargs.pop(k)
            except KeyError:
                pass
            if v != 1:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_axis_map(cls, axis_map: Mapping[ZarrAxis, int]):
        """NB: This should throw an exception if and only if one of the size values is illegal; issues with keys should be safe."""
        kwargs: Mapping[str, int] = {}
        errors: list[str] = []
        keys: set[ZarrAxis] = set(axis_map.keys())
        for attr in _iter_attr_names(cls):
            axis = _ATTR_AXIS_PAIRS[attr]  # In this case, an implementation error was made, fatal.
            try:
                kwargs[attr] = axis_map[axis]
            except KeyError:
                errors.append(f"Missing axis: {axis}")
            else:
                keys.remove(axis)
        errors.extend([f"Extra axis: {ax}" for ax in keys])
        return Result.Error(errors) if errors else Result.Ok(cls(**kwargs))

    def get_axes_data(
        self, array: zarr.Array, axis_value_pairs: Iterable[tuple[ZarrAxis, int]]
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
        self, i: int, *, axis: ZarrAxis, array: zarr.Array
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

    def get_channel_data(self, *, channel: Channel, array: zarr.Array) -> Result[np.ndarray, str]:
        return self.get_axis_data(channel, axis=ZarrAxis.C, array=array)


@attrs.define(frozen=True, kw_only=True, eq=True)
class DimensionsForIlluminationCorrectionScaling(AxisMapping):
    c = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    y = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    x = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int


@attrs.define(frozen=True, kw_only=True, eq=True)
class CanonicalImageDimensions(AxisMapping):
    t = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    c = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    z = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    y = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int
    x = attrs.field(validator=_CHECK_POSITIVE_INT)  # type: int

    def get_z_data(
        self, *, z_slice: int, array: zarr.Array | np.ndarray
    ) -> Result[np.ndarray, str]:
        return self.get_axis_data(z_slice, axis=ZarrAxis.ZPLANE, array=array)


_AM = TypeVar("_AM", CanonicalImageDimensions, DimensionsForIlluminationCorrectionScaling)
ArrayWithDimensions: TypeAlias = tuple[zarr.Array, _AM]


def parse_single_array_and_dimensions_from_zarr_group(
    *, group: zarr.Group, target_type: type[_AM]
) -> Result[ArrayWithDimensions, str]:
    valid_types = (CanonicalImageDimensions, DimensionsForIlluminationCorrectionScaling)
    if not issubclass(target_type, valid_types):
        return Result.Error(
            f"{target_type.__name__} is not a valid target type for parsing ZARR group dimensions"
        )
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
    match list(group.array_keys()):
        case [array_name]:
            arr: zarr.Array = group.get(array_name)
            dims: list[int] = list(arr.shape)
            if len(dims) != len(axis_names):
                return Result.Error(
                    f"{len(axis_names)} axis name(s) but {len(dims)} dimensions; these must match"
                )
            try:
                return Result.Ok(
                    (arr, target_type.from_simple_mapping(dict(zip(axis_names, dims, strict=True))))
                )
            except TypeError as e:
                return Result.Error(f"Could not build target type ({target_type}): {e}")
        case array_names:
            return Result.Error(f"Not a single array, but {len(array_names)} present in ZARR group")


def _get_fields(axis_map: _AM | type[_AM]) -> Iterable[attrs.Attribute]:
    return attrs.fields(axis_map if isinstance(axis_map, type) else axis_map.__class__)


def _iter_attr_names(cls: type) -> Iterable[str]:
    # TODO: refine the argument type here, upper bounding as an attrs class.
    for attr in _get_fields(cls):
        yield attr.name


def compute_corrected_channels(
    *,
    image: zarr.Array,
    image_dimensions: CanonicalImageDimensions,
    weights: zarr.Array,
    weight_dimensions: DimensionsForIlluminationCorrectionScaling,
) -> Result[list[np.ndarray], list[str]]:
    errors: list[str] = []
    by_ch: list[Result[np.ndarray, list[str]]] = []
    if image.ndim != image_dimensions.rank:
        errors.append(
            f"Image is of rank {image.ndim}, but dimensions are of rank {image_dimensions.rank}"
        )
    if weights.ndim != weight_dimensions.rank:
        errors.append(
            f"Weights are of rank {weights.ndim}, but dimensions are of rank {weight_dimensions.rank}"
        )
    if errors:
        return Result.Error(errors)
    match _iterate_channels(dim_img=image_dimensions, dim_wts=weight_dimensions):
        case result.Result(tag="ok", ok=channels):
            for ch_img, ch_wts in channels:
                match sequence_accumulate_errors(
                    (
                        image_dimensions.get_channel_data(channel=ch_img, array=image),
                        weight_dimensions.get_channel_data(channel=ch_wts, array=weights),
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
                            f"For image channel {ch_img} and weights channel {ch_wts}, {len(errors)} error(s) extracting data: {', '.join(messages)}"
                        )
        case result.Result(tag="error", error=err):
            return Result.Error([err])
    return Result.Error(errors) if errors else Result.Ok(by_ch)


def _iterate_channels(
    *,
    dim_img: CanonicalImageDimensions,
    dim_wts: DimensionsForIlluminationCorrectionScaling,
) -> Result[Iterable[Channel], str]:
    if dim_img.c == dim_wts.c:
        return Result.Ok(zip(range(dim_img.c), range(dim_wts.c), strict=False))
    return Result.Error(
        f"Image's dimensions indicate {dim_img.c} channel(s) while weights' dimensions indicate {dim_wts.c} channel(s)"
    )
