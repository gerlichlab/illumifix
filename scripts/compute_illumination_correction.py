"""Computation of the weights/scalings for pixel values, based on nonuniform illumination across a field of view"""

import argparse
from functools import partial
import itertools
import json
import logging
import re
import sys
from pathlib import Path 
from typing import Callable, Iterable, Mapping, TypeAlias

import attrs
from expression import Option, Result, fst, result, snd
from expression.collections import TypedArray
from gertils.pathtools import ExtantFile, ExtantFolder, NonExtantPath, PathWrapperException
import numpy as np
from ome_zarr.axes import Axes  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]

from illumifix import PROJECT_NAME, ZARR_FORMAT
from illumifix.expression_utilities import sequence_accumulate_errors, traverse_accumulate_errors
from illumifix.zarr_tools import (
    CanonicalImageDimensions,
    ChannelIndex,
    ChannelKey,
    ChannelMeta,
    Channels,
    JsonEncoderForChannelMeta,
    ZarrAxis,
    extract_single_channel_single_z_data,
    parse_channels_from_flattened_mapping_with_count,
    parse_channels_from_zarr,
    parse_single_array_and_dimensions_from_zarr_group,
)

ArrayWithDims: TypeAlias = tuple[zarr.Array, CanonicalImageDimensions]
PathErrPair: TypeAlias = tuple[Path, str]


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute scalings for pixel values to correct for uneven illumination across a field of view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-list-file",
        required=True,
        type=ExtantFile.from_string,
        help="Path to file listing the paths in which to find images",
    )
    parser.add_argument(
        "-O",
        "--output-path",
        required=True,
        type=NonExtantPath.from_string,
        help="Path to which to write output",
    )
    parser.add_argument(
        "--version-name",
        required=True,
        help="Name for the version of the weights/scalings to be produced by this run of the program",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting of existing weights/scalings"
    )
    return parser.parse_args(cmdl)


def parse_target_path(
    img_path: ExtantFolder,
) -> Result[ArrayWithDims, PathErrPair]:
    p: Path = img_path.path
    print("Parsing path: ", p)
    group: zarr.Group = zarr.open_group(p, mode="r")
    return parse_single_array_and_dimensions_from_zarr_group(group=group).map_error(
        lambda e: (p, e)
    )


def unsafe_extract_single_channel_single_z_data(
    *, ch: ChannelIndex, img: np.ndarray, dim: CanonicalImageDimensions
) -> np.ndarray:
    print(f"Extracting data from channel {ch}")
    match extract_single_channel_single_z_data(ch=ch, img=img, dim=dim):
        case result.Result(tag="ok", ok=data):
            return data
        case result.Result(tag="error", error=messages):
            raise Exception(
                f"Cannot extract data for channel {ch}; {len(messages)} problem(s): {', '.join(messages)}"
            )
        case unknown:
            raise TypeError(
                f"Expected an expression.Result-wrapped value but got a {type(unknown).__name__}"
            )


def validate_dimensions(
    path_arrdim_pairs: list[tuple[ExtantFolder, ArrayWithDims]],
) -> Result[
    tuple[
        CanonicalImageDimensions,
        list[tuple[ExtantFolder, ArrayWithDims]],
    ],
    list[PathErrPair],
]:
    try:
        first_folder, (_, canonical) = next(iter(path_arrdim_pairs))
    except StopIteration as e:
        raise ValueError("Cannot validate dimensions of no data") from e

    if canonical.t != 1:
        return Result.Error(
            [
                (
                    first_folder.path,
                    f"First path ({first_folder.path}) has time dimension of nontrivial length: {canonical.t}",
                )
            ]
        )

    def _check_one(
        pair: tuple[ExtantFolder, ArrayWithDims],
    ) -> Result[CanonicalImageDimensions, PathErrPair]:
        dim: CanonicalImageDimensions
        data_path: ExtantFolder
        data_path, (_, dim) = pair
        print("Validating dimensions: ", data_path.path)
        return _validate_dimensions_single(obs_dim=dim, exp_dim=canonical).map_error(
            lambda messages: (
                data_path.path,
                f"{len(messages)} dimension error(s): {', '.join(messages)}",
            )
        )

    return traverse_accumulate_errors(_check_one)(path_arrdim_pairs).map(
        lambda _: (canonical, path_arrdim_pairs)
    )


def _validate_dimensions_single(
    *,
    obs_dim: CanonicalImageDimensions,
    exp_dim: CanonicalImageDimensions,
) -> Result[CanonicalImageDimensions, list[str]]:
    problems: list[str] = []
    for ax in ZarrAxis:
        num_obs: int = obs_dim.get_length(ax)
        num_exp: int = exp_dim.get_length(ax)
        if num_obs != num_exp:
            problems.append(
                f"Expected dimension '{ax.name}' to have length {num_exp} but got {num_obs}"
            )
    return Result.Error(problems) if problems else Result.Ok(obs_dim)


def build_target_path(base_path: Path) -> Result[ExtantFolder, PathErrPair]:
    img_path: Path = base_path / "nuc_images_zarr"
    print("Building target path: ", img_path)
    try:
        return Result.Ok(ExtantFolder(img_path))
    except PathWrapperException:
        return result.Error((base_path, f"Not an extant folder: {img_path}"))


def expand_target_folder(folder: ExtantFolder) -> Result[list[ExtantFolder], PathErrPair]:
    print("Finding data paths: ", folder.path)

    extras: list[Path] = []
    goods: list[ExtantFolder] = []

    for p in folder.path.iterdir():
        if re.match(re.compile("P[0-9]{4}.zarr"), p.name):
            goods.append(ExtantFolder(p))
        else:
            extras.append(p)

    if extras:
        return Result.Error((folder.path, f"{len(extras)} extra element(s): {extras}"))
    if not goods:
        return Result.Error((folder.path, "No targets found"))
    return Result.Ok(goods)


def get_sorted_channels_representation(channels: Channels) -> Channels:
    """Order the metadata instances within the wrapper class."""
    ordered: list[ChannelMeta] = sorted(
        channels.values,
        key=lambda ch: (
            None if ch.emissionLambdaNm is None else -ch.emissionLambdaNm,
            ch.name,
            None if ch.excitationLambdaNm is None else -ch.excitationLambdaNm,
        ),
    )
    return Channels(
        count=channels.count,
        values=tuple(attrs.evolve(ch, index=i) for i, ch in enumerate(ordered)),
    )


def determine_channel_order(
    *, new_channels: Channels, ref_channels: Channels
) -> Result[list[int], str]:
    if new_channels == ref_channels:
        # Ideal (frequent) case --> short-circuit
        return Result.Ok(list(range(new_channels.count)))
    if new_channels.count != ref_channels.count:
        return Result.Error(
            f"New channels' count is {new_channels.count}, but reference count is {ref_channels.count}"
        )
    
    lookup: Mapping[ChannelKey, int] = {ch.get_lookup_key(): i for i, ch in enumerate(ref_channels.values)}
    get_ch_index: Callable[[ChannelMeta], Result[ChannelIndex, ChannelMeta]] = (
        lambda ch: Option.of_optional(lookup.get(ch.get_lookup_key())).to_result(ch)
    )
    return traverse_accumulate_errors(get_ch_index)(new_channels.values).map_error(
        lambda bad_chs: f"{len(bad_chs)} channel(s) cannot be resolved: {bad_chs}"
    )


def find_channel_rotations(
    path_arrdim_pairs: list[tuple[ExtantFolder, ArrayWithDims]],
) -> Result[
    list[tuple[ExtantFolder, list[int], zarr.Array, CanonicalImageDimensions]], list[PathErrPair]
]:
    def parse_channels(
        pair: tuple[ExtantFolder, ArrayWithDims],
    ) -> Result[tuple[ExtantFolder, Channels, zarr.Array, CanonicalImageDimensions], PathErrPair]:
        zarr_root: ExtantFolder
        arr_dims: ArrayWithDims
        zarr_root, arr_dims = pair
        return (
            parse_channels_from_zarr(
                zarr_root, parse_channels=parse_channels_from_flattened_mapping_with_count
            )
            .map(lambda chs: (zarr_root, chs, *arr_dims))
            .map_error(
                lambda messages: (
                    zarr_root.path,
                    f"{len(messages)} error(s) parsing channels from {fst(pair)}: {'; '.join(messages)}",
                )
            )
        )

    def det_ord(
        tup: tuple[ExtantFolder, Channels, zarr.Array, CanonicalImageDimensions],
        ref_channels: Channels,
    ) -> Result[tuple[ExtantFolder, list[int], zarr.Array, CanonicalImageDimensions], PathErrPair]:
        return (
            determine_channel_order(new_channels=tup[1], ref_channels=ref_channels)
            .map(lambda ord: (tup[0], ord, tup[2], tup[3]))
            .map_error(lambda msg: (tup[0].path, msg))
        )

    return traverse_accumulate_errors(parse_channels)(path_arrdim_pairs).bind(
        lambda tuples: Result.Ok([])
        if len(tuples) == 0
        else traverse_accumulate_errors(partial(det_ord, ref_channels=tuples[0][1]))(tuples)
    )


def average_all_data_per_channel(
    canonical_dimensions: CanonicalImageDimensions,
    data: list[tuple[ExtantFolder, list[int], zarr.Array, CanonicalImageDimensions]],
) -> Result[tuple[list[tuple[ChannelMeta, np.ndarray]], list[Path]], list[PathErrPair]]:
    print("Averaging data per image channel")

    def proc_one(
        *,
        ch: int,
        dim: CanonicalImageDimensions,
        img: zarr.Array,
        source_folder: ExtantFolder,
        state: Result[tuple[np.ndarray, int], list[PathErrPair]],
    ) -> Result[tuple[np.ndarray, int], list[PathErrPair]]:
        def combine_error_messages(messages: list[str]) -> str:
            return f"{len(messages)} problem(s): {', '.join(messages)}"

        match (extract_single_channel_single_z_data(ch=ch, img=img, dim=dim), state):
            case (result.Result(tag="ok", ok=arr), result.Result(tag="ok", ok=(acc, n))):
                return Result.Ok((arr + acc, n + 1))
            case (result.Result(tag="ok", ok=_), result.Result(tag="error", error=es)):
                return Result.Error(es)
            case (result.Result(tag="error", error=messages), result.Result(tag="ok", ok=_)):
                return Result.Error([(source_folder.path, combine_error_messages(messages))])
            case (result.Result(tag="error", error=messages), result.Result(tag="error", error=es)):
                return Result.Error([*es, (source_folder.path, combine_error_messages(messages))])
            case unknown:
                raise Exception(f"MatchError (type {type(unknown).__name__}): {unknown}")

    init_data: zarr.Array
    init_ch_ord: list[int]
    init_folder: ExtantFolder
    channels: Channels
    try:
        init_folder, init_ch_ord, init_data, _ = data[0]
    except IndexError as e:
        raise Exception("No data over which to average!") from e
    match parse_channels_from_zarr(
        init_folder, parse_channels=parse_channels_from_flattened_mapping_with_count
    ):
        case result.Result(tag="ok", ok=chs):
            channels = get_sorted_channels_representation(chs)
        case result.Result(tag="error", error=err):
            raise RuntimeError(
                f"When re-parsing channels from {init_folder.path}, got error(s): {err}"
            )
        case unknown:
            raise TypeError(
                f"Expected an expression.Result-wrapped value, but got a {type(unknown).__name__}"
            )

    def init_state(ch_idx: ChannelIndex) -> Result[tuple[np.ndarray, int], list[PathErrPair]]:
        return Result.Ok(
            (
                unsafe_extract_single_channel_single_z_data(
                    ch=init_ch_ord[ch_idx], img=init_data, dim=canonical_dimensions
                ).astype(
                    np.uint64
                ),  # Expand the data type so that we can accumulate many values --> ~35 Mb
                1,
            )
        )

    # Iterable[Result[tuple[ChannelMeta, np.ndarray], list[PathErrPair]]]
    per_channel_result: TypedArray = TypedArray.of_seq(enumerate(channels.values)).map(
        lambda i_ch_pair: TypedArray.of_seq(data[1:])
        .fold(
            lambda state, tup: proc_one(
                ch=tup[1][fst(i_ch_pair)], dim=tup[3], img=tup[2], source_folder=tup[0], state=state
            ),
            init_state(fst(i_ch_pair)),
        )
        .map(lambda acc_n_pair: (snd(i_ch_pair), fst(acc_n_pair) / float(snd(acc_n_pair))))
    )

    # Sequence the results, and then inject the paths in the success case.
    return sequence_accumulate_errors(per_channel_result).map(
        lambda ch_arr_pairs: (ch_arr_pairs, [folder.path for folder, _, _, _ in data])
    )


def workflow(
    *,
    base_paths: Iterable[Path],
    output_path: NonExtantPath,
    version_name: str,
    overwrite: bool = False,
) -> None:
    match (
        traverse_accumulate_errors(build_target_path)(base_paths)
        .bind(traverse_accumulate_errors(expand_target_folder))
        .map(lambda targets: list(itertools.chain(*targets)))
        .bind(
            traverse_accumulate_errors(
                lambda p: parse_target_path(p).map(lambda arr_dim: (p, arr_dim))
            )
        )
        .bind(validate_dimensions)
        .bind(
            lambda canonical_and_tuples: find_channel_rotations(snd(canonical_and_tuples)).map(
                lambda tuples: (fst(canonical_and_tuples), tuples)
            )
        )
        .bind(lambda canonical_and_tuples: average_all_data_per_channel(*canonical_and_tuples))
        .map(
            lambda ch_arr_pairs__and__paths: (
                [ch for ch, _ in fst(ch_arr_pairs__and__paths)],
                np.stack([arr.mean() / arr for _, arr in fst(ch_arr_pairs__and__paths)]).swapaxes(
                    0, 1
                ),
                snd(ch_arr_pairs__and__paths),
            )
        )
    ):
        case result.Result(tag="error", error=path_message_pairs):
            raise Exception(f"{len(path_message_pairs)} problem(s): {path_message_pairs}")
        case result.Result(tag="ok", ok=(channels, weights, paths)):
            weights: np.ndarray = reshape_as_necessary(weights)  # type: ignore[no-redef]
            outroot: Path = output_path.path
            logging.info("Creating ZARR root: %s", outroot)
            root = zarr.group(store=zarr.DirectoryStore(path=outroot), overwrite=overwrite)
            logging.info("Creating array")
            arr = root.create_dataset(
                name="0", shape=weights.shape, dtype=weights.dtype, data=weights
            )
            logging.info("Writing data")
            arr[:] = weights
            zarr_metadata_file: Path = outroot / ".zattrs"
            logging.info("Writing metadata: %s", zarr_metadata_file)
            with zarr_metadata_file.open(mode="w") as meta_file:
                json.dump(
                    {
                        PROJECT_NAME: {
                            "version": version_name,
                            "input_paths": list(map(str, paths)),
                        },
                        "channels": channels,
                        "multiscales": [
                            {
                                "axes": Axes(
                                    [ax.name for ax in ZarrAxis],
                                    fmt=ZARR_FORMAT,
                                ).to_list()
                            },
                        ],
                    },
                    meta_file,
                    indent=2,
                    cls=JsonEncoderForChannelMeta,
                )
            logging.info("Done!")
        case unknown:
            raise Exception(
                f"Expected an expression.Result-wrapped value but got a {type(unknown).__name__}"
            )


def reshape_as_necessary(weights: np.ndarray) -> np.ndarray:
    match weights.shape:
        case (1, c, y, x):
            return weights.reshape((1, c, 1, y, x))
        case (1, _, 1, _, _):
            return weights
        case dims:
            raise RuntimeError(f"Illegal weights array shape: {dims}")


def main(cmdl: list[str]) -> None:
    opts = _parse_cmdl(cmdl)
    pathlist: Path = opts.path_list_file.path
    logging.info("Reading paths list: %s", pathlist)
    with open(pathlist) as fh:
        raw_base_paths: list[Path] = [Path(line.strip()) for line in fh.readlines() if line.strip()]
    logging.info("Running workflow with %d base paths", len(raw_base_paths))
    workflow(
        base_paths=raw_base_paths,
        output_path=opts.output_path,
        version_name=opts.version_name,
        overwrite=opts.overwrite,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
