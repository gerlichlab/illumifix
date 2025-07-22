"""Computation of the weights/scalings for pixel values, based on nonuniform illumination across a field of view"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from math import floor
from pathlib import Path
from typing import TypeAlias

import numpy as np
import zarr
from expression import Result, fst, result, snd
from expression.collections import TypedArray
from gertils.pathtools import ExtantFile, ExtantFolder, NonExtantPath, PathWrapperException

from illumifix.expression_utilities import sequence_accumulate_errors, traverse_accumulate_errors
from illumifix.zarr_tools import (
    CanonicalImageDimensions,
    DimensionsForIlluminationCorrectionScaling,
    ZarrAxis,
    parse_single_array_and_dimensions_from_zarr_group,
)

PathErrPair: TypeAlias = tuple[Path, str]


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute scalings for pixel values to correct for uneven illumination across a field of view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-list-file",
        type=ExtantFile.from_string,
        required=True,
        help="Path to file listing the paths in which to find images",
    )
    parser.add_argument(
        "-O", "--output-path", type=NonExtantPath.from_string, help="Path to which to write output"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting of existing weights/scalings"
    )
    return parser.parse_args(cmdl)


def parse_target_path(
    img_path: ExtantFolder,
) -> Result[tuple[zarr.Array, CanonicalImageDimensions], PathErrPair]:
    p: Path = img_path.path
    print("Parsing path: ", p)
    group: zarr.Group = zarr.open_group(p, mode="r")
    return parse_single_array_and_dimensions_from_zarr_group(
        group=group, target_type=CanonicalImageDimensions
    ).map_error(lambda e: (p, e))


def extract_single_channel_single_plane_data(
    *, ch: int, img: np.ndarray, dim: CanonicalImageDimensions
) -> Result[np.ndarray, list[str]]:
    z: int = floor(dim.z / 2)
    return dim.get_axes_data(img, ((ZarrAxis.C, ch), (ZarrAxis.Z, z)))


def unsafe_extract_single_channel_single_plane_data(
    *, ch: int, img: np.ndarray, dim: CanonicalImageDimensions
) -> np.ndarray:
    print(f"Extracting data from channel {ch}")
    match extract_single_channel_single_plane_data(ch=ch, img=img, dim=dim):
        case result.Result(tag="ok", ok=data):
            return data
        case result.Result(tag="error", error=messages):
            raise Exception(
                f"Cannot extract data for channel {ch}; {len(messages)} problem(s): {', '.join(messages)}"
            )


def validate_dimensions(
    path_arrdim_pairs: list[tuple[ExtantFolder, tuple[zarr.Array, CanonicalImageDimensions]]],
) -> Result[
    tuple[
        CanonicalImageDimensions,
        list[tuple[ExtantFolder, tuple[zarr.Array, CanonicalImageDimensions]]],
    ],
    list[PathErrPair],
]:
    try:
        _, (_, canonical) = next(iter(path_arrdim_pairs))
    except StopIteration:
        return Result.Ok([])

    def _check_one(
        pair: tuple[ExtantFolder, tuple[zarr.Array, CanonicalImageDimensions]],
    ) -> Result[tuple[ExtantFolder, tuple[zarr.Array, CanonicalImageDimensions]], PathErrPair]:
        pair: CanonicalImageDimensions
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
    *, obs_dim: CanonicalImageDimensions, exp_dim: DimensionsForIlluminationCorrectionScaling
) -> Result[CanonicalImageDimensions, list[str]]:
    problems: list[str] = [
        f"Expected {num_exp} {dim_name}(s) but got {num_obs}"
        for num_obs, num_exp, dim_name in [
            (obs_dim.t, 1, "timepoint"),
            (obs_dim.c, exp_dim.c, "channel"),
            (obs_dim.y, exp_dim.y, "y"),
            (obs_dim.x, exp_dim.x, "x"),
        ]
        if num_obs != num_exp
    ]
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
        return Result.Error((folder, f"{len(extras)} extra element(s): {', '.join(extras)}"))
    if not goods:
        return Result.Error((folder, "No targets found"))
    return Result.Ok(goods)


def average_all_data_per_channel(
    canonical_dimensions: CanonicalImageDimensions,
    data_pairs: list[tuple[ExtantFolder, tuple[zarr.Array, CanonicalImageDimensions]]],
) -> Result[tuple[list[np.ndarray], list[Path]], list[PathErrPair]]:
    print("Averaging data per image channel")

    def proc_one(
        *,
        ch: int,
        dim: CanonicalImageDimensions,
        img: zarr.Array,
        state: Result[tuple[np.ndarray, int], list[str]],
    ) -> Result[tuple[np.ndarray, int], list[str]]:
        match (extract_single_channel_single_plane_data(ch=ch, img=img, dim=dim), state):
            case (result.Result(tag="ok", ok=arr), result.Result(tag="ok", ok=(acc, n))):
                return Result.Ok((arr + acc, n + 1))
            case (result.Result(tag="ok", ok=_), result.Result(tag="error", error=es)):
                return Result.Error(es)
            case (result.Result(tag="error", error=messages), result.Result(tag="ok", ok=_)):
                return Result.Error([f"{len(messages)} problem(s): {', '.join(messages)}"])
            case (result.Result(tag="error", error=e), result.Result(tag="error", error=es)):
                return Result.Error(es + [e])
            case unknown:
                raise Exception(f"MatchError (type {type(unknown).__name__}): {unknown}")

    # Initialize the data accumulator.
    datastream = iter(data_pairs)
    init_data: zarr.Array
    try:
        _, (init_data, _) = next(datastream)
    except StopIteration as e:
        raise Exception("No data over which to average!") from e

    per_channel_result: TypedArray = TypedArray.of_seq(range(canonical_dimensions.c)).map(
        lambda ch: TypedArray.of_seq(datastream)
        .fold(
            lambda state, path_arrdim_pair: proc_one(
                ch=ch, dim=snd(snd(path_arrdim_pair)), img=fst(snd(path_arrdim_pair)), state=state
            ).map_error(
                lambda messages: (
                    fst(path_arrdim_pair),
                    f"{len(messages)} problem(s) processing data for channel {ch}: {', '.join(messages)}",
                )
            ),
            Result.Ok(
                (
                    unsafe_extract_single_channel_single_plane_data(
                        ch=ch, img=init_data, dim=canonical_dimensions
                    ).astype(np.uint32),
                    1,
                )
            ),  # Expand the data type so that we can accumulate many values.
        )
        .map(lambda acc_n_pair: fst(acc_n_pair) / float(snd(acc_n_pair)))
    )
    return sequence_accumulate_errors(per_channel_result).map(
        lambda arrays: (arrays, [folder.path for folder, _ in data_pairs])
    )


def workflow(
    *, base_paths: Iterable[Path], output_path: NonExtantPath, overwrite: bool = False
) -> None:
    # Extend the declared target to the base path of the images.
    match (
        traverse_accumulate_errors(build_target_path)(base_paths)
        .bind(traverse_accumulate_errors(expand_target_folder))
        .map(lambda targets: TypedArray.of_seq(targets).fold(lambda acc, sub: acc + sub, []))
        .bind(
            traverse_accumulate_errors(
                lambda p: parse_target_path(p).map(lambda arr_dim: (p, arr_dim))
            )
        )
        .bind(validate_dimensions)
        .bind(lambda canonical_and_pairs: average_all_data_per_channel(*canonical_and_pairs))
        .map(
            lambda arrays_and_paths: (
                np.stack(map(lambda arr: arr / arr.mean(), fst(arrays_and_paths))).swapaxes(0, 1),
                snd(arrays_and_paths),
            )
        )
    ):
        case result.Result(tag="ok", ok=(weights, paths)):
            assert weights.ndim == 4, f"Weights should be 4D, not {weights.ndim}D"
            assert (
                weights.shape[0] == 1
            ), f"Weights should have single timepoint, not {weights.shape[0]}"
            outroot: Path = output_path.path / "weights.zarr"
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
            # TODO: add the versioning metadata.
            with zarr_metadata_file.open(mode="w") as meta_file:
                json.dump(
                    {
                        "data_paths": list(map(str, paths)),
                        "multiscales": [
                            {
                                "axes": [
                                    {"name": "t", "type": "time"},
                                    {"name": "c", "type": "channel"},
                                    {"name": "y", "type": "space"},
                                    {"name": "x", "type": "space"},
                                ]
                            }
                        ],
                    },
                    meta_file,
                    indent=2,
                )
            # TODO: create the heatmap of the weights and write to disk
            logging.info("Generating visualization of the weights/scalings")
            visualization_output_path: Path = output_path.path / "visualization"
            logging.info(
                "Writing visualization of the weights/scalings to disk: %s",
                visualization_output_path,
            )
            raise NotImplementedError(
                "Visualization of the weights/scalings has not yet been implemented."
            )
        case result.Result(tag="error", error=path_message_pairs):
            raise Exception(f"{len(path_message_pairs)} problem(s): {path_message_pairs}")


def main(cmdl: list[str]) -> None:
    opts = _parse_cmdl(cmdl)
    pathlist: Path = opts.path_list_file.path
    logging.info("Reading paths list: %s", pathlist)
    with open(pathlist) as fh:
        raw_base_paths: list[Path] = [Path(line.strip()) for line in fh.readlines() if line.strip()]
    logging.info("Running workflow with %d base paths", len(raw_base_paths))
    workflow(base_paths=raw_base_paths, output_path=opts.output_path, overwrite=opts.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
