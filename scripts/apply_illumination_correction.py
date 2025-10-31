"""Correct pixel values for systematic differences in illumination over a field of view."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr
from expression import Result, result

from illumifix.expression_utilities import sequence_accumulate_errors
from illumifix.zarr_tools import (
    CanonicalImageDimensions,
    compute_corrected_channels,
    parse_single_array_and_dimensions_from_zarr_group,
)


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct for systematic differences in illumination between pixels in a field of view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-I", "--image", type=Path, required=True, help="Path to the (ZARR) image data to correct"
    )
    parser.add_argument(
        "-W", "--weights", type=Path, required=True, help="Path to the (ZARR) weights/scalings"
    )
    parser.add_argument(
        "-O", "--output-folder", type=Path, required=True, help="Path to which to write the corrected data"
    )
    return parser.parse_args(cmdl)


def workflow(*, image_path: Path, weights_path: Path, output_folder: Path) -> None:
    image_group: zarr.Group = zarr.open_group(image_path, mode="r")
    weights_group: zarr.Group = zarr.open_group(weights_path, mode="r")
    logging.info("Parsing image and weights/scalings data")
    image_parse: Result[tuple[zarr.Array, CanonicalImageDimensions], str] = (
        parse_single_array_and_dimensions_from_zarr_group(
            group=image_group,
            target_type=CanonicalImageDimensions,
        ).map_error(lambda msg: f"{msg} (parsing image)")
    )
    weights_parse: Result[tuple[zarr.Array, CanonicalImageDimensions], str] = (
        parse_single_array_and_dimensions_from_zarr_group(
            group=weights_group,
            target_type=CanonicalImageDimensions,
        ).map_error(lambda msg: f"{msg} (parsing weights)")
    )
    match sequence_accumulate_errors((image_parse, weights_parse)):
        case result.Result(tag="ok", ok=[(img, img_dim), (wts, wts_dim)]):
            logging.info("Computing illumination correction")
            match compute_corrected_channels(
                image=img, image_dimensions=img_dim, weights=wts, weights_dimensions=wts_dim
            ):
                case result.Result(tag="ok", ok=img_by_channel):
                    logging.debug("Stacking channels and swapping axes...")
                    corrected: np.ndarray = np.stack(img_by_channel).swapaxes(0, 1)
                    logging.info("Writing output: %s", output_folder)
                    write_group(data=corrected, path=output_folder)
                    logging.debug("Copying ZARR metadata")
                    shutil.copy(image_path / ".zattrs", output_folder / ".zattrs")
                    shutil.copy(image_path / ".zgroup", output_folder / ".zgroup")
                case result.Result(tag="error", error=messages):
                    raise Exception(
                        f"{len(messages)} error(s) computing corrected image per channel: {', '.join(messages)}"
                    )
        case result.Result(tag="error", error=messages):
            raise Exception(
                f"{len(messages)} error(s) parsing image and weights data: {', '.join(messages)}"
            )
        case unknown:
            raise Exception(
                f"Unexpected result from sequencing the results of parsing the image and weights data: {unknown}"
            )


def write_group(*, data: np.ndarray, path: Path) -> zarr.Group:
    store: zarr.DirectoryStore = zarr.DirectoryStore(path)
    root: zarr.Group = zarr.group(store=store)
    root.create_dataset("0", data=data)
    return root


def main(args: list[str]) -> None:
    opts = _parse_cmdl(args)
    if opts.output_folder == opts.image:
        raise ValueError(f"The image path and output path are the same: {opts.image}")
    workflow(image_path=opts.image, weights_path=opts.weights, output_folder=opts.output_folder)


if __name__ == "__main__":
    main(sys.argv[1:])
