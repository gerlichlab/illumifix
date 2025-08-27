"""Visualization of the illumination correction scalings"""

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import zarr
from expression import result
from gertils import ExtantFolder

from illumifix.zarr_tools import (
    CanonicalImageDimensions,
    ChannelIndex,
    OmeZarrAxis,
    parse_channels_from_mapping_with_channels_list,
    parse_channels_from_zarr,
    parse_single_array_and_dimensions_from_zarr_group,
)


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the computed illumination correction scalings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-I",
        "--zarr-root",
        required=True,
        type=ExtantFolder.from_string,
        help="Path to ZARR root where the illumination correction weights/scalings are stored",
    )
    parser.add_argument(
        "-O",
        "--output-folder",
        required=True,
        type=Path,
        help="Path to the folder in which to place output files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output",
    )
    return parser.parse_args(cmdl)


@attrs.define(kw_only=True, frozen=True)
class HeatmapSize2D:
    x: int
    y: int

    def check_image(self, img: np.ndarray) -> None:
        match img.shape:
            case (x, y):
                for obs, attr in [(x, "x"), (y, "y")]:
                    exp: int = getattr(self, attr)
                    if obs < exp:
                        raise ValueError(
                            f"For {attr}, image has {obs} pixels; cannot downsize to {exp}"
                        )
                    if obs % exp != 0:
                        logging.warning(
                            f"For {attr}, image pixel count ({obs}) is not evenly divisible by {exp}"  # noqa: G004
                        )
            case shape:
                raise ValueError(
                    f"After selecting timepoint 0 and a specific channel, expected a 2D array, but got {len(shape)}D"
                )

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        self.check_image(img)  # executed for effect only
        return skimage.transform.resize(img, output_shape=(self.x, self.y))


HEATMAP_SIZE = HeatmapSize2D(x=512, y=511)


# TODO: ensure yx vs. xy ordering.
def unsafe_get_yx_data(
    *,
    channel_index: ChannelIndex,
    dimensions: CanonicalImageDimensions,
    weights: zarr.Array | np.ndarray,
) -> np.ndarray:
    match dimensions.get_axes_data(
        weights, ((OmeZarrAxis.T, 0), (OmeZarrAxis.C, channel_index), (OmeZarrAxis.Z, 0))
    ):
        case result.Result(tag="ok", ok=arr):
            return arr
        case result.Result(tag="error", error=messages):
            raise RuntimeError(
                f"{len(messages)} problem(s) getting xy-plane data: {'; '.join(messages)}"
            )
        case unknown:
            raise TypeError(
                f"Expected an expression.Result-wrapped value but got a {type(unknown).__name__}"
            )


def workflow(*, zarr_root: Path, output_folder: Path, overwrite: bool = False) -> None:
    logging.info("Generating visualization of the illumination correction weights/scalings")
    group: zarr.Group = zarr.open_group(zarr_root, mode="r")

    def combine_error_messages(messages: Iterable[str]) -> str:
        return f"{len(messages)} error(s) parsing channels from ZARR ({zarr_root}): {'; '.join(messages)}"

    match (
        parse_channels_from_zarr(
            zarr_root,
            parse_channels=parse_channels_from_mapping_with_channels_list,
        ),
        parse_single_array_and_dimensions_from_zarr_group(group),
    ):
        case (
            result.Result(tag="ok", ok=channels),
            result.Result(tag="ok", ok=(weights, dimensions)),
        ):
            logging.info(f"Weights shape: {weights.shape}")  # noqa: G004
            match dimensions:
                case CanonicalImageDimensions(t=1, c=_, z=1, y=_, x=_):
                    visualization_output_path: Path = output_folder
                    if not overwrite and visualization_output_path.exists():
                        raise FileExistsError(visualization_output_path)
                    visualization_output_path.mkdir(exist_ok=True)
                    for i, ch in enumerate(channels.values):
                        logging.debug(f"Processing {i}-st/-nd/-th channel: {ch.name}")  # noqa: G004
                        logging.debug("Computing averaged weights for heatmap")
                        # TODO: ensure yx vs. xy ordering.
                        wt_2d: np.ndarray = HEATMAP_SIZE.resize_image(
                            unsafe_get_yx_data(
                                channel_index=i, dimensions=dimensions, weights=weights
                            ).swapaxes(0, 1)
                        )
                        logging.debug("Creating heatmap")
                        plt.imshow(wt_2d)
                        plt.colorbar(label="scaling")
                        heatmap_file: Path = (
                            visualization_output_path
                            / f"heatmap_{i}__{ch.name.replace(' ', '_')}.png"
                        )
                        logging.info("Saving heatmap: %s", heatmap_file)
                        plt.savefig(heatmap_file)
                        plt.clf()  # Clear the figure.
                        # TODO: plot title, axes, with at least channel information and perhaps original image size.
                        # TODO: consider also storing and visualizing statistics about the weights/scalings.
                    logging.info("Done!")
                case _:
                    raise RuntimeError(
                        f"Illegal weights array dimensions parsed from ZARR root: {dimensions}"
                    )
        case (result.Result(tag="ok", ok=_), result.Result(tag="error", error=err_msg)):
            raise RuntimeError(
                f"Error parsing array and dimensions from ZARR ({zarr_root}): {err_msg}"
            )
        case (result.Result(tag="error", error=messages), result.Result(tag="ok", ok=_)):
            raise RuntimeError(combine_error_messages(messages))
        case (
            result.Result(tag="error", error=messages),
            result.Result(tag="error", error=err_msg),
        ):
            raise RuntimeError(
                f"{combine_error_messages(messages)} AND error parsing data and dimensions: {err_msg}"
            )
        case unknown:
            raise TypeError(
                f"Expected an expression.Result-wrapped value parsing channels from ZARR, but got a {type(unknown).__name__}"
            )


def main(cmdl: list[str]) -> None:
    opts = _parse_cmdl(cmdl)
    workflow(
        zarr_root=opts.zarr_root.path, output_folder=opts.output_folder, overwrite=opts.overwrite
    )


if __name__ == "__main__":
    main(sys.argv[1:])
