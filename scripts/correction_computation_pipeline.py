"""Simple pipeline for computation and visualization of the illumination correction"""

import argparse
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import pypiper
from compute_illumination_correction import workflow as compute_correction
from gertils import ExtantFile, NonExtantPath
from visualize_scalings import workflow as visualize_correction

NO_TEE_LOGS_OPTNAME = "--do-not-tee-logs"
PIPE_NAME = "illumination_correction_computation"
ZARR_NAME = "illumination_correction_scalings.zarr"


class CorrectionComputationPipeline(pypiper.Pipeline):
    """Pipeline to compute and visualize illumination correction weights/scalings"""

    def __init__(
        self,
        *,
        path_list_file: ExtantFile,
        output_root: Path,
        version_name: str,
        **pl_mgr_kwargs,
    ) -> None:
        """Create a pipeline instance.

        Args:
            path_list_file (ExtantFile): path to the file listing experiments to include
            output_root (Path): path to folder in which to place outputs
            version_name (str): the name of the version of the weights/scalings to output
            pl_mgr_kwargs: keyword arguments for the pypiper.PipelineManager

        """
        self.version_name = version_name
        self.weights_root = NonExtantPath(output_root / ZARR_NAME)
        self.visualization_folder = output_root / "visualization"
        with path_list_file.path.open(mode="r") as pathlist:
            self.input_paths: list[Path] = [Path(line.strip()) for line in pathlist if line.strip()]
        super().__init__(name=PIPE_NAME, outfolder=str(output_root / "pypiper"), **pl_mgr_kwargs)

    def stages(self) -> list[pypiper.Stage]:
        return [
            pypiper.Stage(
                name="computation",
                func=compute_correction,
                f_kwargs={
                    "base_paths": self.input_paths,
                    "output_path": self.weights_root,
                    "version_name": self.version_name,
                    "overwrite": True,
                },
            ),
            pypiper.Stage(
                name="visualization",
                func=visualize_correction,
                f_kwargs={
                    "zarr_root": self.weights_root.path,
                    "output_folder": self.visualization_folder,
                    "overwrite": True,
                },
            ),
        ]


def parse_cli(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A pipeline to compute and visualize correction for uneven illumination across fields of view",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path-list-file",
        required=True,
        type=ExtantFile.from_string,
        help="Path to file listing the paths in which to find images",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Path to folder in which to place main outputs",
    )
    parser.add_argument(
        "--version-name",
        required=True,
        help="Name for the version of the weights/scalings to be produced by this run of the program",
    )
    parser.add_argument(
        NO_TEE_LOGS_OPTNAME,
        action="store_true",
        help="Do not tee logging output from pypiper manager",
    )
    parser = pypiper.add_pypiper_args(
        parser,
        groups=("pypiper", "checkpoint"),
        args=("start-point",),
    )
    return parser.parse_args(args)


def init(opts: argparse.Namespace) -> CorrectionComputationPipeline:
    kwargs = {
        "path_list_file": opts.path_list_file,
        "version_name": opts.version_name,
        "output_root": opts.output_root,
    }
    if opts.do_not_tee_logs:
        kwargs["multi"] = True
    logging.info(
        f"Building {PIPE_NAME} pipeline, using data listed in from {opts.path_list_file.path}"  # noqa: G004
    )
    return CorrectionComputationPipeline(**kwargs)


def main(cmdl: list[str]) -> None:
    opts = parse_cli(cmdl)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Building pipeline")
    pipeline = init(opts)
    logging.info("Running pipeline")
    pipeline.run(start_point=opts.start_point, stop_after=opts.stop_after)
    pipeline.wrapup()


if __name__ == "__main__":
    main(sys.argv[1:])
