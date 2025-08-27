"""Project root"""

import json
from pathlib import Path

from ome_zarr.format import FormatV02  # type: ignore[import-untyped]

# Make the __version__ attribute of the project available at the top level.
from ._version import __version__  # noqa: F401

PROJECT_NAME = Path(__file__).resolve().parent.name


class JsonEncoderForZarrFormat(json.JSONEncoder):
    """Allow encoding of a zarr.Format value as JSON."""

    def default(self, o) -> object:  # type: ignore[no-untyped-def] # noqa: ANN001
        """Attempt to encode the given object as JSON."""
        match o:
            case FormatV02():
                return {"zarr_format": 2}
            case _:
                return super().default(o)


ZARR_FORMAT = FormatV02()
