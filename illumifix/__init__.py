"""Project root"""

from pathlib import Path
from ome_zarr.format import FormatV02

# Make the __version__ attribute of the project available at the top level.
from ._version import __version__


PROJECT_NAME = Path(__file__).resolve().parent.name

ZARR_FORMAT = FormatV02()
