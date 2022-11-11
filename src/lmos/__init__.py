from . import _version
from .lmos import (median_of_squares, minimize_multiple_guesses,
                   numerically_optimize)

__version__ = _version.get_versions()["version"]
