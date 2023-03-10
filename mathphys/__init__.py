"""Math functions, physical constants and auxiliary functions."""


# TODO: update subpackage with 2019 redefinition of SI base units and consts!
# see https://en.wikipedia.org/wiki/2019_redefinition_of_SI_base_units

import os as _os

from . import base_units
from . import units
from . import constants
from . import functions
from . import beam_optics
from . import imgproc

__all__ = [
    'base_units', 'units', 'constants', 'functions', 'beam_optics',
    'imgproc',
    ]

with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
