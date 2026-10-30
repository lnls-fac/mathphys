"""Math functions, physical constants and auxiliary functions."""


# TODO: update subpackage with 2019 redefinition of SI base units and consts!
# see https://en.wikipedia.org/wiki/2019_redefinition_of_SI_base_units

import os as _os

from . import (
    algorithms,
    base_units,
    beam_optics,
    constants,
    functions,
    imgproc,
    units
)

__all__ = [
    'base_units', 'units', 'constants', 'functions', 'beam_optics',
    'imgproc', 'algorithms',
    ]

with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()
