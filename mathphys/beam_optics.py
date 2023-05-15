"""Beam Optics functions."""

import math as _math
import numpy as _np

from mathphys import constants as _c
from mathphys import units as _u


# NOTE: This function is used in siriuspy!
def beam_rigidity(**kwargs):
    """Beam rigidity."""
    # TODO: cleanup this function (and siriuspy Normalizer)
    electron_rest_energy_eV = _c.electron_rest_energy * _u.joule_2_eV
    electron_rest_energy_GeV = electron_rest_energy_eV * _u.eV_2_GeV

    # checks arguments and make necessary conversions
    if len(kwargs) != 1:
        raise Exception('beam rigidity accepts only one argument')
    for k in kwargs:
        if isinstance(kwargs[k], (list, tuple)):
            kwargs[k] = _np.array(kwargs[k])

    if 'brho' in kwargs:
        k = kwargs['brho'] / \
            (electron_rest_energy_eV/_c.light_speed)
        kwargs['gamma'] = _np.sqrt(1.0+k**2)
    if 'velocity' in kwargs:
        kwargs['beta'] = kwargs['velocity'] / _c.light_speed
    if 'beta' in kwargs:
        kwargs['gamma'] = 1.0 / \
            _np.sqrt((1.0 + kwargs['beta'])*(1.0 - kwargs['beta']))
    if 'gamma' in kwargs:
        kwargs['energy'] = kwargs['gamma'] * electron_rest_energy_GeV

    energy = kwargs['energy']
    gamma = kwargs['gamma'] if 'gamma' in kwargs else \
        energy / electron_rest_energy_GeV

    if isinstance(energy, _np.ndarray):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            with _np.errstate(divide='ignore', invalid='ignore'):
                beta = _np.sqrt(((gamma-1.0)/gamma)*((gamma+1.0)/gamma))
                beta[gamma == 0] = 0
    else:
        if energy < electron_rest_energy_GeV:
            beta = 0.0
        else:
            beta = _math.sqrt(((gamma-1.0)/gamma)*((gamma+1.0)/gamma))
    velocity = kwargs['velocity'] if 'velocity' in kwargs else \
        _c.light_speed * beta
    brho = kwargs['brho'] if 'brho' in kwargs else \
        beta * (energy * _u.GeV_2_eV) / _c.light_speed

    return brho, velocity, beta, gamma, energy


def calc_number_of_electrons(energy, current, circumference):
    """Calculate numbe of electrons.

    energy:        beam energy [GeV]
    current:       beam current [mA]
    circumference: ring circumference [m]
    """
    _, velocity, *_ = beam_rigidity(energy=energy)
    mA = 1e-3 / _c.elementary_charge * circumference / velocity
    Np = current * mA
    return Np
