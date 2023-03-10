"""Constants module.

https://en.wikipedia.org/wiki/International_System_of_Units
"""

import math as _math

from . import base_units as _u


# temporary auxiliary derived units
_volt = (_u.kilogram * _u.meter**2) / (_u.ampere * _u.second**2)
_coulomb = _u.second * _u.ampere
_joule = _u.kilogram * _u.meter**2 / _u.second**2


# physical constants
# ==================

# --- exact by definition --

light_speed = 299792458 * (_u.meter / _u.second)

gas_constant = 8.314462618 * (_joule / _u.mole / _u.kelvin)

boltzmann_constant = 1.380649e-23 * (_joule / _u.kelvin)

avogadro_constant = 6.02214076e23 * (1 / _u.mole)

elementary_charge = 1.602176634e-19 * (_coulomb)

reduced_planck_constant = 1.054571817e-34 * (_joule * _u.second)

# --- measured ---

# 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?me|search_for=electron+mass
electron_mass = 9.1093837015e-31 * (_u.kilogram)

# 2022-03-19 - https://physics.nist.gov/cgi-bin/cuu/Value?mu0|search_for=vacuum+permeability
vacuum_permeability = 1.25663706212e-6 * \
    (_volt * _u.second / _u.ampere / _u.meter)

# --- derived ---

# [Kg̣*m^2/s^2] - derived
electron_rest_energy = electron_mass * _math.pow(light_speed, 2)

# [V·s/(A.m)]  - derived
vacuum_permitticity = 1.0/(vacuum_permeability * _math.pow(light_speed, 2))

# [T·m^2/(A·s)] - derived
vacuum_impedance = vacuum_permeability * light_speed

# [m] - derived
electron_radius = _math.pow(elementary_charge, 2) / \
    (4*_math.pi*vacuum_permitticity*electron_rest_energy)

_joule_2_eV = _joule / elementary_charge

# [m]/[GeV]^3 - derived
rad_cgamma = 4*_math.pi*electron_radius / \
    _math.pow(electron_rest_energy/elementary_charge/1.0e9, 3) / 3

# [m] - derived
Cq = (55.0/(32*_math.sqrt(3.0))) * (reduced_planck_constant) * \
    light_speed / electron_rest_energy

# [m^2/(s·GeV^3)] - derived
Ca = electron_radius*light_speed / \
    (3*_math.pow(electron_rest_energy*_joule_2_eV/1.0e9, 3))
