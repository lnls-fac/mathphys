#!/usr/bin/env python-sirius

"""Unitest modeule for constants.py."""

from unittest import TestCase

import mathphys.constants as consts


PUB_INTERFACE = (
    'light_speed',
    'gas_constant',
    'boltzmann_constant',
    'avogadro_constant',
    'elementary_charge',
    'reduced_planck_constant',
    'electron_mass',
    'vacuum_permeability',
    'electron_rest_energy',
    'vacuum_permitticity',
    'vacuum_impedance',
    'electron_radius',
    'rad_cgamma',
    'Cq',
    'Ca',
    )


def check_public_interface_namespace(namespace, valid_interface,
                                     checkdoc_flag=True,
                                     print_flag=True):
    """Check function used in unittests to test module's public interface.

    This function checks only static public interface symbols. It does not
    check those symbols that are created within class methods.
    """
    for name in namespace.__dict__:
        if checkdoc_flag:
            doc = getattr(name, '__doc__')
            if doc is None or len(doc) < 5:
                if print_flag:
                    print('"' + name + '" has an invalid docstring!')
                return False
        if not name.startswith('_') and name not in valid_interface:
            if print_flag:
                print('Invalid symbol: ', name)
            return False
    for name in valid_interface:
        if name not in namespace.__dict__:
            if print_flag:
                print('Missing symbol: ', name)
            return False
    return True


class TestConstants(TestCase):
    """Test constants."""

    def test_public_intercface(self):
        """Test module's public interface."""
        valid = check_public_interface_namespace(consts, PUB_INTERFACE)
        self.assertTrue(valid)
