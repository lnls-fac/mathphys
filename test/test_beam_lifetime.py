
import unittest
import numpy
import mathphys.beam_lifetime as beam_lifetime


class TestLossRates(unittest.TestCase):

    def test_calc_elastic_loss_rate(self):
        loss_rate = beam_lifetime.calc_elastic_loss_rate(
            energy=3.0e9,
            aperture_ratio=0.4,
            acceptances=[0.5, 0.3],
            pressure=1.0e-9,
            betas=[20.0, 15.0],
        )
        self.assertAlmostEqual(loss_rate, 6.913893987477171e-11, 15)

    def test_calc_elastic_loss_rate_with_arrays(self):
        beta_x = numpy.array([20.0, 18.0, 16.0])
        beta_y = numpy.array([15.0, 10.0, 12.5])
        pressure = 1.0e-9*numpy.array([1.0, 1.2, 1.3])
        expected_results = 1.0e-10*numpy.array([
            0.691389398747717,
            0.606214559460527,
            0.740786854286091
        ])
        loss_rates = beam_lifetime.calc_elastic_loss_rate(
            energy=3.0e9,
            aperture_ratio=0.4,
            acceptances=[0.5, 0.3],
            pressure=pressure,
            betas=[beta_x, beta_y],
        )
        self.assertEqual(len(loss_rates), 3)
        for i in range(len(expected_results)):
            self.assertAlmostEqual(loss_rates[i], expected_results[i], 15)

    def test_calc_inelastic_loss_rate(self):
        loss_rate = beam_lifetime.calc_inelastic_loss_rate(
            energy_acceptance=0.05,
            pressure=1.0e-9
        )
        self.assertAlmostEqual(loss_rate, 2.420898310898740e-06, 15)

    def test_calc_inelastic_loss_rate_with_pressure_array(self):
        expected_results = [
            0.242089831089874e-5,
            0.290507797307849e-5,
            0.338925763525824e-5
        ]
        loss_rates = beam_lifetime.calc_inelastic_loss_rate(
            energy_acceptance=0.05,
            pressure=1.0e-9*numpy.array([1.0, 1.2, 1.4])
        )
        self.assertEqual(len(loss_rates), 3)
        for i in range(len(expected_results)):
            self.assertAlmostEqual(loss_rates[i], expected_results[i], 15)

    def test_calc_quantum_loss_rates(self):
        expected_results = (
            0.163746150615596,
            0.239642114195851,
            0.147082817097433
        )
        results = beam_lifetime.calc_quantum_loss_rates(
            natural_emittance=1.5,
            coupling=0.2,
            energy_spread=1.0,
            transverse_acceptances=[0.5, 0.4],
            energy_acceptance=0.5,
            radiation_damping_times=[2.0, 3.0, 1.5]
        )
        self.assertEqual(len(results), 3)
        for i in range(len(expected_results)):
            self.assertAlmostEqual(results[i], expected_results[i], 15)

    def test_calc_quantum_loss_rates_transverse(self):
        alpha_x, alpha_y = beam_lifetime.calc_quantum_loss_rates_transverse(
            natural_emittance=1.5,
            coupling=0.2,
            acceptances=[0.5, 0.4],
            radiation_damping_times=[2.0, 3.0]
        )
        self.assertAlmostEqual(alpha_x, 0.163746150615596, 15)
        self.assertAlmostEqual(alpha_y, 0.239642114195851, 15)

    def test_calc_quantum_loss_rate_longitudinal(self):
        alpha_s = beam_lifetime.calc_quantum_loss_rate_longitudinal(
            energy_spread=1.0,
            energy_acceptance=0.5,
            radiation_damping_time=1.5
        )
        self.assertAlmostEqual(alpha_s, 0.147082817097433, 15)

    def test_calc_quantum_loss_rates_with_acceptance_array(self):
        expected_results = (
            [0.163746150615596, 0.136343006234594],
            0.239642114195851,
            0.147082817097433
        )
        results = beam_lifetime.calc_quantum_loss_rates(
            natural_emittance=1.5,
            coupling=0.2,
            energy_spread=1.0,
            transverse_acceptances=[numpy.array([0.5, 0.4]), 0.4],
            energy_acceptance=0.5,
            radiation_damping_times=[2.0, 3.0, 1.5]
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 2)
        for i in range(2):
            self.assertAlmostEqual(results[0][i], expected_results[0][i], 15)
        for i in range(1, len(expected_results)):
            self.assertAlmostEqual(results[i], expected_results[i], 15)


def loss_rates_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLossRates)
    return suite


def get_suite():
    suite_list = []
    suite_list.append(loss_rates_suite())
    return unittest.TestSuite(suite_list)