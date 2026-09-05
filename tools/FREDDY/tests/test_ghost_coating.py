import cmath
import math
from dataclasses import replace
import unittest
from unittest import mock
from ibc.compute import MaterialTable, LoadedLayer, C0, ETA0
from ibc.ghost_coating import assess_scalar_coating, coating_report_text


def layer(eps=4-.1j, thickness=.03*.0254):
    return LoadedLayer(thickness, False, 0., MaterialTable([1.,18.], [eps,eps], [1.,1.]), None)


class ScalarCoatingTests(unittest.TestCase):
    def test_normal_matches_stack_but_oblique_error_is_not_hidden(self):
        report = assess_scalar_coating([1.,9.,18.], [layer()], [0.,60.])
        self.assertFalse(report['finite_body_accuracy_certified'])
        self.assertEqual(report['reference_surface'], 'outer air/coating interface')
        self.assertAlmostEqual(report['thickness_m'], .000762)
        for row in report['angles'][:2]:
            self.assertLess(row['max_absolute_complex_reflection_error'], 1e-13)
        # Independent short-circuited slab formulas for oblique plane waves.
        eps, theta = 4-.1j, math.radians(60.)
        for pol, row in zip(('TE','TM'), report['angles'][2:]):
            errors=[]
            for f in (1.,9.,18.):
                kd=2*math.pi*f*1e9/C0*.000762
                n=cmath.sqrt(eps)
                q=cmath.sqrt(eps-math.sin(theta)**2)
                z=1j*ETA0/n*cmath.tan(kd*n)
                zc=ETA0/q if pol=='TE' else ETA0*q/eps
                exact=1j*zc*cmath.tan(kd*q)
                z0=ETA0/math.cos(theta) if pol=='TE' else ETA0*math.cos(theta)
                errors.append(abs((z-z0)/(z+z0)-(exact-z0)/(exact+z0)))
            self.assertAlmostEqual(row['max_absolute_complex_reflection_error'],max(errors),places=12)
        self.assertGreater(report['angles'][-1]['max_absolute_complex_reflection_error'], .03)
        self.assertIn('not an RCS percent error',coating_report_text(report))

    def test_refined_frequency_grid_reduces_interpolation_error(self):
        coarse=assess_scalar_coating([1.,18.],[layer()],[0.])
        fine=assess_scalar_coating([1.+i*.25 for i in range(69)],[layer()],[0.])
        self.assertLess(fine['midpoint_absolute_reflection_error_max'],coarse['midpoint_absolute_reflection_error_max']/100)

    def test_tensor_and_excess_work_are_rejected_before_compute(self):
        with self.assertRaisesRegex(ValueError,'isotropic'):
            assess_scalar_coating([1.,18.],[replace(layer(),anisotropic=True)])
        with mock.patch('ibc.ghost_coating.compute_stack_impedance_many') as compute:
            with self.assertRaisesRegex(ValueError,'200,000'):
                assess_scalar_coating([1.+i*.0001 for i in range(20000)],[layer()])
            compute.assert_not_called()


if __name__=='__main__':
    unittest.main()
