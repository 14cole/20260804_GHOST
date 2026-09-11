"""Verify measured claims and explicitly preserve known rejected experiments."""
from common import *
import unittest

def read(name):return json.loads((HERE/name).read_text())
def materials():
    from material_probe import KINDS
    return [read('material-'+kind+'.json') for kind in KINDS]

class EvidenceChecks(unittest.TestCase):
    def test_production_unchanged(self):
        baseline=read('baseline-manifest.json')
        self.assertEqual(source_hashes(),baseline['sources'])
        self.assertEqual(hashlib.sha256((ROOT/'airfoil.geo').read_bytes()).hexdigest(),baseline['airfoil_sha256'])
    def test_material_coverage_and_checked_sweeps(self):
        values=materials();self.assertEqual(len(values),14)
        count=0
        for value in values:
            for pol,r in value['algebra'].items():
                count+=1
                for mode in ('qr','fourier_checked'):
                    with self.subTest(kind=value['kind'],pol=pol,mode=mode):
                        self.assertLessEqual(r[mode]['original_backward'],1e-12)
                        self.assertLess(r[mode]['field_error'],1e-10)
            for mode,channels in value['assembly'].items():
                for error in channels.values():self.assertLess(error,1e-10)
        self.assertEqual(count,26)
        self.assertEqual(read('material-errors.json'),[])
    def test_known_fourier_failure_and_checked_fallback(self):
        value=read('material-sheet.json')['algebra']['TE']
        self.assertGreater(value['fourier']['original_backward'],1e-12)
        self.assertEqual(value['fourier_checked']['zero_columns'],1)
        self.assertEqual(value['fourier_checked']['fallback_columns'],2)
        at10=read('angular-scaling.json')[-1]
        self.assertEqual(at10['n'],28406);self.assertEqual(at10['qr']['rank'],99)
        self.assertLess(at10['qr']['relative_scaled_rhs_error'],1e-14)
        self.assertGreater(at10['fourier'][0]['scaled_relative_error'],1e-14)
        self.assertLess(at10['fourier'][1]['scaled_relative_error'],1e-14)
    def test_screened_table_holdouts(self):
        records=read('table-results.json')
        for r in records:
            if 'independent_relative_error' in r:self.assertLess(r['independent_relative_error'],2e-13)
        self.assertTrue(any(r['degree']==10 and 'rejected' in r for r in records))
        for r in read('partial-table-results.json'):self.assertLess(r['relative_error'],2e-13)
    def test_complete_airfoil_two_channel_assembly(self):
        base=read('assembly-airfoil-n384-f2.0-baseline-t4-tile0.json')
        got=read('assembly-airfoil-n384-f2.0-table12_paired-t4-tile0.json')
        errors={}
        for pol in ('VV','HH'):
            reference=np.asarray(base['runs'][0]['fields_by_pol'][pol])@np.array([1,1j])
            for run in got['runs']:
                value=np.asarray(run['fields_by_pol'][pol])@np.array([1,1j])
                error=difference(value,reference)
                errors[pol]=max(error,errors.get(pol,0))
                self.assertLess(error,1e-10)
                self.assertTrue(run['gate'])
        write('assembly-field-check.json',errors)
    def test_airfoil_compressed_operator_limits(self):
        for pol in ('TE','TM'):
            records=read('hierarchy-airfoil-n256-f2.0-'+pol+'.json')
            loose,tight=records['compressed_operator'][0],records['compressed_operator'][-1]
            self.assertGreater(loose['true_backward'],1e-12)
            self.assertLess(tight['true_backward'],1e-12)
            self.assertLess(tight['residual_bound'],1e-12)
            self.assertLess(tight['field_error'],1e-10)
            self.assertLess(tight['evidence']['bytes'],records['a_bytes']/3)
    def test_material_tolerance_limitations(self):
        tight=read('material-tightening.json')
        self.assertEqual(len(tight),26)
        for r in tight:
            inverse=r['coarse_inverse']
            self.assertNotIn('rejected',inverse)
            self.assertLess(inverse['field_error'],1e-10)
            self.assertLess(inverse['original_backward'],1e-12)
        self.assertEqual(sum('rejected' in r['compressed_operator'] for r in tight),4)
    def test_geometry_oracle_and_mesh_certification(self):
        r=read('coefficient-oracle-results.json');self.assertEqual(len(r),6)
        for value in r:self.assertLess(value['relative_error'],1e-13)
        local=read('mesh-fields-f1.0.json')
        self.assertTrue(local['local_certified'])
        self.assertTrue(local['local_certification']['passed'])
    def test_projection(self):
        for pol in ('TE','TM'):
            r=read('projection-airfoil-n256-f2.0-'+pol+'.json')
            for mode in ('project_basis','harmonic_observation'):self.assertLess(r[mode]['field_error'],1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
