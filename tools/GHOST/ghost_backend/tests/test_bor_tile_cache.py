"""Bounded cache isolation, canonical coefficient tiles, and mode-worker reuse."""
from pathlib import Path
import sys, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor.cache import TileCache, cache_scope, current_cache
from ghost_backend.bor.options import configured, option_scope, validate_options
from ghost_backend.bor import solver as bor
from ghost_backend.bor.tiled import primitive


class BorTileCacheTests(unittest.TestCase):
    def test_lru_limits_payload_and_entries(self):
        value=np.ones(8,complex)
        cache=TileCache(value.nbytes*2)
        cache.put('a',value)
        cache.put('b',value)
        self.assertIs(cache.get('a'),value)
        cache.put('c',value)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.evidence()['evictions'],1)
        self.assertLessEqual(cache.evidence()['peak_payload_bytes'],cache.budget)
        many=TileCache(100000)
        for i in range(5000):many.put(i,np.ones(1,dtype=np.uint8))
        self.assertEqual(many.evidence()['entries'],4096)
        many.put('oversize',np.ones(100001,dtype=np.uint8))
        self.assertIsNone(many.get('oversize'))

    def test_request_scope_restores_and_explicit_options_get_their_own_budget(self):
        @configured
        def request(fail=False):
            if fail:raise RuntimeError('stop')
            return {'budget':current_cache().budget}
        outer=TileCache(1234)
        with cache_scope(outer):
            result=request(bor_options=dict(factorization='compressed',tile_cache_mib=2))
            self.assertEqual(result['budget'],2*1024**2)
            self.assertIs(current_cache(),outer)
            with self.assertRaisesRegex(RuntimeError,'stop'):
                request(True,bor_options=dict(factorization='compressed'))
            self.assertIs(current_cache(),outer)
        self.assertIsNone(current_cache())

    def test_canonical_cross_tiles_match_dense_for_negative_modes_and_reordered_nodes(self):
        sp=bor.BorPecSolver(bor.sphere_generatrix(.035,10),1e9,gauss_order=3,medium=(2.5-.05j,1.))
        sq=bor.BorPecSolver(bor.sphere_generatrix(.02,8),1e9,gauss_order=3,medium=(2.5-.05j,1.))
        cross=bor.BorCrossOperators(sp,sq)
        cross.prepare(2)
        rows=np.array([0,sp.Nn,7,2,sp.Nn+7,7])
        cols=np.array([1,sq.Nn+1,6,3,6])
        cache=TileCache(1024**2)
        with cache_scope(cache):
            for m in (0,1,-1,2):
                for kind,method in [('T',cross.assemble_T),('P',cross.assemble_P)]:
                    expected=method(m,2)[np.ix_(rows,cols)]
                    query=primitive(cross,kind,m,2)
                    for _ in range(2):
                        np.testing.assert_allclose(query.get(rows,cols),expected,rtol=2e-11,
                            atol=2e-12*max(np.max(abs(expected)),1e-30))
        self.assertGreater(cache.hits,0)
        self.assertLessEqual(cache.peak,cache.budget)

    def test_cache_is_shared_by_mode_workers_and_preserves_complex_fields(self):
        kwargs=dict(points=bor.sphere_generatrix(.025,12),freq_hz=1e9,
                    thetas_deg=[0.,37.,90.,180.],gauss_order=3,workers=2,formulation='cfie')
        reference=bor.solve_bor(bor_options=dict(factorization='compressed',compression_tile=8,tile_cache_mib=0),**kwargs)
        actual=bor.solve_bor(bor_options=dict(factorization='compressed',compression_tile=8,tile_cache_mib=1),**kwargs)
        for key in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[key],reference[key],rtol=2e-10,atol=2e-12)
        self.assertGreater(actual['bor_tile_cache']['hits'],0)
        self.assertLessEqual(actual['bor_tile_cache']['peak_payload_bytes'],1024**2)
        self.assertEqual(reference['bor_tile_cache']['peak_payload_bytes'],0)
        self.assertIsNone(current_cache())

    def test_memory_plan_charges_one_shared_cache(self):
        with option_scope(validate_options(dict(factorization='compressed',tile_cache_mib=0))):
            without=bor.estimate_bor_dense_peak_gb(300,100,workers=4)
        with option_scope(validate_options(dict(factorization='compressed',tile_cache_mib=16))):
            with_cache=bor.estimate_bor_dense_peak_gb(300,100,workers=4)
        self.assertAlmostEqual(with_cache-without,16*1024**2/1e9)


if __name__=='__main__':unittest.main()
