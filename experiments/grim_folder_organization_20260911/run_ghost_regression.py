"""Run GHOST regressions with the separately checked coated-sphere case excluded."""
from pathlib import Path
import ctypes
import os
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
BACKEND=ROOT/'tools/GHOST/ghost_backend'
os.chdir(BACKEND)
sys.path[:0]=[str(ROOT),str(BACKEND.parent),str(BACKEND),str(BACKEND/'tests')]
ctypes.windll.kernel32.SetErrorMode(3)
def cases(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from cases(test)
        else:yield test
suite=unittest.TestSuite(test for test in cases(unittest.defaultTestLoader.discover('tests'))
    if test.id()!='test_bor_physics_regression.AnalyticSphereRegressionTests.test_lossy_coated_pec_sphere_matches_mie')
with (Path(__file__).parent/'ghost-final-tests.log').open('w',encoding='utf-8') as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
print('GHOST tests:',result.testsRun,'failures:',len(result.failures),'errors:',len(result.errors),'skips:',len(result.skipped))
raise SystemExit(not result.wasSuccessful())
