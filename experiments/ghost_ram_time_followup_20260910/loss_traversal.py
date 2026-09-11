"""Experimental attenuation-based pruning; original near-pair rules retained."""
from support import *
from owned_assembly import clone
from contextlib import contextmanager
from unittest.mock import patch

STATS={}
@contextmanager
def pruned(optical=32):
    STATS.clear();STATS.update(far_pairs=0,skipped_pairs=0,tiles=0,skipped_tiles=0)
    def record(original,kept):
        STATS['far_pairs']+=int(np.count_nonzero(original))
        STATS['skipped_pairs']+=int(np.count_nonzero(original & ~kept))
        STATS['tiles']+=1
        STATS['skipped_tiles']+=int(original.any() and not kept.any())
    candidate=clone(ops._assemble_linear_operator_matrices_multi,[
        ('            far_ij = {}',
         '''            if complex(k0).imag < 0:
                lower_distance = np.maximum(centre_dist-.5*(obs_len[:,None]+src_len[None,:]),0)
                far_evaluation = far_sym & (lower_distance*(-complex(k0).imag) < audit_optical)
            else:
                far_evaluation = far_sym
            far_ij = {}'''),
        ('fij = far_sym & pairs_ij','fij = far_evaluation & pairs_ij'),
        ('fji = far_sym & pairs_ji','fji = far_evaluation & pairs_ji'),
        ('            if not (any_ij or any_ji):',
         '''            with write_lock:
                audit_record(far_sym & (eligible_ij|eligible_ji),far_evaluation & (eligible_ij|eligible_ji))
            if not (any_ij or any_ji):'''),
        ('any_far = far_sym & (eligible_ij | eligible_ji)','any_far = far_evaluation & (eligible_ij | eligible_ji)')],
        dict(audit_optical=float(optical),audit_record=record))
    with patch.object(ops,'_assemble_linear_operator_matrices_multi',candidate),patch.object(rcs,'_assemble_linear_operator_matrices_multi',candidate):yield
