"""Compatibility wrapper for the shared checked CPU factorization."""
from ghost_backend.execution.cpu import current_state
from ghost_backend.linalg.dense import DenseFactor


class CheckedLU(DenseFactor):
    def __init__(self, a, diagnostics, label, evidence):
        state = current_state()
        super().__init__(a, diagnostics, label, evidence,
            checkpoint=state.checkpoint if state is not None else None,
            force_double=True)
