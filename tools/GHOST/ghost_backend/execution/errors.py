"""Narrow failure class for numerical backend retries; input errors propagate."""


class BackendNumericalError(RuntimeError):
    pass
