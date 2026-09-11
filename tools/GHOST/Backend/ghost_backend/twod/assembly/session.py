"""One owned system retained between adjacent TE/TM solves, never their LUs."""
from functools import wraps
import inspect
from ghost_backend.execution.runtime import ScopedValue

_SESSION = ScopedValue('ghost_2d_assembly_session', default=None)


class AssemblySession:
    def __init__(self):
        self.pending = None
        self.reuses = 0
        self.abort_event = None
        self.compressed_partner = None
        self.memory_storage = {}

    def checkpoint(self):
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError('Solve canceled by user.')

    def take(self, key, polarization):
        item, self.pending = self.pending, None
        if item is not None and polarization == 'TM' and item[0] == key:
            self.reuses += 1
            return item[1]
        return None

    def save(self, key, polarization, value):
        if polarization == 'TE':
            self.pending = (key, value)


def current_session():
    return _SESSION.get()


def shared_assembly(function):
    signature = inspect.signature(function)
    @wraps(function)
    def call(*args, **kwargs):
        if current_session() is not None:
            return function(*args, **kwargs)
        session = AssemblySession()
        session.abort_event = signature.bind(*args, **kwargs).arguments.get('abort_event')
        with _SESSION.override(session):
            try:
                result = function(*args, **kwargs)
                result.setdefault('metadata', {})['assembled_system_reuses'] = session.reuses
                return result
            finally:
                session.pending = None
                session.compressed_partner = None
                session.memory_storage.clear()
    return call


def system_key(mesh, infos, kind, obs_order, src_order):
    from ghost_backend.twod.assembly.kernels import mesh_key


    media = tuple((i.minus_region, i.plus_region, i.bc_kind, complex(i.k_minus),
                   complex(i.k_plus), complex(i.eps_minus), complex(i.eps_plus),
                   complex(i.mu_minus), complex(i.mu_plus), complex(i.robin_impedance)) for i in infos)
    return kind, mesh_key(mesh), media, int(obs_order), int(src_order)
