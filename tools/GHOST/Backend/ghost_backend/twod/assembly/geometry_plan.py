"""Explicit geometry preparation for repeated queries on an immutable mesh."""
import numpy as np


class AssemblyGeometry:
    def __init__(self, mesh):
        self.mesh = mesh
        self.elements = tuple(mesh.elements)
        self.centers = np.asarray([e.center for e in self.elements], float).reshape(-1, 2)
        self.lengths = np.asarray([e.length for e in self.elements], float)
        self.node_ids = np.asarray([e.node_ids for e in self.elements], int).reshape(-1, 2)
        self.p0 = np.asarray([e.p0 for e in self.elements], float).reshape(-1, 2)
        self.segments = np.asarray([e.p1-e.p0 for e in self.elements], float).reshape(-1, 2)
        self.normals = np.asarray([e.normal for e in self.elements], float).reshape(-1, 2)
        if self.elements:
            points = np.asarray([p for e in self.elements for p in (e.p0, e.p1)])
            self.domain_upper = float(np.linalg.norm(np.ptp(points, axis=0)))*(1+1e-12)+1e-12
        else:
            self.domain_upper = 0.
        for array in (self.centers, self.lengths, self.node_ids, self.p0, self.segments, self.normals):
            array.flags.writeable = False

    def validate(self, mesh):
        if mesh is not self.mesh or len(mesh.elements) != len(self.elements):
            raise ValueError('Prepared assembly geometry belongs to another mesh.')
