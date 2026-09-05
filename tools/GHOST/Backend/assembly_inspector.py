"""Read-only complex contribution inspection of a sealed Assembly plan."""
from collections import OrderedDict
from pathlib import Path
import json
import zipfile
import numpy as np


def interference_metrics(body, contributions, gains=None):
    """Exact algebra within the coherent, independent-feature Assembly model.

    Removal change includes interference with every remaining component.
    Sensitivities hold geometry, illumination, and occlusion fixed.
    """
    fields = np.asarray(contributions, complex)
    gains = np.ones(len(fields), complex) if gains is None else np.asarray(gains, complex)
    if fields.ndim != 1 or gains.shape != fields.shape or not np.all(np.isfinite(fields)) or not np.all(np.isfinite(gains)) or not np.isfinite(body):
        raise ValueError("Finite complex feature fields and matching gains are required.")
    applied = fields * gains
    total = complex(body) + np.sum(applied)
    rest = total - applied
    cross = 8*np.pi*np.real(np.conj(rest)*applied)
    return dict(total=total, sigma_total=float(4*np.pi*abs(total)**2), applied=applied,
                relative_phase_deg=np.where((abs(rest)>1e-30)&(abs(applied)>1e-30), np.degrees(np.angle(applied*np.conj(rest))), np.nan),
                interference_m2=cross,
                removal_change_m2=4*np.pi*abs(applied)**2+cross,
                gain_derivative_m2=8*np.pi*np.real(np.conj(total)*fields*np.exp(1j*np.angle(gains))),
                phase_derivative_m2_per_deg=8*np.pi*np.real(np.conj(total)*1j*applied)*np.pi/180.)


def _stored_complex_sample(path, frequency, azimuth, elevation, cancel_check=lambda: False):
    """Read one exact radar sample, streaming archive gaps in bounded chunks."""
    with np.load(path, allow_pickle=False) as data:
        axes = [np.asarray(data[key]) for key in ("azimuths", "elevations", "frequencies", "polarizations")]
        units = json.loads(str(np.asarray(data["units"]).item()))
    if units.get("rcs_linear_quantity") != "sigma_3d" or units.get("frequency") != "GHz":
        raise ValueError("Inspector requires a 3-D body response in GHz and square metres.")
    fixed = []
    for axis, value in zip(axes[:3], (azimuth, elevation, frequency)):
        match = np.flatnonzero(np.isclose(axis.astype(float), value, rtol=0, atol=1e-10))
        if len(match) != 1:
            raise ValueError("Inspector requires an exact stored body azimuth/elevation/frequency sample.")
        fixed.append(int(match[0]))
    out = np.zeros(3, complex)
    with zipfile.ZipFile(path) as archive:
        for field, multiplier in (("rcs_amp_real", 1.), ("rcs_amp_imag", 1j)):
            if field+".npy" not in archive.namelist():
                raise ValueError("Inspector requires preserved complex body amplitudes; power alone is insufficient.")
            with archive.open(field+".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if shape != tuple(len(axis) for axis in axes) or dtype.kind != "f":
                    raise ValueError("Body complex array does not match its axes.")
                start = stream.tell()
                positions = []
                for channel, pol in enumerate(("VV", "HH", "VH")):
                    match = np.flatnonzero(axes[3].astype(str) == pol)
                    if not len(match) and pol == "VH":
                        raise ValueError("Body cross-polarization is not stored; the inspector will not infer it.")
                    if len(match) != 1:
                        raise ValueError("Body requires unique VV, HH, VH channels.")
                    index = np.ravel_multi_index(tuple(fixed)+(int(match[0]),), shape, order="F" if fortran else "C")
                    positions.append((start+int(index)*dtype.itemsize, channel))
                for offset, channel in sorted(positions):
                    while stream.tell() < offset:
                        if cancel_check():
                            raise InterruptedError("Inspection cancelled.")
                        if not stream.read(min(262144, offset-stream.tell())):
                            raise ValueError("Truncated body response.")
                    raw = stream.read(dtype.itemsize)
                    if len(raw) != dtype.itemsize:
                        raise ValueError("Truncated body response.")
                    out[channel] += multiplier*np.frombuffer(raw, dtype=dtype, count=1)[0]
    if not np.all(np.isfinite(out)):
        raise ValueError("Body sample has nonfinite complex amplitudes.")
    return out


class ContributionInspector:
    """Bounded LRU of evaluated samples; toggles reuse their complex fields."""
    def __init__(self, max_bytes=16*1024**2):
        self.max_bytes = int(max_bytes)
        self.cache = OrderedDict()
        self.bytes = 0

    def evaluate(self, plan, frequency, azimuth, elevation, cancel_check=lambda: False):
        from feature_workflow import feature_assembly_plan_sha256
        from workflow_provenance import sha256_file
        from feature_sum import (sum_features, _prepared_line_placements_at_frequency,
                                 radar_frame_basis)
        if not plan.prepared_plan_sha256 or feature_assembly_plan_sha256(plan) != plan.prepared_plan_sha256:
            raise ValueError("Assembly changed; validate it again before inspecting.")
        for source, expected in plan.prepared_source_sha256.items():
            if cancel_check():
                raise InterruptedError("Inspection cancelled.")
            if sha256_file(source) != expected:
                raise ValueError(f"Assembly source changed: {Path(source).name}. Validate again.")
        if any(Path(path).exists() for path in plan.prepared_absent_paths):
            raise ValueError("A feature manifest appeared after validation; validate again.")
        key = (plan.prepared_plan_sha256, float(frequency), float(azimuth), float(elevation))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        count = len(plan.line_placements)+len(plan.point_placements)
        if count > 20000:
            raise ValueError("Inspector supports up to 20,000 enabled feature instances per sample.")
        grid = plan.radar_grid
        for field, value in (("frequencies_ghz", frequency), ("azimuths_deg", azimuth), ("elevations_deg", elevation)):
            if not np.any(np.isclose(np.asarray(grid[field], float), value, atol=1e-10, rtol=0)):
                raise ValueError("Requested inspector sample lies outside the validated Assembly grid.")
        body = _stored_complex_sample(plan.base_path, frequency, azimuth, elevation, cancel_check)
        directions, basis = radar_frame_basis([azimuth], [elevation], grid["axis_az_deg"], grid["axis_el_deg"], grid.get("roll_deg", 0.))
        placements = _prepared_line_placements_at_frequency(plan.line_placements, frequency, {})
        result = sum_features(None, placements, directions, frequency,
                              normal_fn=plan.surface_normal_fn, points=plan.point_placements,
                              occluder=plan.occluder, cancel_check=cancel_check,
                              retain_feature_amplitudes=True)
        fields = []
        for feature in result["feature_amps"]:
            matrix = np.array([[feature["F_vv"][0], feature["F_vh"][0]],
                               [feature["F_vh"][0], feature["F_hh"][0]]], complex)
            radar = basis[0].T @ matrix @ basis[0]
            fields.append([radar[0,0], radar[1,1], radar[0,1]])
        labels = (["Line "+str(item.get("line_id", i+1)) for i,item in enumerate(plan.line_placements)] +
                  ["Point "+str(item.get("placement_id", i+1)) for i,item in enumerate(plan.point_placements)])
        output = dict(body=body, fields=np.asarray(fields, complex).reshape(-1, 3), labels=labels, key=key)
        for source, expected in plan.prepared_source_sha256.items():
            if cancel_check():
                raise InterruptedError("Inspection cancelled.")
            if sha256_file(source) != expected:
                raise ValueError("An Assembly source changed during inspection; validate again.")
        output["body"].setflags(write=False)
        output["fields"].setflags(write=False)
        size = body.nbytes + output["fields"].nbytes + sum(len(label)*4+128 for label in labels)
        while self.cache and (self.bytes+size > self.max_bytes or len(self.cache)>=8):
            _, (_, used) = self.cache.popitem(last=False)
            self.bytes -= used
        if size <= self.max_bytes:
            self.cache[key] = (output, size)
            self.bytes += size
        return output
