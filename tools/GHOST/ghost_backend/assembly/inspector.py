"""Read-only complex contribution inspection of a sealed Assembly plan."""
from collections import OrderedDict
from pathlib import Path
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


def _stored_complex_sample(path, frequency, azimuth, elevation, cancel_check=lambda: False,
                           *, return_channels=False):
    """Read the same coherent fields and channel aliases accepted by Assembly."""
    from ghost_backend.assembly.fields import (
        _load_grim_sample,
        _canonical_3d_channel_indices,
        _require_linear_quantity,
    )
    data = _load_grim_sample(path, frequency, azimuth, elevation, cancel_check)
    _require_linear_quantity(data, "Inspector body", "sigma_3d")
    channels, indices = _canonical_3d_channel_indices(data["polarizations"], "Inspector body")
    sample = np.asarray(data["_amp"][0, 0, 0, indices], complex)
    return (sample, channels) if return_channels else sample


class ContributionInspector:
    """Bounded LRU of evaluated samples; toggles reuse their complex fields."""
    def __init__(self, max_bytes=16*1024**2):
        self.max_bytes = int(max_bytes)
        self.cache = OrderedDict()
        self.bytes = 0

    def evaluate(self, plan, frequency, azimuth, elevation, cancel_check=lambda: False):
        from ghost_backend.assembly.workflow import feature_assembly_plan_sha256
        from ghost_backend.execution.provenance import sha256_file
        from ghost_backend.assembly.fields import (
            sum_features,
            _prepared_line_placements_at_frequency,
            radar_frame_basis,
        )
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
        body, channels = _stored_complex_sample(
            plan.base_path, frequency, azimuth, elevation, cancel_check, return_channels=True)
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
            channel_fields = {"VV": radar[0,0], "HH": radar[1,1],
                              "VH": radar[0,1], "HV": radar[1,0]}
            fields.append([channel_fields[channel] for channel in channels])
        labels = (["Line "+str(item.get("line_id", i+1)) for i,item in enumerate(plan.line_placements)] +
                  ["Point "+str(item.get("placement_id", i+1)) for i,item in enumerate(plan.point_placements)])
        output = dict(body=body, fields=np.asarray(fields, complex).reshape(-1, len(channels)),
                      polarizations=channels, labels=labels, key=key)
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
