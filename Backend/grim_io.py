import json
import cmath
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

C0 = 299_792_458.0
EPS = 1e-12
SOLVER_METADATA_SCHEMA = 'ghost.solver_metadata.v1'


def _json_safe(value: 'Any') -> 'Any':
    """Convert solver metadata to deterministic, standards-compliant JSON data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return {'__nonfinite_float__': (
            'nan' if math.isnan(value) else 'infinity' if value > 0.0
            else '-infinity'
        )}
    if isinstance(value, complex):
        return {
            '__complex__': {
                'real': _json_safe(float(value.real)),
                'imag': _json_safe(float(value.imag)),
            }
        }
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(',', ':'), ensure_ascii=False
            ),
        )
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def _solver_metadata_json(result: 'Dict[str, Any]') -> 'str':
    """Build the stable audit envelope stored alongside every solver export."""

    diagnostics = []
    diagnostic_keys = (
        'linear_residual',
        'linear_backward_error',
        'constraint_residual',
        'constraint_residual_norm',
        'condition_est',
    )
    for row in result.get('samples', []) or []:
        values = {
            key: row[key]
            for key in diagnostic_keys
            if key in row
        }
        if not values:
            continue
        diagnostics.append({
            'frequency_ghz': row.get('frequency_ghz'),
            'theta_inc_deg': row.get('theta_inc_deg'),
            'theta_scat_deg': row.get('theta_scat_deg'),
            **values,
        })
    diagnostics.sort(key=lambda row: (
        float(row.get('frequency_ghz', 0.0)),
        float(row.get('theta_inc_deg', 0.0)),
        float(row.get('theta_scat_deg', 0.0)),
    ))

    envelope = {
        'schema': SOLVER_METADATA_SCHEMA,
        'solver': result.get('solver', ''),
        'scattering_mode': result.get('scattering_mode', ''),
        'polarization': result.get('polarization', ''),
        'polarization_export': result.get('polarization_export', ''),
        'rcs_log_unit': result.get('rcs_log_unit', ''),
        'rcs_linear_quantity': result.get('rcs_linear_quantity', ''),
        'amplitude_convention': result.get('amplitude_convention', ''),
        'metadata': result.get('metadata', {}) or {},
        'sample_diagnostics': diagnostics,
    }
    return json.dumps(
        _json_safe(envelope),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    )


def _required_finite_sample_value(row: 'Dict[str, Any]', key: 'str') -> 'float':
    if key not in row:
        raise ValueError(f"Solver sample is missing required field '{key}'.")
    try:
        value = float(row[key])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Solver sample field '{key}' must be a finite numeric value; "
            f"got {row[key]!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Solver sample field '{key}' must be finite; got {row[key]!r}."
        )
    return value

def _canonical_user_polarization_label(label: 'Optional[str]') -> 'str':
    # Elevation-cut convention: z (out-of-plane) is horizontal, so TM == HH.
    text = str(label or '').strip().upper()
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    if not text:
        return 'TM'
    raise ValueError(
        f"Unsupported polarization '{label}'. Use TM/HH/H/HORIZONTAL or TE/VV/V/VERTICAL."
    )

def _primary_alias_for_user_polarization(label: 'str') -> 'str':
    return 'HH' if _canonical_user_polarization_label(label) == 'TM' else 'VV'

def _alias_list_for_user_polarization(label: 'str') -> 'List[str]':
    canonical = _canonical_user_polarization_label(label)
    return ['TM', 'HH', 'H', 'HORIZONTAL'] if canonical == 'TM' else ['TE', 'VV', 'V', 'VERTICAL']

def _ensure_grim_ext(path: 'str') -> 'str':
    return path if path.lower().endswith('.grim') else f'{path}.grim'

def _suffix_for_incidence(theta_inc_deg: 'float') -> 'str':
    value = f'{theta_inc_deg:.6f}'.rstrip('0').rstrip('.')
    value = value.replace('-', 'm').replace('.', 'p')
    return f'inc_{value or "0"}'

def _freq_value_to_hz(freq_value: 'float', unit: 'str' = 'GHz') -> 'float':
    unit_key = str(unit or 'GHz').strip().lower()
    scale = {
        'hz': 1.0,
        'khz': 1.0e3,
        'mhz': 1.0e6,
        'ghz': 1.0e9,
    }.get(unit_key, 1.0e9)
    return float(freq_value) * scale

def compute_dbke_from_linear(
    rcs_linear: 'float',
    frequency_value: 'float',
    frequency_unit: 'str' = 'GHz',
) -> 'float':
    """
    Convert linear 2D scattering width to absolute dBke.

    Absolute dBke uses the knife-edge normalization:
        dBke = 10*log10((2*pi/lambda) * sigma_2d)
             = 10*log10((2*pi*f/c0) * sigma_2d)
    """

    lin = float(rcs_linear)
    if (not math.isfinite(lin)) or lin <= 0.0:
        lin = EPS
    freq_hz = _freq_value_to_hz(frequency_value, unit=frequency_unit)
    if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
        raise ValueError('frequency_value must be a positive finite frequency.')
    return 10.0 * math.log10(((2.0 * math.pi * freq_hz) / C0) * lin)

def compute_linear_from_dbke(dbke_value: 'float', frequency_value: 'float', frequency_unit: 'str' = 'GHz') -> 'float':
    """Convert absolute dBke to linear 2D scattering width sigma_2d."""

    dbke = float(dbke_value)
    freq_hz = _freq_value_to_hz(frequency_value, unit=frequency_unit)
    if (not math.isfinite(freq_hz)) or freq_hz <= 0.0:
        raise ValueError('frequency_value must be a positive finite frequency.')
    return (C0 / (2.0 * math.pi * freq_hz)) * (10.0 ** (dbke / 10.0))

def _build_grid_for_samples(
    samples: 'List[Dict[str, Any]]',
    polarization: 'str',
    source_path: 'str' = '',
    history: 'str' = '',
    preserve_raw_complex_amplitude: 'bool' = True,
    rcs_log_unit: 'str' = 'dBke',
    rcs_linear_quantity: 'str' = 'sigma_2d',
    solver_metadata_json: 'str' = '',
) -> 'Dict[str, Any]':
    if not samples:
        raise ValueError('No samples available to export.')
    linear_quantity = str(rcs_linear_quantity).strip().lower()
    if linear_quantity not in {'sigma_2d', 'sigma_3d'}:
        raise ValueError(
            "rcs_linear_quantity must be 'sigma_2d' or 'sigma_3d'."
        )

    validated_rows = []
    for row_index, row in enumerate(samples):
        try:
            az = _required_finite_sample_value(row, 'theta_scat_deg')
            freq = _required_finite_sample_value(row, 'frequency_ghz')
            lin = _required_finite_sample_value(row, 'rcs_linear')
            amp_real = _required_finite_sample_value(row, 'rcs_amp_real')
            amp_imag = _required_finite_sample_value(row, 'rcs_amp_imag')
        except ValueError as exc:
            raise ValueError(f"Invalid solver sample at index {row_index}: {exc}") from exc
        if freq <= 0.0:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: frequency_ghz "
                f"must be positive; got {freq!r}."
            )
        if lin < 0.0:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: rcs_linear "
                f"must be non-negative; got {lin!r}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            amp_abs2 = amp_real * amp_real + amp_imag * amp_imag
        if not math.isfinite(amp_abs2):
            raise ValueError(
                f"Invalid solver sample at index {row_index}: the finite "
                "complex amplitude is too large to form finite physical power."
            )
        if linear_quantity == 'sigma_2d':
            k0 = 2.0 * math.pi * freq * 1.0e9 / C0
            expected_linear = amp_abs2 / (4.0 * k0)
        else:
            expected_linear = 4.0 * math.pi * amp_abs2
        if not math.isfinite(expected_linear):
            raise ValueError(
                f"Invalid solver sample at index {row_index}: normalized "
                "linear RCS is not finite."
            )
        consistency_tol = (
            2.0e-8 * max(expected_linear, lin)
            + np.finfo(float).tiny
        )
        if abs(lin - expected_linear) > consistency_tol:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: rcs_linear "
                f"{lin:.12g} is inconsistent with the complex amplitude under "
                f"{linear_quantity} normalization (expected "
                f"{expected_linear:.12g})."
            )
        validated_rows.append((az, freq, lin, amp_real, amp_imag))

    azimuths = np.asarray(sorted({row[0] for row in validated_rows}), dtype=float)
    elevations = np.asarray([0.0], dtype=float)
    frequencies = np.asarray(sorted({row[1] for row in validated_rows}), dtype=float)
    polarization_label = _canonical_user_polarization_label(polarization)
    polarization_primary = _primary_alias_for_user_polarization(polarization_label)
    polarizations = np.asarray([polarization_primary], dtype=str)

    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    rcs_phase = np.full(shape, np.nan, dtype=np.float32)
    rcs_power = np.full(shape, np.nan, dtype=np.float32)
    # Raw coherent fields stay float64.  A feature delta can be many orders of
    # magnitude below the two full fields being subtracted; float32 storage of
    # those operands corrupts the small complex difference even when each
    # full-object RCS plot looks unchanged.
    rcs_amp_real = np.full(shape, np.nan, dtype=np.float64) if preserve_raw_complex_amplitude else None
    rcs_amp_imag = np.full(shape, np.nan, dtype=np.float64) if preserve_raw_complex_amplitude else None

    az_index = {value: i for i, value in enumerate(azimuths)}
    f_index = {value: i for i, value in enumerate(frequencies)}

    for az, freq, lin, amp_real, amp_imag in validated_rows:
        idx = (az_index[az], 0, f_index[freq], 0)
        amp_value_raw = complex(amp_real, amp_imag)
        phase_value = (
            0.0 if amp_value_raw == 0.0j
            else float(cmath.phase(amp_value_raw))
        )

        existing_power = rcs_power[idx]
        if not np.isnan(existing_power):
            if abs(existing_power - lin) > EPS:
                raise ValueError(
                    f'Duplicate sample conflict at az={az}, el=0, f={freq}, pol={polarization}.'
                )
            existing_phase = rcs_phase[idx]
            if np.isfinite(existing_phase):
                # Angular distance modulo 2*pi, so +pi and -pi register as equal.
                # Tolerance is sized for the float32 storage round-trip on phase.
                two_pi = 2.0 * math.pi
                diff_angle = (phase_value - float(existing_phase) + math.pi) % two_pi - math.pi
                if abs(diff_angle) > 1e-5:
                    raise ValueError(
                        f'Duplicate amplitude conflict at az={az}, el=0, f={freq}, pol={polarization}.'
                    )
            continue

        # Exact physical nulls remain exact zeros.  Floors belong only in
        # logarithmic presentation, never in the stored linear quantity.
        rcs_power[idx] = 0.0 if lin == 0.0 else float(lin)
        rcs_phase[idx] = phase_value
        if preserve_raw_complex_amplitude:
            rcs_amp_real[idx] = float(amp_value_raw.real)
            rcs_amp_imag[idx] = float(amp_value_raw.imag)

    missing = np.argwhere(~np.isfinite(rcs_power))
    if missing.size:
        first = tuple(int(value) for value in missing[0])
        raise ValueError(
            "Solver samples do not form a complete rectangular "
            f"(angle x frequency x polarization) grid: {len(missing)} cell(s) "
            f"are missing; first missing index is {first}."
        )
    if not np.all(np.isfinite(rcs_phase)):
        raise ValueError("Solver sample phases are not finite after grid construction.")
    if preserve_raw_complex_amplitude and (
        not np.all(np.isfinite(rcs_amp_real))
        or not np.all(np.isfinite(rcs_amp_imag))
    ):
        raise ValueError(
            "Solver complex amplitudes are not finite after grid construction."
        )

    units_payload = {
        'azimuth': 'deg',
        'elevation': 'deg',
        'frequency': 'GHz',
        # 2D solver: sigma_2d scattering width, absolute dBke on display.
        # BoR solver: sigma_3d RCS in m^2, displayed directly as dBsm.
        'rcs_log_unit': str(rcs_log_unit),
        'rcs_linear_quantity': str(rcs_linear_quantity),
    }

    is_2d_width = linear_quantity == 'sigma_2d'
    if is_2d_width:
        phase_reference = (
            'origin=(0,0), convention=exp(+jwt); stored complex field is the '
            '2D layer-potential bare-integral amplitude B. The coefficient '
            'in u_s~exp(-j(kr-pi/4))/sqrt(8*pi*k*r)*A is A=j*B.'
        )
        complex_field_domain = '2d_layer_potential_bare_integral_amplitude_B'
        amplitude_convention = 'A_physical_asymptotic = +j * B_stored'
    else:
        phase_reference = (
            'origin=(0,0), convention=exp(+jwt), monostatic far-field amplitude'
        )
        complex_field_domain = 'solver_raw_far_field_amplitude'
        amplitude_convention = 'solver physical far-field amplitude'

    payload = {
        'azimuths': azimuths,
        'elevations': elevations,
        'frequencies': frequencies,
        'polarizations': polarizations,
        'polarization_alias_primary': polarization_label,
        'polarization_aliases_json': json.dumps(_alias_list_for_user_polarization(polarization_label)),
        'rcs_power': rcs_power,
        'rcs_phase': rcs_phase,
        'rcs_domain': 'power_phase',
        'power_domain': 'linear_rcs',
        'source_path': source_path,
        'history': history,
        'units': json.dumps(units_payload),
        'phase_reference': phase_reference,
        'amplitude_convention': amplitude_convention,
        'raw_complex_amplitude_preserved': bool(preserve_raw_complex_amplitude),
    }
    if solver_metadata_json:
        payload['solver_metadata_json'] = str(solver_metadata_json)
    if preserve_raw_complex_amplitude:
        payload['rcs_amp_real'] = rcs_amp_real
        payload['rcs_amp_imag'] = rcs_amp_imag
        payload['complex_field_domain'] = complex_field_domain
    return payload

def _validate_grim_payload(payload: 'Dict[str, Any]') -> 'None':
    """Reject malformed/non-finite grids before opening the destination file."""

    axes = {}
    for key in ('azimuths', 'elevations', 'frequencies'):
        try:
            axis = np.asarray(payload[key], dtype=float)
        except KeyError as exc:
            raise ValueError(f"GRIM payload is missing required axis '{key}'.") from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"GRIM axis '{key}' must be numeric.") from exc
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"GRIM axis '{key}' must be a non-empty 1-D array.")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"GRIM axis '{key}' contains NaN or infinite values.")
        if np.any(np.diff(axis) <= 0.0):
            raise ValueError(
                f"GRIM axis '{key}' must be strictly increasing.")
        axes[key] = axis

    if np.any(axes['frequencies'] <= 0.0):
        raise ValueError("GRIM frequencies must be positive.")

    try:
        polarizations = np.asarray(payload['polarizations']).astype(str)
    except KeyError as exc:
        raise ValueError("GRIM payload is missing 'polarizations'.") from exc
    if polarizations.ndim != 1 or polarizations.size == 0:
        raise ValueError("GRIM polarizations must be a non-empty 1-D array.")
    if any(not label.strip() for label in polarizations):
        raise ValueError("GRIM polarization labels cannot be empty.")
    if len(set(polarizations.tolist())) != polarizations.size:
        raise ValueError("GRIM polarization labels must be unique.")

    for key in (
            'rcs_domain', 'power_domain', 'source_path', 'history', 'units',
            'phase_reference', 'amplitude_convention'):
        if key not in payload:
            raise ValueError(f"GRIM payload is missing metadata {key!r}.")
        value = np.asarray(payload[key])
        if value.size != 1:
            raise ValueError(f"GRIM metadata {key!r} must be scalar.")
        if key not in ('source_path', 'history') \
                and not str(value.reshape(-1)[0]).strip():
            raise ValueError(f"GRIM metadata {key!r} cannot be empty.")
    try:
        units_metadata = json.loads(str(np.asarray(
            payload['units']).reshape(-1)[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("GRIM units metadata is not valid JSON.") from exc
    if not isinstance(units_metadata, dict):
        raise ValueError("GRIM units metadata must decode to an object.")

    expected_shape = (
        len(axes['azimuths']),
        len(axes['elevations']),
        len(axes['frequencies']),
        len(polarizations),
    )

    def finite_grid(key: 'str', *, nonnegative: 'bool' = False) -> 'np.ndarray':
        if key not in payload:
            raise ValueError(f"GRIM payload is missing required grid '{key}'.")
        try:
            grid = np.asarray(payload[key], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"GRIM grid '{key}' must be numeric.") from exc
        if grid.shape != expected_shape:
            raise ValueError(
                f"GRIM grid '{key}' shape {grid.shape} does not match axes "
                f"{expected_shape}; the grid is not rectangular/complete."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"GRIM grid '{key}' contains NaN or infinite values.")
        if nonnegative and np.any(grid < 0.0):
            raise ValueError(f"GRIM grid '{key}' contains negative linear power.")
        return grid

    stored_power = finite_grid('rcs_power', nonnegative=True)
    stored_phase = finite_grid('rcs_phase')

    has_real = 'rcs_amp_real' in payload
    has_imag = 'rcs_amp_imag' in payload
    if has_real != has_imag:
        raise ValueError(
            "GRIM complex amplitude must provide both rcs_amp_real and "
            "rcs_amp_imag."
        )
    if bool(payload.get('raw_complex_amplitude_preserved', False)) and not has_real:
        raise ValueError(
            "raw_complex_amplitude_preserved is true but amplitude grids are absent."
        )
    if has_real:
        if 'complex_field_domain' not in payload \
                or not str(np.asarray(
                    payload['complex_field_domain']).reshape(-1)[0]).strip():
            raise ValueError(
                "GRIM raw complex amplitude requires a nonempty "
                "complex_field_domain.")
        stored_real = finite_grid('rcs_amp_real')
        stored_imag = finite_grid('rcs_amp_imag')
        stored_amp = stored_real + 1j * stored_imag
        live = (
            np.abs(stored_real) > np.finfo(np.float32).tiny
        ) | (
            np.abs(stored_imag) > np.finfo(np.float32).tiny
        )
        if np.any(live):
            phase_error = np.abs(np.angle(np.exp(
                1j * (stored_phase[live] - np.angle(stored_amp[live]))
            )))
            if float(np.max(phase_error)) > 2.0e-5:
                raise ValueError(
                    "GRIM rcs_phase is inconsistent with its stored complex "
                    "amplitude."
                )

        quantity = str(
            units_metadata.get('rcs_linear_quantity', '')).strip().lower()
        with np.errstate(over='ignore', invalid='ignore'):
            amp_abs2 = stored_real * stored_real + stored_imag * stored_imag
            if quantity == 'sigma_3d':
                expected_power = 4.0 * math.pi * amp_abs2
            elif quantity == 'sigma_2d':
                k0 = 2.0 * math.pi * axes['frequencies'] * 1.0e9 / C0
                expected_power = amp_abs2 / (
                    4.0 * k0[None, None, :, None]
                )
            else:
                expected_power = None
        if expected_power is None:
            raise ValueError(
                "GRIM units.rcs_linear_quantity must be 'sigma_2d' or "
                "'sigma_3d' when complex amplitudes are stored."
            )
        if not np.all(np.isfinite(amp_abs2)) or not np.all(
            np.isfinite(expected_power)
        ):
            raise ValueError(
                "GRIM complex amplitude is too large to form finite physical "
                f"power under {quantity} normalization."
            )
        power_tolerance = (
            16.0 * np.finfo(np.float32).eps
            * np.maximum(expected_power, stored_power)
            + np.finfo(np.float32).tiny
        )
        if np.any(np.abs(stored_power - expected_power) > power_tolerance):
            raise ValueError(
                "GRIM rcs_power is inconsistent with its stored complex "
                f"amplitude under {quantity} normalization."
            )

    if 'combination_estimate_power' in payload:
        finite_grid('combination_estimate_power', nonnegative=True)

    if 'solver_metadata_json' in payload:
        try:
            decoded = json.loads(str(payload['solver_metadata_json']))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("solver_metadata_json is not valid JSON.") from exc
        if not isinstance(decoded, dict) or decoded.get('schema') != SOLVER_METADATA_SCHEMA:
            raise ValueError(
                "solver_metadata_json does not use the supported "
                f"{SOLVER_METADATA_SCHEMA!r} schema."
            )


def _save_grim_npz(payload: 'Dict[str, Any]', path: 'str') -> 'str':
    _validate_grim_payload(payload)
    out = _ensure_grim_ext(path)
    with open(out, 'wb') as f:
        save_payload = dict(
            azimuths=payload['azimuths'],
            elevations=payload['elevations'],
            frequencies=payload['frequencies'],
            polarizations=payload['polarizations'],
            polarization_alias_primary=payload.get('polarization_alias_primary', ''),
            polarization_aliases_json=payload.get('polarization_aliases_json', ''),
            rcs_power=payload['rcs_power'],
            rcs_phase=payload['rcs_phase'],
            rcs_domain=payload['rcs_domain'],
            power_domain=payload['power_domain'],
            source_path=payload['source_path'],
            history=payload['history'],
            units=payload['units'],
            phase_reference=payload['phase_reference'],
            amplitude_convention=payload.get('amplitude_convention', ''),
            raw_complex_amplitude_preserved=payload.get('raw_complex_amplitude_preserved', False),
        )
        if 'rcs_amp_real' in payload and 'rcs_amp_imag' in payload:
            # Coherent subtraction and feature placement can expose deltas far
            # below the full-object field. Never let a caller's float32 array
            # silently quantize the authoritative complex amplitude on disk.
            save_payload['rcs_amp_real'] = np.asarray(
                payload['rcs_amp_real'], dtype=np.float64
            )
            save_payload['rcs_amp_imag'] = np.asarray(
                payload['rcs_amp_imag'], dtype=np.float64
            )
            save_payload['complex_field_domain'] = payload.get('complex_field_domain', 'solver_raw_far_field_amplitude')
        # Optional statistical/engineering estimates stay separate from the
        # physical rcs_power/complex-amplitude pair.
        for key in ('combination_estimate_power',
                    'combination_estimate_mode',
                    'combination_estimate_semantics',
                    'solver_metadata_json',
                    'combine_role',
                    'combine_role_note',
                    'component_provenance_json',
                    'production_mesh_certification_json',
                    'source_body_mesh_certification_json',
                    'source_body_sha256',
                    'pattern_frame_convention',
                    'geometry_input_sha256',
                    'solver_source_sha256',
                    'runtime_environment_sha256',
                    'run_solve_spec_sha256',
                    'collection_source_sha256',
                    'requested_radar_grid_json',
                    'body_profile_rho_m',
                    'body_profile_z_m'):
            if key in payload:
                save_payload[key] = payload[key]
        np.savez_compressed(f, **save_payload)
    return out

def export_result_to_grim(
    result: 'Dict[str, Any]',
    output_path: 'str',
    polarization: 'Optional[str]' = None,
    source_path: 'str' = '',
    history: 'str' = '',
    preserve_raw_complex_amplitude: 'bool' = True,
) -> 'List[str]':
    samples = result.get('samples', []) or []
    if not samples:
        raise ValueError('No solver samples were returned, nothing to export.')

    pol = _canonical_user_polarization_label(
        polarization or result.get('polarization_export') or result.get('polarization') or 'TM'
    )
    mode = str(result.get('scattering_mode', 'monostatic')).strip().lower()
    # BoR results carry their own units (sigma_3d / dBsm); 2D defaults apply
    # when the keys are absent.
    unit_kwargs = {
        'rcs_log_unit': str(result.get('rcs_log_unit', 'dBke')),
        'rcs_linear_quantity': str(result.get('rcs_linear_quantity', 'sigma_2d')),
    }
    if mode != 'bistatic':
        payload = _build_grid_for_samples(
            samples,
            pol,
            source_path=source_path,
            history=history,
            preserve_raw_complex_amplitude=preserve_raw_complex_amplitude,
            **unit_kwargs,
        )
        payload['solver_metadata_json'] = _solver_metadata_json(result)
        return [os.path.abspath(_save_grim_npz(payload, output_path))]

    by_inc: 'Dict[float, List[Dict[str, Any]]]' = {}
    for row_index, row in enumerate(samples):
        try:
            inc = _required_finite_sample_value(row, 'theta_inc_deg')
        except ValueError as exc:
            raise ValueError(
                f"Invalid bistatic solver sample at index {row_index}: {exc}"
            ) from exc
        by_inc.setdefault(inc, []).append(row)

    rootspec = _ensure_grim_ext(output_path)
    root_no_ext = rootspec[:-5]
    written: 'List[str]' = []
    for inc in sorted(by_inc.keys()):
        payload = _build_grid_for_samples(
            by_inc[inc],
            pol,
            source_path=source_path,
            history=(history + f' | theta_inc_deg={inc:g}').strip(' |'),
            preserve_raw_complex_amplitude=preserve_raw_complex_amplitude,
            **unit_kwargs,
        )
        payload['solver_metadata_json'] = _solver_metadata_json(result)
        out = f'{root_no_ext}_{_suffix_for_incidence(inc)}.grim'
        written.append(os.path.abspath(_save_grim_npz(payload, out)))
    return written

def save_bor_az_el_grim(grid: 'Dict[str, Any]', output_path: 'str',
                        source_path: 'str' = '', history: 'str' = '',
                        channel_metadata: 'Dict[str, Any]' = None) -> 'List[str]':
    """
    Write a bor_dispatch.bor_az_el_grid radar-frame polarimetric grid as
    .grim files -- one per channel (VV, HH, VH), each with REAL azimuth and
    elevation axes (unlike the single-cut aspect exports).  sigma_3d in m^2
    (dBsm); complex amplitudes preserved.

    ``channel_metadata`` maps a channel name to a metadata dict stored in that
    channel's file as ``solver_metadata_json``.  The HPC driver uses it to
    carry each derived product's run attestation inside the artifact rather
    than in a sidecar beside it.
    """

    channel_metadata = dict(channel_metadata or {})

    az = np.asarray(grid['azimuths_deg'], dtype=float)
    el = np.asarray(grid['elevations_deg'], dtype=float)
    freqs = np.asarray(grid['frequencies_ghz'], dtype=float)
    rootspec = _ensure_grim_ext(output_path)
    root = rootspec[:-5]
    units_payload = {
        'azimuth': 'deg', 'elevation': 'deg', 'frequency': 'GHz',
        'rcs_log_unit': 'dBsm', 'rcs_linear_quantity': 'sigma_3d',
    }
    aliases = {'VV': ['TE', 'VV', 'V', 'VERTICAL'],
               'HH': ['TM', 'HH', 'H', 'HORIZONTAL'],
               'VH': ['VH', 'HV']}
    written: 'List[str]' = []
    for ch in ('VV', 'HH', 'VH'):
        amp = np.asarray(grid['amp'][ch], dtype=np.complex128)[..., None]
        expected_shape = (len(az), len(el), len(freqs), 1)
        if amp.shape != expected_shape:
            raise ValueError(
                f"BoR az/el channel {ch} shape {amp.shape} does not match "
                f"axes {expected_shape}."
            )
        if not np.all(np.isfinite(amp.real) & np.isfinite(amp.imag)):
            raise ValueError(f"BoR az/el channel {ch} contains NaN or infinity.")
        amp_real = amp.real.astype(np.float64)
        amp_imag = amp.imag.astype(np.float64)
        stored_amp = amp_real.astype(float) + 1j * amp_imag.astype(float)
        power = (4.0 * math.pi * np.abs(stored_amp) ** 2).astype(np.float32)
        phase = np.angle(stored_amp).astype(np.float32)
        payload = {
            'azimuths': az, 'elevations': el, 'frequencies': freqs,
            'polarizations': np.asarray([ch], dtype=str),
            'polarization_alias_primary': ch,
            'polarization_aliases_json': json.dumps(aliases[ch]),
            'rcs_power': power, 'rcs_phase': phase,
            'rcs_domain': 'power_phase', 'power_domain': 'linear_rcs',
            'source_path': source_path,
            'history': (history + f' | bor_az_el_grid axis_az='
                        f"{grid['axis_az_deg']:g} axis_el={grid['axis_el_deg']:g}"
                        ).strip(' |'),
            'units': json.dumps(units_payload),
            'phase_reference': 'origin=(0,0), convention=exp(+jwt), '
                               'monostatic radar-frame amplitude',
            'amplitude_convention': 'F physical far-field amplitude; sigma_3d=4*pi*|F|^2',
            'raw_complex_amplitude_preserved': True,
            'rcs_amp_real': amp_real,
            'rcs_amp_imag': amp_imag,
            'complex_field_domain': 'solver_raw_far_field_amplitude',
        }
        if ch in channel_metadata:
            payload['solver_metadata_json'] = _solver_metadata_json({
                'solver': 'bor_mom_rcs',
                'scattering_mode': 'monostatic',
                'polarization': ch,
                'polarization_export': ch,
                'rcs_log_unit': 'dBsm',
                'rcs_linear_quantity': 'sigma_3d',
                'amplitude_convention': (
                    'F physical far-field amplitude; sigma_3d=4*pi*|F|^2'
                ),
                'metadata': channel_metadata[ch],
                'samples': [],
            })
        written.append(os.path.abspath(_save_grim_npz(payload, f'{root}_{ch}')))
    return written


def _ensure_csv_ext(path: 'str') -> 'str':
    return path if path.lower().endswith('.csv') else f'{path}.csv'

def export_result_to_dbke_csv(
    result: 'Dict[str, Any]',
    output_path: 'str',
    source_path: 'str' = '',
    history: 'str' = '',
) -> 'str':
    """
    Export 2-D scattering-width samples with an absolute dBke column.

    This format is intentionally limited to ``sigma_2d``.  A BoR result stores
    three-dimensional RCS (``sigma_3d``), whose logarithmic unit is dBsm; writing
    it through a function and column named dBke would be physically misleading.
    """

    linear_quantity = str(
        result.get('rcs_linear_quantity', 'sigma_2d')
    ).strip().lower()
    log_unit = str(result.get('rcs_log_unit', 'dBke')).strip().lower()
    solver_name = str(result.get('solver', '')).strip().lower()
    if (
        linear_quantity != 'sigma_2d'
        or log_unit != 'dbke'
        or solver_name.startswith('bor')
    ):
        raise ValueError(
            "export_result_to_dbke_csv only accepts 2-D sigma_2d/dBke "
            "results; BoR sigma_3d results must be exported as dBsm."
        )

    samples = result.get('samples', []) or []
    if not samples:
        raise ValueError('No solver samples were returned, nothing to export.')

    out = _ensure_csv_ext(output_path)
    rows = sorted(
        samples,
        key=lambda row: (
            float(row.get('frequency_ghz', 0.0)),
            float(row.get('theta_inc_deg', 0.0)),
            float(row.get('theta_scat_deg', 0.0)),
        ),
    )
    header = [
        'frequency_ghz',
        'theta_inc_deg',
        'theta_scat_deg',
        'rcs_linear',
        'rcs_db',
        'dbke',
        'rcs_amp_real',
        'rcs_amp_imag',
        'rcs_amp_phase_deg',
        'source_path',
        'history',
    ]
    with open(out, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            lin = float(row.get('rcs_linear', 0.0))
            if not math.isfinite(lin) or lin < 0.0:
                raise ValueError('rcs_linear must be finite and non-negative.')
            freq_ghz = float(row.get('frequency_ghz', 0.0))
            if 'rcs_db' in row:
                rcs_db = float(row['rcs_db'])
            else:
                rcs_db = 10.0 * math.log10(max(lin, EPS))
            dbke = compute_dbke_from_linear(lin, freq_ghz, frequency_unit='GHz')
            vals = [
                f"{freq_ghz:.12g}",
                f"{float(row.get('theta_inc_deg', 0.0)):.12g}",
                f"{float(row.get('theta_scat_deg', 0.0)):.12g}",
                f"{lin:.12g}",
                f"{rcs_db:.12g}",
                f"{dbke:.12g}",
                f"{float(row.get('rcs_amp_real', 0.0)):.12g}",
                f"{float(row.get('rcs_amp_imag', 0.0)):.12g}",
                f"{float(row.get('rcs_amp_phase_deg', 0.0)):.12g}",
                source_path.replace(',', ';'),
                history.replace(',', ';'),
            ]
            f.write(','.join(vals) + '\n')
    return os.path.abspath(out)
