"""Qt-free material recipe display data and stack-performance evaluation."""
from __future__ import annotations
import math
from .compute import (
    NUMPY_AVAILABLE, INCH_TO_M, LoadedLayer, MaterialTable, MixComponent,
    MIX_RULES, MIX_RULE_LABELS, prepare_layer_properties_many,
    compute_angle_metrics_many, mix_material_tables, parts_to_fractions,
    weight_fractions_from_volume, blend_density_gcc, normalize_mix_rule,
    mix_model_advisories, interp_complex_many, property_match_error_curve,
)

def mix_performance_values(metrics: dict[str, list[float]], config: dict) -> list[float]:
    values = [float(value) for value in metrics[config["metric_key"]]]
    if config["unit"] == "%":
        return [100.0 * 10.0 ** (value / 10.0) for value in values]
    return values


def mix_performance_gap(values: list[float], config: dict) -> float:
    if not values:
        raise ValueError("A performance target requires at least one grid point.")
    if config["direction"] == "at_most":
        return max(values) - config["target"]
    return config["target"] - min(values)


def evaluate_mix_performance(
    table: MaterialTable,
    thickness_in: float,
    config: dict,
    *,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> dict:
    layer = LoadedLayer(
        thickness_m=thickness_in * INCH_TO_M,
        anisotropic=False,
        polarization_deg=0.0,
        table_0deg=table,
        table_90deg=None,
    )
    freqs = list(table.freq_ghz)
    prepared_properties = (
        prepare_layer_properties_many(freqs, [layer]) if NUMPY_AVAILABLE else None
    )
    grid = [[0.0 for _angle in config["angles"]] for _freq in freqs]
    all_values: list[float] = []
    for angle_index, angle_deg in enumerate(config["angles"]):
        metrics = compute_angle_metrics_many(
            freqs,
            angle_deg,
            [layer],
            config["wave_pol"],
            thickness_scale=thickness_scale,
            eps_scale=eps_scale,
            mu_scale=mu_scale,
            prepared_properties=prepared_properties,
        )
        values = mix_performance_values(metrics, config)
        all_values.extend(values)
        for freq_index, value in enumerate(values):
            grid[freq_index][angle_index] = value
    return {
        **config,
        "freqs": freqs,
        "grid": grid,
        "gap": mix_performance_gap(all_values, config),
    }


def build_mix_display(
    components: list[MixComponent],
    rule: str,
    thickness_in: float,
    grid_ghz: list[float],
    target: dict | None = None,
    performance: dict | None = None,
    densities: list[float] | None = None,
    component_names: list[str] | None = None,
) -> dict:
    # Synthesize on the frequency grid selected in the Material Mix tab.
    # When a property target is given, also carry target curves and
    # per-frequency mismatch.
    # A model comparison at the band midpoint makes morphology sensitivity
    # visible rather than implying that one mixing law is ground truth.
    disp_table = mix_material_tables(components, rule, grid_ghz)
    lo, hi = disp_table.freq_ghz[0], disp_table.freq_ghz[-1]
    fractions = parts_to_fractions([component.parts for component in components])
    eps_tan = [
        (-value.imag / value.real) if value.real > 0 else math.nan
        for value in disp_table.eps_r
    ]
    mu_tan = [
        (-value.imag / value.real) if value.real > 0 else math.nan
        for value in disp_table.mu_r
    ]
    midpoint = disp_table.freq_ghz[len(disp_table.freq_ghz) // 2]
    comparison: list[dict] = []
    for candidate_rule in MIX_RULES:
        try:
            candidate_table = mix_material_tables(
                components, candidate_rule, [midpoint]
            )
        except Exception:
            continue
        comparison.append(
            {
                "rule": candidate_rule,
                "label": MIX_RULE_LABELS[candidate_rule].split(" — ")[0],
                "eps_re": candidate_table.eps_r[0].real,
                "eps_im": candidate_table.eps_r[0].imag,
                "mu_re": candidate_table.mu_r[0].real,
                "mu_im": candidate_table.mu_r[0].imag,
            }
        )
    density_values = (
        densities
        if densities is not None and len(densities) == len(fractions)
        else None
    )
    names = (
        component_names
        if component_names is not None
        and len(component_names) == len(fractions)
        else [f"Material {index}" for index in range(1, len(fractions) + 1)]
    )
    out = {
        "freqs": list(disp_table.freq_ghz),
        "eps_re": [v.real for v in disp_table.eps_r],
        "eps_im": [v.imag for v in disp_table.eps_r],
        "mu_re": [v.real for v in disp_table.mu_r],
        "mu_im": [v.imag for v in disp_table.mu_r],
        "loss_tan_eps": eps_tan,
        "loss_tan_mu": mu_tan,
        "thickness_in": thickness_in,
        "fractions": fractions,
        "component_names": names,
        "weight_fractions": weight_fractions_from_volume(
            fractions, density_values
        ) if density_values is not None else None,
        "density_gcc": blend_density_gcc(fractions, density_values)
        if density_values is not None
        else None,
        "model": normalize_mix_rule(rule),
        "advisories": mix_model_advisories(rule, fractions),
        "comparison_frequency": midpoint,
        "model_comparison": comparison,
    }
    if target is not None:
        sel = [
            (f, te, tm)
            for f, te, tm in zip(target["freqs"], target["eps"], target["mu"])
            if lo - 1e-9 <= f <= hi + 1e-9
        ]
        if sel:
            tg_f = [s[0] for s in sel]
            tg_eps = [s[1] for s in sel]
            tg_mu = [s[2] for s in sel]
            blend_eps = interp_complex_many(tg_f, disp_table.freq_ghz, disp_table.eps_r)
            blend_mu = interp_complex_many(tg_f, disp_table.freq_ghz, disp_table.mu_r)
            out["target_freqs"] = tg_f
            out["target_eps_re"] = [v.real for v in tg_eps]
            out["target_eps_im"] = [v.imag for v in tg_eps]
            out["target_mu_re"] = [v.real for v in tg_mu]
            out["target_mu_im"] = [v.imag for v in tg_mu]
            out["err_pct"] = property_match_error_curve(
                blend_eps, blend_mu, tg_eps, tg_mu, target["w_eps"], target["w_mu"]
            )
    if performance is not None:
        out["performance"] = evaluate_mix_performance(
            disp_table, thickness_in, performance
        )
    return out
