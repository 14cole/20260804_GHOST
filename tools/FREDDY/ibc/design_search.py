"""Qt-free inverse-stack and bounded material-recipe search services.

Requests capture inputs before dispatch. Search execution never reads widgets;
the caller supplies cancellation, progress, and optional evaluation adapters.
"""
from __future__ import annotations

from .mix_analysis import evaluate_mix_performance, build_mix_display
from dataclasses import dataclass
try:
    import numpy as np
except ImportError:  # The scalar FREDDY path remains available.
    np = None
import math
import heapq
import random
from collections import deque
import time
from pathlib import Path
from .compute import (
    INCH_TO_M,
    InverseCandidate,
    LayerConfig,
    LoadedLayer,
    MIX_RULE_LABELS,
    MaterialTable,
    MixCandidate,
    MixComponent,
    UncertaintyConfig,
    blend_density_gcc,
    build_uncertainty_scales,
    combine_mix,
    compute_angle_metrics_many,
    interp_components_on_grid,
    is_nominal_scale,
    mix_model_advisories,
    parts_to_fractions,
    prepare_layer_properties_many,
    prepare_layer_wave_terms_many,
    project_bounded_fractions,
    property_match_error,
    validate_sweep_coverage,
    weight_fractions_from_volume,
)
from .io import read_material_table
from .inverse_workflow import StopInverseSearch, search_identity
from .inverse_grid import DesignGrid


def score_inverse_candidate(target_freqs, target_angles, candidate_layers, wave_pol,
                            scales, score_mode, prepared_wave_terms=None, *,
                            stop_requested, statistics,
                            compute_metrics=compute_angle_metrics_many):
    """Score a complete candidate, observing cancellation between responses."""
    corner_means: list[float] = []
    nominal_mean: float | None = None
    for t_scale, e_scale, m_scale in scales:
        values: list[float] = []
        for angle_deg in target_angles:
            if stop_requested():
                raise StopInverseSearch()
            prepared = (
                prepared_wave_terms.get((angle_deg, e_scale, m_scale))
                if prepared_wave_terms is not None
                else None
            )
            metrics = compute_metrics(
                target_freqs,
                angle_deg,
                candidate_layers,
                wave_pol,
                thickness_scale=t_scale,
                eps_scale=e_scale,
                mu_scale=m_scale,
                prepared_wave_terms=prepared,
            )
            values.extend(metrics["metal_loss_db"])
        mean_db, _mn, _mx = statistics(values)
        corner_means.append(mean_db)
        if abs(t_scale - 1.0) < 1e-12 and abs(e_scale - 1.0) < 1e-12 and abs(m_scale - 1.0) < 1e-12:
            nominal_mean = mean_db

    if not corner_means:
        raise ValueError("No corner scores computed for inverse candidate.")
    if nominal_mean is None:
        nominal_mean = corner_means[0]

    worst_mean = max(corner_means)
    avg_mean = sum(corner_means) / len(corner_means)
    best_mean = min(corner_means)
    score_db = worst_mean if "worst" in score_mode.lower() else avg_mean
    return score_db, nominal_mean, worst_mean, avg_mean, best_mean


@dataclass(frozen=True)
class InverseSearchRequest:
    """Captured inputs for a background search, independent of widgets."""
    layer_snapshot: list[LayerConfig]
    target_freqs: list[float]
    target_angles: list[float]
    wave_pol: str
    uncertainty_cfg: UncertaintyConfig
    score_mode: str
    checkpoint: dict | None
    grid: DesignGrid
    skiprows: int
    top_n: int
    target_freq_desc: str
    a_start: float
    a_stop: float
    numpy_available: bool


def run_inverse_search(request: InverseSearchRequest, *, stop_requested, progress,
                       score_candidate, read_table=read_material_table,
                       compute_metrics=compute_angle_metrics_many,
                       checkpoint_callback=None, checkpoint_interval=30.0):
    """Evaluate captured inputs; callbacks carry cancellation/progress or pure calculations."""
    completed = {}
    layer_snapshot = request.layer_snapshot
    target_freqs = request.target_freqs
    target_angles = request.target_angles
    wave_pol = request.wave_pol
    uncertainty_cfg = request.uncertainty_cfg
    score_mode = request.score_mode
    checkpoint = request.checkpoint
    grid = request.grid
    skiprows = request.skiprows
    top_n = request.top_n
    target_freq_desc = request.target_freq_desc
    a_start = request.a_start
    a_stop = request.a_stop
    numpy_available = request.numpy_available

    from array import array
    import heapq
    identity = search_identity(layer_snapshot, target_freqs, target_angles, wave_pol,
                               uncertainty_cfg, score_mode, skiprows=skiprows)
    if checkpoint and checkpoint['identity'] != identity:
        raise ValueError("Inputs or material files changed. Start a new analysis instead of resuming.")
    def check_stop():
        if stop_requested():
            raise StopInverseSearch()
    scales = build_uncertainty_scales(uncertainty_cfg)
    table_cache: dict[str, MaterialTable] = {}

    def get_table(path_str: str) -> MaterialTable:
        key = str(Path(path_str))
        if key not in table_cache:
            table_cache[key] = read_table(Path(key), skiprows)
        return table_cache[key]

    def prepare_material_combo(
        chosen_files: list[str],
    ) -> tuple[bool, list[MaterialTable | None], list[MaterialTable | None]]:
        tables_0_local: list[MaterialTable | None] = []
        tables_90_local: list[MaterialTable | None] = []
        for i, layer in enumerate(layer_snapshot, start=1):
            if layer.is_sheet:
                tables_0_local.append(None)
                tables_90_local.append(None)
                continue
            table_0 = get_table(chosen_files[i - 1])
            try:
                validate_sweep_coverage(
                    target_freqs, table_0, f"inverse layer {i} 0deg/isotropic"
                )
            except Exception as exc:
                raise ValueError(f"Layer {i}: {exc}") from exc
            table_90: MaterialTable | None = None
            if layer.anisotropic:
                table_90 = get_table(layer.file_90deg)
                try:
                    validate_sweep_coverage(
                        target_freqs, table_90, f"inverse layer {i} 90deg"
                    )
                except Exception as exc:
                    raise ValueError(f"Layer {i}, 90 deg: {exc}") from exc
            tables_0_local.append(table_0)
            tables_90_local.append(table_90)
        return True, tables_0_local, tables_90_local

    # Validated material tables are reused for every
    # candidate. Each candidate is fully described by its per-layer
    # thicknesses (bulk) and resistances (sheets).
    def build_loaded_layers(
        thicknesses: list[float], resistances: list[float]
    ) -> list[LoadedLayer]:
        out: list[LoadedLayer] = []
        for i, layer in enumerate(layer_snapshot):
            if layer.is_sheet:
                out.append(
                    LoadedLayer(
                        thickness_m=0.0,
                        anisotropic=False,
                        polarization_deg=0.0,
                        table_0deg=None,
                        table_90deg=None,
                        is_sheet=True,
                        sheet_resistance=resistances[i],
                    )
                )
            else:
                out.append(
                    LoadedLayer(
                        thickness_m=thicknesses[i] * INCH_TO_M,
                        anisotropic=layer.anisotropic,
                        polarization_deg=layer.polarization_deg,
                        table_0deg=tables_0[i],
                        table_90deg=tables_90[i],
                    )
                )
        return out


    chosen_files = ['' if layer.is_sheet else layer.file_0deg for layer in layer_snapshot]
    _coverage_ok, tables_0, tables_90 = prepare_material_combo(chosen_files)
    base_thick, base_rs = grid.design(0)
    prepared_inverse_wave_terms = {}
    if numpy_available:
        reference_layers = build_loaded_layers(base_thick, base_rs)
        prepared_properties = prepare_layer_properties_many(
            target_freqs, reference_layers
        )
        for _t_scale, e_scale, m_scale in scales:
            for angle_deg in target_angles:
                key = (angle_deg, e_scale, m_scale)
                if key not in prepared_inverse_wave_terms:
                    prepared_inverse_wave_terms[key] = (
                        prepare_layer_wave_terms_many(
                            target_freqs,
                            angle_deg,
                            reference_layers,
                            wave_pol,
                            eps_scale=e_scale,
                            mu_scale=m_scale,
                            prepared_properties=prepared_properties,
                        )
                    )


    # Store five scores per completed design, rather than millions of
    # proposal objects or full response grids. The grid index recovers
    # the exact design and permits resumption without repeating scores.
    score_rows = checkpoint['score_rows'] if checkpoint else array('d')
    next_index = checkpoint['next_index'] if checkpoint else 0
    if checkpoint and (checkpoint.get('total') != grid.total or
                       type(next_index) is not int or not 0 <= next_index <= grid.total or
                       len(score_rows) != 5 * next_index):
        raise ValueError('Checkpoint does not match the design grid.')
    last_checkpoint = -math.inf

    def publish_checkpoint(*, force=False, plots_complete=False):
        nonlocal last_checkpoint
        if checkpoint_callback is None:
            return
        now = time.monotonic()
        if not force and now - last_checkpoint < checkpoint_interval:
            return
        if search_identity(layer_snapshot, target_freqs, target_angles, wave_pol,
                           uncertainty_cfg, score_mode, skiprows=skiprows) != identity:
            raise ValueError('Material files changed during the analysis. Run again with stable inputs.')
        checkpoint_callback(dict(identity=identity, score_rows=score_rows,
                                 next_index=next_index, total=grid.total,
                                 plots_complete=plots_complete))
        last_checkpoint = now

    publish_checkpoint(force=True)
    try:
        while next_index < grid.total:
            check_stop()
            thicknesses, resistances = grid.design(next_index)
            scores = score_candidate(
                target_freqs, target_angles, build_loaded_layers(thicknesses, resistances),
                wave_pol, scales, score_mode, prepared_inverse_wave_terms)
            if len(scores) != 5 or not all(math.isfinite(v) for v in scores):
                raise ValueError(f'Combination {next_index + 1}: incomplete or nonfinite scores.')
            score_rows.extend(scores)
            next_index += 1
            progress(next_index, grid.total, 'Analyzing')
            publish_checkpoint()
    except StopInverseSearch:
        pass
    top_candidates = []
    for index in heapq.nsmallest(min(top_n, next_index), range(next_index),
                                key=lambda i: (score_rows[5*i], i)):
        thicknesses, resistances = grid.design(index)
        top_candidates.append(InverseCandidate(*score_rows[5*index:5*index+5],
                                               thicknesses, chosen_files[:], resistances))
    completed.update(identity=identity, score_rows=score_rows, next_index=next_index,
                     total=grid.total, plots_complete=False)
    publish_checkpoint(force=True)
    progress(next_index, grid.total, 'Preparing comparison plots')
    inverse_samples: list[list[list[float]]] = []
    try:
        for cand in top_candidates:
            cand_layers = build_loaded_layers(cand.thickness_in, cand.sheet_resistance_ohm)
            freq_samples = [[] for _ in target_freqs]
            for t_scale, e_scale, m_scale in scales:
                for angle_deg in target_angles:
                    check_stop()
                    metrics = compute_metrics(
                        target_freqs,
                        angle_deg,
                        cand_layers,
                        wave_pol,
                        thickness_scale=t_scale,
                        eps_scale=e_scale,
                        mu_scale=m_scale,
                        prepared_wave_terms=prepared_inverse_wave_terms.get(
                            (angle_deg, e_scale, m_scale)
                        ),
                    )
                    for fi, val in enumerate(metrics["metal_loss_db"]):
                        freq_samples[fi].append(val)
            inverse_samples.append(freq_samples)

    except StopInverseSearch:
        inverse_samples = []
    completed['plots_complete'] = len(inverse_samples) == len(top_candidates)
    if search_identity(layer_snapshot, target_freqs, target_angles, wave_pol,
                       uncertainty_cfg, score_mode, skiprows=skiprows) != identity:
        raise ValueError("Material files changed during the analysis. Run again with stable inputs.")
    complete = next_index == grid.total
    publish_checkpoint(force=True, plots_complete=completed['plots_complete'])
    status = ('All combinations analyzed' if complete else
              'Stopped; analysis is incomplete. Resume to analyze the remaining combinations')
    if not completed['plots_complete']:
        status += '. Comparison plots unfinished; Resume completes them'
    best_text = (f'Best score: {top_candidates[0].score_db:.3f} dB' if top_candidates
                 else 'No combination was fully evaluated.')
    msg = (f'{status}.\n'
           f'Evaluated: {next_index:,} of {grid.total:,} combinations\n'
           f'Objective: {score_mode}\n'
           f'Region: {target_freq_desc}, {a_start:g}-{a_stop:g} deg, pol={wave_pol.upper()}\n'
           f'Tolerance/nominal cases per combination: {len(scales)}\n'
           f'Allowed values: {grid.description()}\n'
           'Values advance from each minimum by its step; an off-step maximum is excluded.\n'
           f'{best_text}\n'
           f'Retained {len(top_candidates)} best candidates for comparison. '
           'Keep best does not limit the combinations analyzed.')
    return (top_candidates, msg, [float(v) for v in target_freqs], inverse_samples), completed


@dataclass(frozen=True)
class MixSearchRequest:
    """Captured inputs for a background search, independent of widgets."""
    uncertainty_cfg: UncertaintyConfig
    comp_snapshot: list[dict]
    target_freqs: list[float]
    rule_norm: str
    property_mode: bool
    performance_mode: bool
    target: dict | None
    thickness_in: float
    performance_config: dict | None
    score_mode: str
    search_seed: int | None
    lower: list[float]
    upper: list[float]
    max_evals: int
    top_n: int
    refine: bool
    prop_desc: str
    target_desc: str
    numpy_available: bool


MIX_REFINE_MAX_EVALS = 300
MAX_MIX_RETAINED = 100


class StopMixSearch(RuntimeError):
    """A requested stop, distinct from an invalid material or calculation error."""


def run_mix_search(request: MixSearchRequest, *, evaluate_performance=evaluate_mix_performance,
                   build_display=build_mix_display,
                   read_table=read_material_table, optimizer=None,
                   stop_requested=lambda: False, progress=lambda *_args: None):
    """Evaluate captured inputs; callbacks carry cancellation/progress or pure calculations."""
    if request.refine and optimizer is None:
        from scipy import optimize as optimizer
    uncertainty_cfg = request.uncertainty_cfg
    comp_snapshot = request.comp_snapshot
    target_freqs = request.target_freqs
    rule_norm = request.rule_norm
    property_mode = request.property_mode
    performance_mode = request.performance_mode
    target = request.target
    thickness_in = request.thickness_in
    performance_config = request.performance_config
    score_mode = request.score_mode
    search_seed = request.search_seed
    lower = request.lower
    upper = request.upper
    max_evals = request.max_evals
    top_n = request.top_n
    refine = request.refine
    prop_desc = request.prop_desc
    target_desc = request.target_desc
    numpy_available = request.numpy_available

    if max_evals < 1 or not 1 <= top_n <= MAX_MIX_RETAINED:
        raise ValueError(f'Recipe samples must be positive; keep between 1 and {MAX_MIX_RETAINED} recipes.')

    def check_stop():
        if stop_requested():
            raise StopMixSearch('Material Mix search stopped. No new result was published.')

    check_stop()

    scales = build_uncertainty_scales(uncertainty_cfg)
    cache: dict[str, MaterialTable] = {}
    comps: list[dict] = []
    for index, component in enumerate(comp_snapshot, start=1):
        check_stop()
        path = component["file"]
        if not path:
            raise ValueError(f"Material {index}: property file is required.")
        key = str(Path(path))
        if key not in cache:
            cache[key] = read_table(Path(key), 0)
        comps.append({**component, "file": key, "table": cache[key]})

    base_components = [
        MixComponent(table=component["table"], parts=1.0)
        for component in comps
    ]
    eps_cols, mu_cols = interp_components_on_grid(
        base_components, target_freqs
    )
    component_files = [component["file"] for component in comps]
    densities = [component["density"] for component in comps]

    def score_fractions(fractions: list[float]):
        check_stop()
        table = combine_mix(
            target_freqs, eps_cols, mu_cols, fractions, rule_norm
        )
        corner_values: list[float] = []
        nominal: float | None = None
        for t_scale, eps_scale, mu_scale in scales:
            check_stop()
            if property_mode:
                error = property_match_error(
                    table.eps_r,
                    table.mu_r,
                    target["eps"],
                    target["mu"],
                    target["w_eps"],
                    target["w_mu"],
                    eps_scale,
                    mu_scale,
                )
            else:
                error = evaluate_performance(
                    table,
                    thickness_in,
                    performance_config,
                    thickness_scale=t_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                    check_stop=check_stop,
                )["gap"]
            corner_values.append(error)
            if is_nominal_scale(t_scale, eps_scale, mu_scale):
                nominal = error
        if nominal is None:
            nominal = corner_values[0]
        worst = max(corner_values)
        average = sum(corner_values) / len(corner_values)
        score = worst if "worst" in score_mode.lower() else average
        if not all(math.isfinite(v) for v in (score, nominal, worst, average, min(corner_values))):
            raise ValueError('Material recipe produced a nonfinite score.')
        return score, nominal, worst, average, min(corner_values)

    rng = random.Random(search_seed)
    current_amounts = [component["parts"] for component in comps]
    if sum(current_amounts) > 0:
        seed_values = parts_to_fractions(current_amounts)
    else:
        # An all-zero forward recipe is still meaningful while setting
        # inverse bounds; begin the search at the middle of those bounds.
        seed_values = [0.5 * (lo + hi) for lo, hi in zip(lower, upper)]
    recipe_seed = project_bounded_fractions(seed_values, lower, upper)
    fixed = (sum(hi - lo > 1e-12 for lo, hi in zip(lower, upper)) <= 1
             or abs(sum(lower) - 1.0) <= 1e-12 or abs(sum(upper) - 1.0) <= 1e-12)
    sample_budget = 1 if fixed else max_evals

    def fraction_key(values):
        return tuple(round(value, 10) for value in values)

    def proposals():
        # Bound duplicate bookkeeping and score immediately; a large requested
        # budget must not allocate the entire search before its first result.
        recent = deque([fraction_key(recipe_seed)])
        seen = set(recent)
        yield recipe_seed
        generated, attempts, duplicates = 1, 0, 0
        while generated < sample_budget and attempts < sample_budget * 30:
            check_stop()
            attempts += 1
            raw = [-math.log(max(rng.random(), 1e-15)) for _ in comps]
            proposal = project_bounded_fractions(parts_to_fractions(raw), lower, upper)
            key = fraction_key(proposal)
            if key in seen:
                duplicates += 1
                if duplicates >= max(100, 50 * len(comps)):
                    break
                continue
            duplicates = 0
            if len(recent) == 4096:
                seen.remove(recent.popleft())
            seen.add(key)
            recent.append(key)
            generated += 1
            yield proposal

    def scored_recipe(fractions):
        score, nominal, worst, average, best = score_fractions(fractions)
        return dict(score=score, nominal=nominal, worst=worst, avg=average,
                    best=best, fractions=list(fractions))

    kept = []
    kept_keys = set()
    invalid_count = 0
    evaluated = 0
    for fractions in proposals():
        check_stop()
        evaluated += 1
        try:
            candidate = scored_recipe(fractions)
        except ValueError:
            invalid_count += 1
            progress(evaluated, sample_budget, 'Sampling')
            continue
        key = fraction_key(fractions)
        entry = (-candidate['score'], -evaluated, candidate)
        if key not in kept_keys:
            if len(kept) < top_n:
                heapq.heappush(kept, entry)
                kept_keys.add(key)
            elif entry[:2] > kept[0][:2]:
                removed = heapq.heapreplace(kept, entry)
                kept_keys.remove(fraction_key(removed[2]['fractions']))
                kept_keys.add(key)
        progress(evaluated, sample_budget, 'Sampling')
    check_stop()
    candidates_raw = [entry[2] for entry in sorted(kept, key=lambda item: (-item[0], -item[1]))]
    if not candidates_raw:
        raise ValueError(
            "No physically valid recipe was found for these model "
            "assumptions and volume bounds."
        )

    refine_evals = 0
    if refine and numpy_available and not fixed:
        class RefinementBudgetReached(Exception):
            pass

        refined: list[dict] = []
        bounds = list(zip(lower, upper))
        constraint = {
            "type": "eq",
            "fun": lambda x: float(np.sum(x) - 1.0),
        }
        for candidate in candidates_raw:
            check_stop()
            best_refined = candidate
            local_evals = 0

            def objective(x: "np.ndarray") -> float:
                nonlocal refine_evals, local_evals, best_refined
                check_stop()
                if local_evals >= MIX_REFINE_MAX_EVALS:
                    raise RefinementBudgetReached()
                local_evals += 1
                refine_evals += 1
                progress(refine_evals, len(candidates_raw) * MIX_REFINE_MAX_EVALS, 'Refining')
                try:
                    fractions = project_bounded_fractions(
                        [float(value) for value in x], lower, upper
                    )
                    result = scored_recipe(fractions)
                    if result['score'] < best_refined['score']:
                        best_refined = result
                    return result['score']
                except ValueError:
                    return 1e12

            try:
                optimizer.minimize(
                    objective, np.asarray(candidate["fractions"], dtype=float),
                    method="SLSQP", bounds=bounds, constraints=(constraint,),
                    options={"ftol": 1e-9, "maxiter": 300, "disp": False},
                )
            except (RefinementBudgetReached, ValueError):
                pass
            refined.append(best_refined)
        refined.sort(key=lambda candidate: candidate["score"])
        candidates_raw = refined[:top_n]

    candidates: list[MixCandidate] = []
    plot_data: list[dict] = []
    for candidate in candidates_raw:
        check_stop()
        fractions = candidate["fractions"]
        candidates.append(
            MixCandidate(
                score_db=candidate["score"],
                nominal_mean_db=candidate["nominal"],
                worst_mean_db=candidate["worst"],
                avg_mean_db=candidate["avg"],
                best_mean_db=candidate["best"],
                fractions=list(fractions),
                thickness_in=thickness_in,
                component_files=list(component_files),
                rule=rule_norm,
                objective_kind="property" if property_mode else "performance",
                score_unit="%" if property_mode else performance_config["unit"],
                weight_fractions=weight_fractions_from_volume(
                    fractions, densities
                ),
                density_gcc=blend_density_gcc(fractions, densities),
            )
        )
        display_components = [
            MixComponent(table=comps[i]["table"], parts=fractions[i])
            for i in range(len(comps))
        ]
        plot_data.append(
            build_display(
                display_components,
                rule_norm,
                thickness_in,
                target_freqs,
                target=target if property_mode else None,
                performance=performance_config if performance_mode else None,
                densities=densities,
                component_names=[Path(path).name for path in component_files],
                check_stop=check_stop,
            )
        )

    best_candidate = candidates[0]
    recipe = " | ".join(
        f"{Path(path).name}: {100 * fraction:.1f}%"
        for path, fraction in zip(
            best_candidate.component_files, best_candidate.fractions
        )
    )
    advisories = " ".join(
        mix_model_advisories(rule_norm, best_candidate.fractions)
    )
    if property_mode:
        target_text = f"Target: {prop_desc}; {target_desc}"
        result_text = (
            f"Match error: {best_candidate.score_db:.3f}% "
            f"(nominal {best_candidate.nominal_mean_db:.3f}%, "
            f"worst {best_candidate.worst_mean_db:.3f}%)"
        )
    else:
        relation = "<=" if performance_config["direction"] == "at_most" else ">="
        angles = performance_config["angles"]
        status = "PASS" if best_candidate.worst_mean_db <= 0.0 else "MISS"
        target_text = (
            f"Target: {performance_config['label']} {relation} "
            f"{performance_config['target']:g} {performance_config['unit']}; "
            f"{target_desc}; angles {angles[0]:g}-{angles[-1]:g} deg; "
            f"pol={performance_config['wave_pol'].upper()}"
        )
        result_text = (
            f"Search score: {best_candidate.score_db:+.3f} "
            f"{best_candidate.score_unit}; certified worst-corner gap "
            f"{best_candidate.worst_mean_db:+.3f} {best_candidate.score_unit} "
            f"({status}; gap <= 0 passes); nominal gap "
            f"{best_candidate.nominal_mean_db:+.3f}"
        )
    message = (
        "Inverse material recipe search complete.\n"
        f"{target_text}\n"
        f"Model: {MIX_RULE_LABELS[rule_norm]}\n"
        f"Best volume recipe: {recipe}\n"
        f"{result_text}\n"
        f"Evaluated {evaluated} bounded recipe sample(s) of up to {sample_budget}; "
        f"{invalid_count} invalid under model; refinement evaluations "
        f"{refine_evals}.\nApplicability: {advisories}"
    )
    return candidates, plot_data, message
