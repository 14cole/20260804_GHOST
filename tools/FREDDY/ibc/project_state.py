"""FREDDY project capture/restoration; file portability is handled by ibc.io."""
from __future__ import annotations

from .ui_controls import BooleanVar, StringVar
from .compute import LayerConfig, normalize_backing, normalize_wave_polarization
from .io import layer_config_from_dict, layer_config_to_dict
from .ui_options import (
    INVERSE_SCORE_MODE_OPTIONS,
    MIX_OBJECTIVE_FORWARD,
    MIX_OBJECTIVE_PERFORMANCE,
    MIX_OBJECTIVE_PROPERTY,
)


class ProjectStateMixin:
    """Translate between workspace controls and portable project dictionaries."""

    def _collect_project_state(self) -> dict[str, object]:
        controls: dict[str, object] = {
            "f_start": self.f_start_var.get(),
            "f_stop": self.f_stop_var.get(),
            "f_step": self.f_step_var.get(),
            "backing": self.backing_var.get(),
            "output": self.output_var.get(),
            "ibc_batch_layer": self.ibc_batch_layer_var.get(),
            "ibc_batch_start": self.ibc_batch_start_var.get(),
            "ibc_batch_stop": self.ibc_batch_stop_var.get(),
            "ibc_batch_step": self.ibc_batch_step_var.get(),
            "ibc_batch_unit": self.ibc_batch_unit_var.get(),
            "ibc_batch_output_dir": self.ibc_batch_output_dir_var.get(),
            "ibc_batch_prefix": self.ibc_batch_prefix_var.get(),
            "uncertainty": self.uncertainty_var.get(),
            "unc_t_pct": self.unc_t_pct_var.get(),
            "unc_eps_pct": self.unc_eps_pct_var.get(),
            "unc_mu_pct": self.unc_mu_pct_var.get(),
            "angle_f_start": self.angle_f_start_var.get(),
            "angle_f_stop": self.angle_f_stop_var.get(),
            "angle_f_step": self.angle_f_step_var.get(),
            "angle_start": self.angle_start_var.get(),
            "angle_stop": self.angle_stop_var.get(),
            "angle_step": self.angle_step_var.get(),
            "wave_pol": self.wave_pol_var.get(),
            "angle_output": self.angle_output_var.get(),
            "angle_compare_both": self.angle_compare_both.isChecked(),
            "angle_uncertainty": self.angle_uncertainty_var.get(),
            "angle_unc_t_pct": self.angle_unc_t_pct_var.get(),
            "angle_unc_eps_pct": self.angle_unc_eps_pct_var.get(),
            "angle_unc_mu_pct": self.angle_unc_mu_pct_var.get(),
            "thk_f_start": self.thk_f_start_var.get(),
            "thk_f_stop": self.thk_f_stop_var.get(),
            "thk_f_step": self.thk_f_step_var.get(),
            "thk_start": self.thk_start_var.get(),
            "thk_stop": self.thk_stop_var.get(),
            "thk_step": self.thk_step_var.get(),
            "thk_layer": self.thk_layer_var.get(),
            "thk_angle": self.thk_angle_var.get(),
            "thk_wave_pol": self.thk_wave_pol_var.get(),
            "thk_output": self.thk_output_var.get(),
            "thk_uncertainty": self.thk_uncertainty_var.get(),
            "thk_unc_t_pct": self.thk_unc_t_pct_var.get(),
            "thk_unc_eps_pct": self.thk_unc_eps_pct_var.get(),
            "thk_unc_mu_pct": self.thk_unc_mu_pct_var.get(),
            "heatmap_metric": self.heatmap_metric_var.get(),
            "uncertainty_view": self.uncertainty_view_var.get(),
            "cbar_auto": self.cbar_auto_var.get(),
            "cbar_min": self.cbar_min_var.get(),
            "cbar_max": self.cbar_max_var.get(),
            "slice_angle": self.slice_angle_var.get(),
            "slice_freq": self.slice_freq_var.get(),
            "inv_freq_mode": self.inv_freq_mode_var.get(),
            "inv_freq_list": self.inv_freq_list_var.get(),
            "inv_target_start": self.inv_target_start_var.get(),
            "inv_target_stop": self.inv_target_stop_var.get(),
            "inv_target_step": self.inv_target_step_var.get(),
            "inv_angle_start": self.inv_angle_start_var.get(),
            "inv_angle_stop": self.inv_angle_stop_var.get(),
            "inv_angle_step": self.inv_angle_step_var.get(),
            "inv_wave_pol": self.inv_wave_pol_var.get(),
            "inv_top_n": self.inv_top_n_var.get(),
            "inv_percentile": self.inv_percentile_var.get(),
            "inv_uncertainty": self.inv_uncertainty_var.get(),
            "inv_unc_t_pct": self.inv_unc_t_pct_var.get(),
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var.get(),
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var.get(),
            "inv_score_mode": self.inv_score_mode_var.get(),
            "mix_rule": self.mix_rule_var.get(),
            "mix_objective": self.mix_objective_var.get(),
            "mix_prop_source": self.mix_prop_source_var.get(),
            "mix_prop_eps_re": self.mix_prop_eps_re_var.get(),
            "mix_prop_eps_im": self.mix_prop_eps_im_var.get(),
            "mix_prop_mu_re": self.mix_prop_mu_re_var.get(),
            "mix_prop_mu_im": self.mix_prop_mu_im_var.get(),
            "mix_prop_file": self.mix_prop_file_var.get(),
            "mix_prop_weps": self.mix_prop_weps_var.get(),
            "mix_prop_wmu": self.mix_prop_wmu_var.get(),
            "mix_perf_metric": self.mix_perf_metric_var.get(),
            "mix_perf_target": self.mix_perf_target_var.get(),
            "mix_perf_angle_start": self.mix_perf_angle_start_var.get(),
            "mix_perf_angle_stop": self.mix_perf_angle_stop_var.get(),
            "mix_perf_angle_step": self.mix_perf_angle_step_var.get(),
            "mix_perf_wave_pol": self.mix_perf_wave_pol_var.get(),
            "mix_thickness": self.mix_thickness_var.get(),
            "mix_freq_mode": self.mix_freq_mode_var.get(),
            "mix_freq_list": self.mix_freq_list_var.get(),
            "mix_target_start": self.mix_target_start_var.get(),
            "mix_target_stop": self.mix_target_stop_var.get(),
            "mix_target_step": self.mix_target_step_var.get(),
            "mix_max_evals": self.mix_max_evals_var.get(),
            "mix_top_n": self.mix_top_n_var.get(),
            "mix_seed": self.mix_seed_var.get(),
            "mix_score_mode": self.mix_score_mode_var.get(),
            "mix_unc_t_pct": self.mix_unc_t_pct_var.get(),
            "mix_unc_eps_pct": self.mix_unc_eps_pct_var.get(),
            "mix_unc_mu_pct": self.mix_unc_mu_pct_var.get(),
            "mix_refine": self.mix_refine_var.get(),
            "mix_uncertainty": self.mix_uncertainty_var.get(),
            "dark_mode": self.dark_mode_var.get(),
        }
        return {
            "layers": [layer_config_to_dict(layer) for layer in self.layers],
            "controls": controls,
            "mixes": {"components": [dict(c) for c in self.mix_components]},
        }

    def _apply_project_state(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Project state must be an object.")
        layers_data = state.get("layers", [])
        controls = state.get("controls", {})
        if not isinstance(layers_data, list):
            raise ValueError("Project layers must be a list.")
        if not isinstance(controls, dict):
            raise ValueError("Project controls must be an object.")

        loaded_layers: list[LayerConfig] = []
        for idx, raw_layer in enumerate(layers_data, start=1):
            if not isinstance(raw_layer, dict):
                raise ValueError(f"Layer {idx}: expected an object.")
            loaded_layers.append(layer_config_from_dict(raw_layer, idx))

        str_vars: dict[str, StringVar] = {
            "f_start": self.f_start_var,
            "f_stop": self.f_stop_var,
            "f_step": self.f_step_var,
            "backing": self.backing_var,
            "output": self.output_var,
            "ibc_batch_start": self.ibc_batch_start_var,
            "ibc_batch_stop": self.ibc_batch_stop_var,
            "ibc_batch_step": self.ibc_batch_step_var,
            "ibc_batch_unit": self.ibc_batch_unit_var,
            "ibc_batch_output_dir": self.ibc_batch_output_dir_var,
            "ibc_batch_prefix": self.ibc_batch_prefix_var,
            "unc_t_pct": self.unc_t_pct_var,
            "unc_eps_pct": self.unc_eps_pct_var,
            "unc_mu_pct": self.unc_mu_pct_var,
            "angle_f_start": self.angle_f_start_var,
            "angle_f_stop": self.angle_f_stop_var,
            "angle_f_step": self.angle_f_step_var,
            "angle_start": self.angle_start_var,
            "angle_stop": self.angle_stop_var,
            "angle_step": self.angle_step_var,
            "wave_pol": self.wave_pol_var,
            "angle_output": self.angle_output_var,
            "angle_unc_t_pct": self.angle_unc_t_pct_var,
            "angle_unc_eps_pct": self.angle_unc_eps_pct_var,
            "angle_unc_mu_pct": self.angle_unc_mu_pct_var,
            "thk_f_start": self.thk_f_start_var,
            "thk_f_stop": self.thk_f_stop_var,
            "thk_f_step": self.thk_f_step_var,
            "thk_start": self.thk_start_var,
            "thk_stop": self.thk_stop_var,
            "thk_step": self.thk_step_var,
            "thk_angle": self.thk_angle_var,
            "thk_wave_pol": self.thk_wave_pol_var,
            "thk_output": self.thk_output_var,
            "thk_unc_t_pct": self.thk_unc_t_pct_var,
            "thk_unc_eps_pct": self.thk_unc_eps_pct_var,
            "thk_unc_mu_pct": self.thk_unc_mu_pct_var,
            "heatmap_metric": self.heatmap_metric_var,
            "uncertainty_view": self.uncertainty_view_var,
            "cbar_min": self.cbar_min_var,
            "cbar_max": self.cbar_max_var,
            "slice_angle": self.slice_angle_var,
            "slice_freq": self.slice_freq_var,
            "inv_freq_mode": self.inv_freq_mode_var,
            "inv_freq_list": self.inv_freq_list_var,
            "inv_target_start": self.inv_target_start_var,
            "inv_target_stop": self.inv_target_stop_var,
            "inv_target_step": self.inv_target_step_var,
            "inv_angle_start": self.inv_angle_start_var,
            "inv_angle_stop": self.inv_angle_stop_var,
            "inv_angle_step": self.inv_angle_step_var,
            "inv_wave_pol": self.inv_wave_pol_var,
            "inv_max_evals": self.inv_max_evals_var,
            "inv_top_n": self.inv_top_n_var,
            "inv_percentile": self.inv_percentile_var,
            "inv_unc_t_pct": self.inv_unc_t_pct_var,
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var,
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var,
            "inv_score_mode": self.inv_score_mode_var,
            "inv_seed": self.inv_seed_var,
            "mix_rule": self.mix_rule_var,
            "mix_objective": self.mix_objective_var,
            "mix_prop_source": self.mix_prop_source_var,
            "mix_prop_eps_re": self.mix_prop_eps_re_var,
            "mix_prop_eps_im": self.mix_prop_eps_im_var,
            "mix_prop_mu_re": self.mix_prop_mu_re_var,
            "mix_prop_mu_im": self.mix_prop_mu_im_var,
            "mix_prop_file": self.mix_prop_file_var,
            "mix_prop_weps": self.mix_prop_weps_var,
            "mix_prop_wmu": self.mix_prop_wmu_var,
            "mix_perf_metric": self.mix_perf_metric_var,
            "mix_perf_target": self.mix_perf_target_var,
            "mix_perf_angle_start": self.mix_perf_angle_start_var,
            "mix_perf_angle_stop": self.mix_perf_angle_stop_var,
            "mix_perf_angle_step": self.mix_perf_angle_step_var,
            "mix_perf_wave_pol": self.mix_perf_wave_pol_var,
            "mix_thickness": self.mix_thickness_var,
            "mix_freq_mode": self.mix_freq_mode_var,
            "mix_freq_list": self.mix_freq_list_var,
            "mix_target_start": self.mix_target_start_var,
            "mix_target_stop": self.mix_target_stop_var,
            "mix_target_step": self.mix_target_step_var,
            "mix_max_evals": self.mix_max_evals_var,
            "mix_top_n": self.mix_top_n_var,
            "mix_seed": self.mix_seed_var,
            "mix_score_mode": self.mix_score_mode_var,
            "mix_unc_t_pct": self.mix_unc_t_pct_var,
            "mix_unc_eps_pct": self.mix_unc_eps_pct_var,
            "mix_unc_mu_pct": self.mix_unc_mu_pct_var,
        }
        bool_vars: dict[str, BooleanVar] = {
            "uncertainty": self.uncertainty_var,
            "angle_uncertainty": self.angle_uncertainty_var,
            "thk_uncertainty": self.thk_uncertainty_var,
            "cbar_auto": self.cbar_auto_var,
            "inv_uncertainty": self.inv_uncertainty_var,
            "inv_refine": self.inv_refine_var,
            "mix_refine": self.mix_refine_var,
            "mix_uncertainty": self.mix_uncertainty_var,
            "dark_mode": self.dark_mode_var,
        }

        restored_controls = {}
        for key, var in str_vars.items():
            if key in controls:
                value = str(controls[key])
                if key in {
                    "wave_pol",
                    "thk_wave_pol",
                    "inv_wave_pol",
                    "mix_perf_wave_pol",
                }:
                    # Migrate legacy HH/VV project values to unambiguous
                    # plane-wave TE/TM labels.
                    value = normalize_wave_polarization(value).upper()
                elif key == "backing":
                    value = normalize_backing(value)
                elif key == "heatmap_metric":
                    legacy_metrics = {
                        "Metal backed loss (dB)": "PEC-backed reflection |Γ| (dB)",
                        "Metal phase (deg)": "PEC reflection phase (deg)",
                        "Metal absorption (dB)": "PEC absorbed power (dB)",
                        "Air backed loss (dB)": "Air-backed reflection |Γ| (dB)",
                        "Air phase (deg)": "Air reflection phase (deg)",
                        "Air absorption (dB)": "Air absorbed power (dB)",
                        "Insertion loss (dB)": "Transmission |S21| (dB)",
                        "Insertion phase (deg)": "Transmission phase (deg)",
                    }
                    value = legacy_metrics.get(value, value)
                elif key == "inv_score_mode":
                    if value.startswith("Worst-case"):
                        value = INVERSE_SCORE_MODE_OPTIONS[0]
                    elif value.startswith("Average mean"):
                        value = INVERSE_SCORE_MODE_OPTIONS[1]
                elif key == "mix_objective":
                    lowered = value.strip().lower()
                    if "performance" in lowered:
                        value = MIX_OBJECTIVE_PERFORMANCE
                    elif lowered.startswith("match") or lowered.startswith("find"):
                        value = MIX_OBJECTIVE_PROPERTY
                    elif lowered.startswith("predict"):
                        value = MIX_OBJECTIVE_FORWARD
                restored_controls[key] = value
        restored_bools = {key: self._coerce_bool(controls[key])
                          for key in bool_vars if key in controls}

        mixes = state.get("mixes", {})
        mix_components: list[dict] = []
        if not isinstance(mixes, dict):
            raise ValueError("Project mixes must be an object.")
        raw_components = mixes.get("components", [])
        if not isinstance(raw_components, list):
            raise ValueError("Project mix components must be a list.")
        for index, raw in enumerate(raw_components, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"Mix component {index}: expected an object.")
            try:
                migrated = dict(raw)
                if migrated.get("units") != "volume_percent":
                    migrated.update(min=0.0, max=100.0, units="volume_percent")
                mix_components.append(self._coerce_mix_component(migrated))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Mix component {index}: {exc}") from exc

        # Finish parsing the complete project before touching controls, layers,
        # results or dirty-state signals. An invalid load leaves the study intact.
        for key, value in restored_controls.items():
            str_vars[key].set(value)
        for key, value in restored_bools.items():
            bool_vars[key].set(value)

        self.layers = loaded_layers
        self.mix_components = mix_components
        self._refresh_layers()
        self.angle_compare_both.setChecked(self._coerce_bool(controls.get('angle_compare_both', True)))
        # The Thickness layer choices are rebuilt from the restored stack, so
        # the saved selection is re-applied after the combo is repopulated.
        if "thk_layer" in controls:
            saved_layer = str(controls["thk_layer"])
            if saved_layer in [label for _idx, label in self._thickness_layer_choices()]:
                self.thk_layer_var.set(saved_layer)
        if "ibc_batch_layer" in controls:
            saved_layer = str(controls["ibc_batch_layer"])
            if saved_layer in [label for _idx, label in self._thickness_layer_choices()]:
                self.ibc_batch_layer_var.set(saved_layer)
        self._refresh_ibc_batch_preview()
        self._sync_uncertainty_state()
        self._sync_angle_uncertainty_state()
        self._sync_thickness_uncertainty_state()
        self._sync_inverse_freq_mode_state()
        self._sync_inverse_uncertainty_state()
        self._sync_mix_freq_mode_state()
        self._sync_mix_uncertainty_state()
        self._sync_mix_objective_state()
        self._sync_cbar_state()

        self._apply_theme()
        self.last_heatmap_results = None
        self.last_heatmap_uncertainty_min = None
        self.last_heatmap_uncertainty_max = None
        self.last_thickness_results = None
        self.last_thickness_uncertainty_min = None
        self.last_thickness_uncertainty_max = None
        self.inverse_plot_freqs = []
        self.inverse_plot_samples = []
        self.selected_x_idx = None
        self.selected_freq_idx = None
        self.inverse_candidates = []
        self._inverse_checkpoint = None
        self.inverse_recovery_path.clear()
        self._inverse_result_identity = None
        self._inverse_progress = None
        self.inv_extend_btn.setEnabled(False)
        self.inv_setup_status.setText('Review the allowed values and combination count, then analyze the setup.')
        self.inverse_result_metadata = {}
        self._clear_analysis_results()
        self._inverse_summary = ''
        self._inverse_page_index = 0
        self.inv_result_status.setText('Analyze all combinations from Setup to compare candidates here.')
        self._sync_mode_chrome()
        self.mix_candidates = []
        self.mix_plot_data = []
        self.mix_preview = None
        self._refresh_inverse_results_list()
        self._refresh_mix_components_list()
        self._refresh_mix_results_list()
        self._update_plot()
