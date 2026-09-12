# Audited source excerpts

These excerpts preserve the code inspected in the audit working tree. They are evidence, not application changes. See `source-snapshot.json` for full-file hashes.

<a id="rf-agreement"></a>
## RF agreement numerical threshold

Original file: `GRIM_Backend/plotting/modes/compare_mode.py`, lines 70–92.

```python
def _lin_concordance(left, right) -> float:
    """Return Lin's concordance correlation coefficient.

    Unlike Pearson correlation, CCC penalizes both gain and offset errors. A
    pair of equal constant traces is defined as perfect concordance; unequal
    constant traces have zero concordance.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    centered_left = left - mean_left
    centered_right = right - mean_right
    var_left = float(np.mean(centered_left ** 2))
    var_right = float(np.mean(centered_right ** 2))
    denominator = var_left + var_right + (mean_left - mean_right) ** 2
    scale = max(float(np.mean(left ** 2) + np.mean(right ** 2)), 1.0)
    if denominator <= np.finfo(float).eps * scale:
        return 1.0 if np.array_equal(left, right) else 0.0
    covariance = float(np.mean(centered_left * centered_right))
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))

```

<a id="output-replacement"></a>
## FREDDY output replacement policy

Original file: `tools/FREDDY/ibc/ui.py`, lines 960–994.

```python
    def _confirm_output_replacements(
        self, paths: list[Path], *, operation: str
    ) -> bool:
        """Preflight every output on the GUI thread before starting a worker."""

        unique: list[Path] = []
        seen: set[str] = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            key = os.path.normcase(os.path.abspath(path)).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)

        invalid = [path for path in unique if path.exists() and not path.is_file()]
        if invalid:
            messagebox.showerror(
                f"{operation} Output",
                "An output path is not a file:\n" + "\n".join(str(p) for p in invalid),
                parent=self,
            )
            return False
        existing = [path for path in unique if path.is_file()]
        if not existing:
            return True
        shown = "\n".join(str(path.resolve()) for path in existing[:10])
        if len(existing) > 10:
            shown += f"\n…and {len(existing) - 10} more"
        return messagebox.askyesno(
            f"Replace Existing {operation} Output?",
            f"{len(existing)} output file(s) already exist:\n\n{shown}\n\n"
            "Replace all listed files?",
            parent=self,
        )
```

<a id="impedance-worker"></a>
## FREDDY impedance worker

Original file: `tools/FREDDY/ibc/ui.py`, lines 5664–5731.

```python
    def _compute_impedance(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")

            layer_snapshot = self._snapshot_layers()
            output_path = Path(self.output_var.get().strip())
            _validate_csv_path(output_path)
            uncertainty = self._read_uncertainty_config(
                self.uncertainty_var,
                self.unc_t_pct_var,
                self.unc_eps_pct_var,
                self.unc_mu_pct_var,
            )
            uncertainty_has_bounds = uncertainty.enabled and any(
                value > 0
                for value in (
                    uncertainty.thickness_pct,
                    uncertainty.eps_pct,
                    uncertainty.mu_pct,
                )
            )
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            freqs = make_frequency_sweep(f_start, f_stop, f_step)
            backing = normalize_backing(self.backing_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        planned_outputs = [output_path]
        if uncertainty_has_bounds:
            planned_outputs.append(uncertainty_report_path(output_path))
        if not self._confirm_output_replacements(
            planned_outputs, operation="Impedance"
        ):
            return

        # Impedance is a broadside (normal-incidence) solve; polarization is unused.
        wave_pol = normalize_wave_polarization("TE")

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(layer_snapshot)
            capture = {}
            n, summary = self._compute_frequency_mode(
                output_path,
                loaded_layers,
                backing,
                uncertainty,
                freqs,
                wave_pol,
                capture=capture,
            )
            metrics = compute_angle_metrics_many(freqs, 0., loaded_layers, wave_pol)
            analysis = impedance_result(freqs, capture, metrics, backing,
                f'Impedance · {backing.upper()} backing · normal incidence · {freqs[0]:g}–{freqs[-1]:g} GHz · {tolerance_description(uncertainty)}')
            analysis.summary = summary + f'\nExport: {output_path.resolve()}'
            analysis.layers = stack_description(layer_snapshot)
            return {"count": n, "summary": summary, "analysis": analysis}

        def on_success(result: dict[str, object]) -> None:
            self._show_analysis_result('Impedance', result['analysis'])
            # Only the PEC-backed broadside result is a physically suitable
            # one-sided IBC for a closed Type 2 GHOST body. Air-backed and all
            # other analysis products intentionally never enter the handoff.
            if backing == "pec":
                try:
```

<a id="phase-mapping"></a>
## GRIM optional phase mapping

Original file: `GRIM_Backend/ui/table_import.py`, lines 52–90.

```python
                source, unit = candidates[0] if len(candidates) == 1 else (None, "As written")
                constant = {"elevation": "0", "azimuth": "0", "polarization": "VV"}.get(name, "")
                input_units = output_units = ("As written",)
                output_unit = "As written"
                if name == "frequency":
                    input_units = ("Choose unit…", "Hz", "kHz", "MHz", "GHz", "THz")
                    if unit not in input_units:
                        unit = "Choose unit…"
                    output_units, output_unit = ("GHz",), "GHz"
                elif name in ("azimuth", "elevation", "phase"):
                    input_units, output_units, output_unit = ("deg", "rad"), ("deg",), "deg"
                    unit = unit if unit in input_units else "deg"
                else:
                    unit = "As written"
                self.add_mapping(name, source=source, constant=constant, input_unit=unit,
                    output_unit=output_unit, checked=name != "phase" or source is not None,
                    locked=True, input_units=input_units, output_units=output_units)

        def submit(self):
            try:
                source, options, columns, names = self.snapshot()
                required = {"frequency", "azimuth", "elevation", "polarization", "magnitude"}
                if not required.issubset(column.name for column in columns):
                    raise ValueError("Frequency, azimuth, elevation, polarization and magnitude are required; "
                                     "use constants for missing columns.")
                representation = self.magnitude_format.currentText()
                if representation not in MAGNITUDE_FORMATS:
                    raise ValueError("Choose what the magnitude column represents.")
                provenance = {"options": asdict(options), "columns": [asdict(c) for c in columns],
                              "source_headers": list(names), "magnitude_format": representation}
                self.start_job(lambda: grid_from_mapped_rows(
                    core.converted_rows(source, options, columns, names), source, representation,
                    mapping=provenance))
            except Exception as exc:
                self.status.setText(str(exc))

        def conversion_finished(self, result):
            self.dataset = result
            self.accept()
```

<a id="unit-conversion"></a>
## Shared column unit conversion

Original file: `tools/FREDDY/ibc/table_conversion.py`, lines 156–181.

```python
def converted_rows(path, options, columns, expected_names=None):
    records = table_rows(path, options)
    try:
        _, names = next(records)
        if expected_names is not None and tuple(expected_names) != names:
            raise ValueError("The source columns changed. Refresh the preview and review the mapping.")
        validate_mapping(columns, len(names))
        for line_no, row in records:
            result = {}
            for column in columns:
                raw = column.constant if column.source is None else row[column.source]
                if column.input_unit == "As written":
                    value = raw
                else:
                    try:
                        value = number(raw) * (UNIT_FACTORS[column.input_unit][1] /
                                               UNIT_FACTORS[column.output_unit][1])
                    except (ValueError, OverflowError) as exc:
                        raise ValueError(f"Line {line_no}, {column.name}: invalid number {raw!r}.") from exc
                    if not math.isfinite(value):
                        raise ValueError(f"Line {line_no}, {column.name}: value must be finite.")
                result[column.name.strip()] = value
            yield line_no, result
    finally:
        records.close()

```
