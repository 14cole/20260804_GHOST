#!/usr/bin/env python3
"""Convert the NIST BaM/PDMS publication data to FREDDY material CSVs.

The NIST tables publish dielectric and magnetic loss as positive magnitudes.
FREDDY uses exp(+j*omega*t), so this converter negates those loss columns.
Measured rows with a negative published loss magnitude are measurement noise
that would represent gain after conversion; they are reported and omitted,
never clipped or silently sign-flipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


MATERIAL_HEADER = "frequency_hz,eps_real,eps_imag,mu_real,mu_imag"
DATASET_DOI = "https://doi.org/10.18434/mds2-2911"
NIST_LICENSE = "https://www.nist.gov/open/license"
MIN_FREQUENCY_HZ = 100_000_000.0
VOLUME_FRACTION_30WT = 0.0726
VOLUME_FRACTION_60WT = 0.215
BAM_EPSILON = complex(16.65, 0.0)

SOURCE_SHA256 = {
    "README.txt": "19467fd7cb303ecf1f5cb702be162705c391418305fba18286b27ff8fac1a319",
    "Figure 7a.csv": "b1a98f1fcb2ddb0b91798dfc303c3cb3812d6c28a64ca02254e0ba8c5c56aa8d",
    "Figure 7b.csv": "a164abf2ce5fa89d5549d365585fd498175f3a6784566e6e196b6029b68c55fd",
    "Figure 8a.csv": "4287e2a46b510b8778009c6cc1f76700424f9bb0c9adeaaffbd85316ce1bcfb5",
    "Figure 8b.csv": "9058163e9f9aa5e5a87948e127944f96a0fa348d1b9e1752d061970f1b93ca63",
}


@dataclass(frozen=True)
class MaterialRow:
    frequency_hz: float
    epsilon: complex
    mu: complex


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(source_dir: Path) -> None:
    for name, expected in SOURCE_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing NIST source file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )


def read_figure(path: Path, expected_columns: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(3):
            next(reader)
        for line_number, raw in enumerate(reader, start=4):
            if not raw or not any(cell.strip() for cell in raw):
                continue
            if len(raw) != expected_columns:
                raise ValueError(
                    f"{path}: line {line_number} has {len(raw)} columns; "
                    f"expected {expected_columns}."
                )
            try:
                values = [float(cell) for cell in raw]
            except ValueError as exc:
                raise ValueError(
                    f"{path}: line {line_number} contains a non-numeric value."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: line {line_number} is non-finite.")
            rows.append(values)
    return rows


def _same_frequency(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)


def align_source_tables(
    source_dir: Path,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    eps_real = [
        row
        for row in read_figure(source_dir / "Figure 7a.csv", 11)
        if row[0] >= MIN_FREQUENCY_HZ
    ]
    eps_loss = [
        row
        for row in read_figure(source_dir / "Figure 7b.csv", 11)
        if row[0] >= MIN_FREQUENCY_HZ
    ]
    mu_real = read_figure(source_dir / "Figure 8a.csv", 8)
    mu_loss = read_figure(source_dir / "Figure 8b.csv", 8)
    for table_name, table in (
        ("Figure 7a.csv", eps_real),
        ("Figure 7b.csv", eps_loss),
        ("Figure 8a.csv", mu_real),
        ("Figure 8b.csv", mu_loss),
    ):
        for index, row in enumerate(table, start=1):
            if not _same_frequency(row[0], row[2]):
                raise ValueError(
                    f"{table_name}: duplicated frequency columns disagree at "
                    f"common-grid row {index}: {row[0]} versus {row[2]}"
                )
    lengths = {len(eps_real), len(eps_loss), len(mu_real), len(mu_loss)}
    if len(lengths) != 1:
        raise ValueError(f"NIST common-grid row counts differ: {sorted(lengths)}")
    for index, rows in enumerate(zip(eps_real, eps_loss, mu_real, mu_loss), start=1):
        frequencies = [row[0] for row in rows]
        if not all(_same_frequency(frequencies[0], value) for value in frequencies[1:]):
            raise ValueError(
                f"NIST source tables disagree at common-grid row {index}: "
                f"{frequencies}"
            )
    return eps_real, eps_loss, mu_real, mu_loss


def inverse_maxwell_garnett(
    effective: complex, host: complex, inclusion_fraction: float
) -> complex:
    """Recover a spherical inclusion property from a two-phase MG result."""
    if not 0.0 < inclusion_fraction < 1.0:
        raise ValueError("Inclusion fraction must be strictly between 0 and 1.")
    y = effective / host
    if abs(y + 2.0) <= 1e-14:
        raise ValueError("Singular effective/host ratio in MG inversion.")
    polarizability = (y - 1.0) / (y + 2.0) / inclusion_fraction
    if abs(1.0 - polarizability) <= 1e-14:
        raise ValueError("Singular inclusion polarizability in MG inversion.")
    return host * (1.0 + 2.0 * polarizability) / (1.0 - polarizability)


def _material_from_columns(
    eps_real_rows: list[list[float]],
    eps_loss_rows: list[list[float]],
    mu_real_rows: list[list[float]],
    mu_loss_rows: list[list[float]],
    eps_column: int,
    mu_column: int,
) -> list[MaterialRow]:
    return [
        MaterialRow(
            frequency_hz=e_real[0],
            epsilon=complex(e_real[eps_column], -e_loss[eps_column]),
            mu=complex(m_real[mu_column], -m_loss[mu_column]),
        )
        for e_real, e_loss, m_real, m_loss in zip(
            eps_real_rows, eps_loss_rows, mu_real_rows, mu_loss_rows
        )
    ]


def _is_passive_ordinary(row: MaterialRow) -> bool:
    return (
        row.epsilon.real > 0.0
        and row.mu.real > 0.0
        and row.epsilon.imag <= 0.0
        and row.mu.imag <= 0.0
        and all(
            math.isfinite(value)
            for value in (
                row.frequency_hz,
                row.epsilon.real,
                row.epsilon.imag,
                row.mu.real,
                row.mu.imag,
            )
        )
    )


def write_material(path: Path, rows: list[MaterialRow]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty material file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MATERIAL_HEADER.split(","))
        previous = 0.0
        for row in rows:
            if row.frequency_hz <= previous:
                raise ValueError(f"Frequencies are not strictly increasing in {path}.")
            if not _is_passive_ordinary(row):
                raise ValueError(f"Non-passive/invalid row passed to writer: {row}")
            writer.writerow(
                [
                    format(row.frequency_hz, ".15g"),
                    format(row.epsilon.real, ".15g"),
                    format(row.epsilon.imag, ".15g"),
                    format(row.mu.real, ".15g"),
                    format(row.mu.imag, ".15g"),
                ]
            )
            previous = row.frequency_hz


def convert(source_dir: Path, output_dir: Path) -> dict[str, object]:
    verify_sources(source_dir)
    eps_real, eps_loss, mu_real, mu_loss = align_source_tables(source_dir)

    pdms_fit = [
        MaterialRow(
            row_real[0],
            complex(row_real[8], -row_loss[8]),
            complex(1.0, 0.0),
        )
        for row_real, row_loss in zip(eps_real, eps_loss)
    ]
    fit_30 = _material_from_columns(
        eps_real, eps_loss, mu_real, mu_loss, eps_column=9, mu_column=6
    )
    fit_60 = _material_from_columns(
        eps_real, eps_loss, mu_real, mu_loss, eps_column=10, mu_column=7
    )
    particle_fit = [
        MaterialRow(
            target.frequency_hz,
            BAM_EPSILON,
            inverse_maxwell_garnett(
                target.mu, complex(1.0, 0.0), VOLUME_FRACTION_30WT
            ),
        )
        for target in fit_30
    ]

    measured_specs = {
        "bam_pdms_30wt_sample_1_passive.csv": (4, 1),
        "bam_pdms_30wt_sample_2_passive.csv": (5, 3),
        "bam_pdms_30wt_sample_3_passive.csv": (6, 4),
        "bam_pdms_60wt_measured_passive.csv": (7, 5),
    }
    outputs: dict[str, list[MaterialRow]] = {
        "pdms_fit.csv": pdms_fit,
        "bam_particle_fit.csv": particle_fit,
        "bam_pdms_30wt_fit.csv": fit_30,
        "bam_pdms_60wt_fit.csv": fit_60,
    }
    dropped: dict[str, int] = {}
    for name, (eps_column, mu_column) in measured_specs.items():
        raw_rows = _material_from_columns(
            eps_real,
            eps_loss,
            mu_real,
            mu_loss,
            eps_column=eps_column,
            mu_column=mu_column,
        )
        passive_rows = [row for row in raw_rows if _is_passive_ordinary(row)]
        outputs[name] = passive_rows
        dropped[name] = len(raw_rows) - len(passive_rows)

    for name, rows in outputs.items():
        write_material(output_dir / name, rows)

    manifest: dict[str, object] = {
        "dataset": "Broadband Electromagnetic Properties of Engineered Flexible Absorber Materials",
        "doi": DATASET_DOI,
        "license": NIST_LICENSE,
        "source_sha256": SOURCE_SHA256,
        "conversion": {
            "frequency_unit": "Hz (unchanged)",
            "source_loss_columns": "positive loss magnitudes",
            "freddy_time_convention": "exp(+j*omega*t)",
            "freddy_loss_conversion": "eps_imag=-source_eps_loss; mu_imag=-source_mu_loss",
            "minimum_frequency_hz": MIN_FREQUENCY_HZ,
            "measured_nonpassive_policy": "omit and count; never clamp or change sign",
            "bam_particle_epsilon": [BAM_EPSILON.real, BAM_EPSILON.imag],
            "bam_particle_mu": "inverted from the published 30 wt% MG fit",
            "volume_fraction_30wt": VOLUME_FRACTION_30WT,
            "volume_fraction_60wt": VOLUME_FRACTION_60WT,
        },
        "outputs": {
            name: {"rows": len(rows), "dropped_nonpassive_rows": dropped.get(name, 0)}
            for name, rows in outputs.items()
        },
    }
    manifest_path = output_dir.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    validation_root = Path(__file__).resolve().parents[1] / "materials" / "validation" / "nist_bam_pdms"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=validation_root / "source",
        help="Directory containing the five unmodified NIST source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=validation_root / "freddy",
        help="Directory for generated FREDDY five-column material CSVs.",
    )
    args = parser.parse_args()
    manifest = convert(args.source_dir, args.output_dir)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
