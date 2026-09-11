from pathlib import Path
import json,statistics
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(name):return json.loads((HERE/name).read_text())
def link(label,path):return '[{}]({})'.format(label,Path(path).as_posix())
v=read('validated-results.json');p=v['production']
dense=[p['airfoil-f2.0-baseline-dense-r{}.json'.format(i)] for i in (1,2)]
old=p['airfoil-f2.0-baseline-hierarchical-r1.json']
new=[p['airfoil-f2.0-updated-hierarchical-r{}.json'.format(i)] for i in (4,5)]
med=lambda rows,key:statistics.median(r[key] for r in rows)
newtime=med(new,'seconds');denstime=med(dense,'seconds')
newrss=med(new,'peak_mib');densrss=med(dense,'peak_mib')
stage=lambda rows,name:statistics.median(r['stage_seconds'][name] for r in rows)
spooled=[read('paired-f2.0-tile512-tol1e-06-spool.json'),read('paired-f2.0-tile512-tol1e-06-spool-r2.json')]
def core(r):return r['preparation_seconds']+r['paired_assembly_seconds']+sum(x['load_seconds']+x['factor_seconds']+x['rhs_solve_projection_seconds'] for x in r['channels'].values())
prototype_time=statistics.median(core(r) for r in spooled)
prototype_rss=[r['sampled_core_rss_bytes']/2**20 for r in spooled]
native=v['native'];ratios=list(native['payload_ratios'].values())
material_error=max(e for channels in v['materials']['hierarchical'].values() for e in channels.values())
base1=p['airfoil-f1.0-baseline-hierarchical-r1.json'];new1=p['airfoil-f1.0-updated-hierarchical-r1.json']
text=f'''# Final GHOST 2D time and RAM investigation

September 10, 2026. This report separates this pass's changes from the earlier matrix-lifetime and assembly work. It covers practical assembly, storage, factorization, sweep and fallback candidates; it does not claim that every possible numerical method has been exhausted.

## Result and release scope

The regular Backend now builds a smaller **hierarchical inverse with checked refinement and a tighter rebuild fallback**. On the supplied airfoil at 2 GHz, both polarizations and all 361 azimuths, measurements with the updated policy were **{newtime:.2f} s and about {newrss:.0f} MiB**, versus **{old['seconds']:.2f} s / {old['peak_mib']:.0f} MiB** for the previous hierarchical method. This is a measured {100*(1-newtime/old['seconds']):.1f}% time reduction and {100*(1-newrss/old['peak_mib']):.1f}% peak-RAM reduction within that method. Factor payload fell by approximately one third. The end-to-end timing gain is small; these few runs do not establish a precise universal speedup.

The new **paired, disk-spooled compressed assembly prototype** finished in **{prototype_time:.2f} s** with **{min(prototype_rss):.0f}–{max(prototype_rss):.0f} MiB** sampled peak RAM. The earlier sequential compressed prototype required about **88.86 s** for the two polarizations and peaked at **415 MiB**. That is approximately **{100*(1-prototype_time/88.86):.0f}% less time** and **{100*(1-statistics.median(prototype_rss)/415):.0f}% less peak RAM**. The newer timing also includes mesh preparation and field projection, which the earlier core timing excluded.

Dense LU remains the default. The Backend improvement applies when `GHOST_CPU_FACTORIZATION=hierarchical` is selected, or when `auto` selects a hierarchy. `auto` may allocate full dense LU if the hierarchy is rejected; strict `hierarchical` reports a failure instead. Restart GHOST to load the changed Backend. The much smaller compressed-assembly path remains in this experiment folder, outside the GUI and production import path.

## Production changes

1. **Coarse inverse, accurate equations.** Off-diagonal inverse construction starts at `1e-6`, reducing typical airfoil maximum rank from 100 to 63. The original dense A is still retained and used for every normal, transpose and adjoint refinement, condition estimate and physical-angle check. Operator assembly and material equations are unchanged.
2. **Additional accuracy margin.** The smaller inverse targets `3e-15` normwise backward error. A first trial at the old `3e-14` target passed all material fields, but magnetic TE was close to the `1e-10` field-comparison threshold at `8.20e-11`. The final stricter target reduced the largest material field difference to **{material_error:.3g}**. A rejected construction or stalled solve releases the old tree and retries once at the previous `2e-10` block tolerance and `3e-14` refinement limit. Rejection after that retains existing strict/automatic behavior. Final physical-angle acceptance still requires the existing `1e-12` backward-error gate.
3. **Reliable lifetimes and diagnostics.** Exception tracebacks and the rejected tree are released before replacement allocation. Rebuild counts, selected tolerances, refinements and rejection reasons are reported; outer factorization counts include a rebuild. Invalid RHS values/shapes reject before attempting a rebuild. A failed refinement no longer performs a last inverse correction that would be discarded without being checked. Storage admission remains conservative and does not assume a particular achieved rank.

| 2 GHz, full public solve | Elapsed | Sampled peak RSS | Factor construction |
|---|---:|---:|---:|
| Dense baseline, median of two | {denstime:.2f} s | {densrss:.0f} MiB | {stage(dense,'factorization'):.2f} s |
| Previous hierarchical, one run | {old['seconds']:.2f} s | {old['peak_mib']:.0f} MiB | {old['stage_seconds']['factorization']:.2f} s |
| Final hierarchical, median of two | {newtime:.2f} s | {newrss:.0f} MiB | {stage(new,'factorization'):.2f} s |

Final hierarchical versus dense is {100*(1-newtime/denstime):.1f}% faster and {100*(1-newrss/densrss):.1f}% lower in sampled peak RAM in these runs. Most of that RAM difference comes from the previously available hierarchical method; it is **not all a new saving from this pass**. The new factor's TE/TM payloads are {new[0]['factors']['VV'][0]['factor_bytes']/1e6:.2f}/{new[0]['factors']['HH'][0]['factor_bytes']/1e6:.2f} MB, versus {old['factors']['VV'][0]['factor_bytes']/1e6:.2f}/{old['factors']['HH'][0]['factor_bytes']/1e6:.2f} MB previously. Extra refinement partly offsets faster construction.

The 1 GHz check took {base1['seconds']:.2f} s before and {new1['seconds']:.2f} s after, with sampled peaks {base1['peak_mib']:.0f} and {new1['peak_mib']:.0f} MiB. These are individual measurements. All public measurements include both channels, condition estimation, RHS compression, all 361 physical angles and field projection. They are base-mesh discrete solves, not base/fine mesh certificates. Timed stages nest and must not be added together as independent costs.

## Compressed assembly implementation

**Share expensive polarization assembly.** A paired regional oracle prepares TE and TM destinations for a spatial tile, combines destinations with the same wavenumber, and performs a shared quadrature traversal. Each polarization retains its own material coefficients, source/observer masks, jump terms and pruning-error bounds. Every final coefficient is requested once from geometry; repeated inverse queries subsequently read the stored compressed tiles. Empty single-layer destinations no longer request that kernel.

**Keep one accurate operator and an inverse-only preconditioner.** The operator uses full-checked tile QR at `1e-14`, retaining a dense local tile where compression is not profitable. The inverse is built at `1e-6` and retains only LU leaves, low-rank coupling factors and Woodbury solve data. It drops raw leaf operators and U factors after their solved counterparts exist. It skips the second operator-error bookkeeping scan because it cannot supply the accurate operator action. ACA still verifies its own candidate block error. Accurate matvecs and the incoming coefficient-error bound determine acceptance.

**Refine an incident basis and check every requested angle.** Global row-scaled QR needs 44 basis columns for the 2 GHz airfoil's 361 illuminations, versus the production incremental basis's 51. Bounded corrections use the accurate tiled operator. A capped, preconditioned GMRES fallback repairs stalled columns; every original physical column is then checked with pruning/tile error included. GMRES was covered by an intentionally difficult test; the accepted 1/2 GHz airfoil sweeps did not require it. The prototype accepts at most 512 input columns per call and does not yet implement the public solver's arbitrary-grid streaming lifecycle.

**Spool the second polarization.** Compressed TM tiles are written to an owned temporary file while TE is assembled/solved. The TE operator/inverse and RHS are released before TM tiles are loaded. About 100 MiB of temporary compressed TM data at 2 GHz is removed after loading. Use before load, truncated files, cancellation and budget failures are checked. Disk spooling is optional. The unspooled paired run took about 55.96 s at 485 MiB; spooling reduced that peak by about 22%, with little elapsed-time difference on this machine's filesystem. Other disks can have different costs.

The 512 MiB allowance covers retained numeric operator/preconditioner payload, including stored TM payload while spooled. It is **not a process-RAM cap**: geometry, Python objects, QR/RHS work and construction temporaries are additional. Coefficient queries are capped at 16 MiB, tile width at 1024 and dense inverse leaves at 512. There is no full dense fallback in the prototype. Qualification writes currents to local `.npy` files; those writes and independent dense-reference qualification are excluded from core elapsed time. RSS sampling stops before loading dense qualification matrices.

At 1 GHz the paired spooled path took {v['paired']['paired-f1.0-tile512-tol1e-06-spool-r1.json']['core_seconds']:.2f} s at {v['paired']['paired-f1.0-tile512-tol1e-06-spool-r1.json']['core_peak_mib']:.0f} MiB. Across 1/2 GHz it passed every requested angle, independent dense-matrix residuals, coefficient-error bounds and complex-field comparisons. The largest independent field difference in the final paired runs was {max(x['field_error'] for r in v['paired'].values() for x in r['channels'].values()):.3g}, normalized to the reference channel peak.

## Material coverage and limits

Both public factors passed **14 families**: PEC, IBC, PEC/IBC, lossless, lossy and magnetic dielectric, coated, layered and mixed regions, sheet, mixed sheet/PEC, thin dielectric, magnetic thin layer and transparent thin layer. Comparisons used the full complex fields at all 361 angles.

The experimental preconditioner passed **all 26 nontransparent native polarization matrices**, versus 22 with the earlier strict compressed inverse. Mixed-sheet TE, thin-layer TE and both magnetic thin-layer channels now pass. Two transparent cases require no factor. Operator-plus-inverse numeric payload ranges from **{100*min(ratios):.0f}% to {100*max(ratios):.0f}% of dense A+LU**, depending on material/formulation. Largest independent backward error: {native['max_backward']:.3g}; largest field difference: {native['max_field_error']:.3g}. This last magnetic result is close to the experimental field-comparison threshold and remains a qualification concern for broader promotion.

Those native tests use captured dense matrices as coefficient oracles: they establish algebra/storage suitability, **not** a native geometry assembly speedup. The paired geometry implementation covers regional S/K equations. Production-equivalent W, sheet and thin-layer coefficient oracles, adjoint operator-error evidence, condition estimation and public mesh-certification integration remain necessary before promoting the complete compressed backend.

## Other methods reviewed

| Method | Finding and decision |
|---|---|
| Smaller RHS batches | At 256/128/64 angles: 41.95/42.42/43.11 s and 1244/1195/1181 MiB. Basis solves increased from 51/51 to 60/60 and 166/138 for TE/TM. Keep 256 by default; existing configuration permits a deliberate RAM/time tradeoff. |
| BLAS AXPY in kernel accumulation | Some large-buffer microbenchmarks were several times faster, but the complete solve was 42.37 s versus 41.95 s. Not promoted on the strength of a microbenchmark. |
| Preconditioner tolerance | 2 GHz TE captured-matrix trials: `1e-4` factor+solve 6.28 s, `1e-6` 5.23 s, `1e-8` 7.11 s; 7/3/1 corrections. Choose `1e-6` with accurate-operator checks; very loose construction is not automatically faster. |
| Unpreconditioned/local-block GMRES | Earlier airfoil investigation found nonconvergence or many iterations and physical-gate failures. Reuse a stronger preconditioner and an incident basis; do not launch 361 independent iterative solves. |
| Mixed-precision LU | Previous trials saved factor payload but had inconsistent total times after correction. Retain the existing checked option and experimental CPU's double-precision contract. |
| Separate dense S/K/W/mass matrices, duplicate LU, cached RHS | Earlier work already assembled into owned final systems, used sparse/local mass operations, reused one factor per channel/mesh, bounded sweep state and released source operators. No reason to restore those allocations. |
| Reciprocal/symmetric storage | Existing compatible S/K reuse remains useful. The full multi-material lossy system is generally not Hermitian; treating it as one would change the solve. Paired assembly shares valid kernel work without that assumption. |
| Less quadrature / fused W assembly | Exact near, endpoint and material terms constrain this. The existing W path is preserved; native W tile integration needs separate accuracy evidence. |
| Mesh or basis changes | The high-index material controls most airfoil interfaces. Earlier interface-local meshing reduced unknowns only 3.37% at 10 GHz. Changing panel density, pulse/linear equations or replacing bulk material by impedance requires a separate accuracy study. |
| Cache bookkeeping | The measured kernel tables were around 0.64 MiB. Removing small coefficient duplication cannot address multi-gigabyte A/LU storage. |
| More parallel angle workers | Duplicates assembly and factor memory. Preserve per-frequency/channel reuse; more assembly/BLAS workers need workload and RAM measurements. |
| Directional hierarchical/FMM matvec | Remains the larger architectural opportunity for high-frequency rank growth. It needs material-specific accurate kernels, near blocks, reusable preconditioning, adjoints and certification. No FMM or full 10 GHz performance result is claimed here. |

## Why 12 inches and one angle can still need substantial RAM

The earlier audited 10 GHz mesh has **28,406 unknowns**. Its requested `N=-20` density and maximum refractive-index magnitude about **22.33** produce a controlling material wavelength near 1.34 mm. One complex128 dense A alone needs **12.02 GiB**. LU and other work are additional. Azimuth changes the excitation, not this matrix, so one angle cannot remove the main quadratic allocation.

The earlier approximately 87 GiB prediction was reduced by previous allocation fixes to about **25.51 GiB** for the dense base mesh and **56.41 GiB** for its certification fine mesh. Those remain allocation plans, not full-size measurements from this pass. The final inverse can use less than its reserved allowance, but planner limits have not been reduced based on these smaller examples. **No complete 10 GHz solve or new mesh-convergence certificate was run.** Neither the 2 GHz prototype RSS nor the pulse-basis solver's reported success establishes equal accuracy at 10 GHz.

## Verification and reproduction

- 143 production regression tests passed after the final refinement change; the compression regression module was rerun after removing the unused rejection correction. Timings precede that failure-only change; successful measured paths are unchanged. Five additional experimental test methods cover paired material equations, ownership, caps, GMRES repair, invalid inputs, cancellation, spool loading/truncation and cleanup.
- Independent saved-field validation covers both public factors, 14 material families and full airfoil sweeps; all reported comparisons require channel-peak-relative complex error below `1e-10` and the normal solver gates. This is not relative error in dB at a radiation null.
- Benchmarks ran sequentially in fresh processes on the same Windows machine, Python 3.12.14 / NumPy 2.5.2 / SciPy 1.18.1, two BLAS threads and four assembly threads. The production RSS column uses the built-in sampler; supplemental 20 ms samples are recorded separately and sometimes capture higher short peaks. Prototype sampling is every 50 ms. These are measurements, not hard worst-case memory bounds.
- Changed Backend modules parse with Python 3.6 grammar; no new execution on an actual Python 3.6 interpreter is claimed. The experimental GMRES driver uses the installed modern SciPy API.
- Existing user changes were preserved. This pass changes `hierarchical_factor.py`, `dense_factor.py`, their regression tests and execution documentation; new prototype code stays here.

Evidence: {link('validated results',HERE/'validated-results.json')}, {link('this pass’s Backend diff',HERE/'final.diff')}, {link('final source hashes',HERE/'final-source-hashes.json')}, {link('native matrix results',HERE/'native-results.json')}, {link('previous investigation',HERE.parent/'ghost_hardening_20260910/REPORT.md')}.

From the repository root, run `.venv/Scripts/python.exe experiments/ghost_final_20260910/benchmark.py --factor hierarchical --repeat 4` for the full 2 GHz public solve. Add `--materials --repeat 2` for the material sweep. Run `paired_probe.py --frequency 2 --spool --repeat 2` in this folder for the paired compressed path, or `test_final_methods.py -q` for its regression tests. `validate_results.py` checks saved runs and writes the source inventory; `build_report.py` regenerates this report. Baseline runs use the preserved local `baseline_backend` copy, and independent dense references come from `ghost_ram_time_20260910/systems`. These generated local dependencies must exist to reproduce the saved comparisons.
'''
(HERE/'REPORT.md').write_text(text,encoding='utf-8')
print(HERE/'REPORT.md')
