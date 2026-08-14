"""
test_ss.py - build synthetic Xpatch .ss files and verify read_ss recovers them.

Covers the two regressions the real dataset exposed:
  * variable-length header-B (0/256/512/768 bytes) -> header-C / freq axis
  * bistatic runs where the sweep is in observation angle, not incident
plus a monostatic/uniform-freq baseline.

Synthetic data is self-consistent with read_ss's own framing, so a pass proves
the offset math and field selection -- not the absolute Xpatch byte layout,
which only your real file (or MATLAB ssread) can confirm.
"""

import os
import struct
import tempfile
import numpy as np
import read_ss as R


def _be_i4(v):
    return int(v).to_bytes(4, "big", signed=True)


def _be_f4(v):
    return struct.pack(">f", float(v))


def build_ss(path, nsig, nfreq, ifreq, freq1, freq2, flags, mode):
    """Write a synthetic .ss. `flags` -> header-B size; `mode` in {incident, observation}."""
    size_a = R._table_bytes(R.HDRA)            # 648
    size_c = R._table_bytes(R.HDRC)
    hdrbsize = 256 * (int(bool(flags["edge_diff"]))
                      + int(bool(flags["iqmatrix"]))
                      + int(flags["ibspsave"] > 1))
    freqblock = 4 * nfreq if ifreq == 2 else 0
    nbytesb = size_a + hdrbsize + size_c + freqblock
    nbytesd = 408 + nfreq * 32
    rec = nbytesb + nbytesd

    coff = lambda n: R._field_offset(R.HDRC, n)
    sweep = [0.0] * nsig if nsig == 1 else [360.0 * i / (nsig - 1) for i in range(nsig)]
    expl_freq = [freq1] if nfreq == 1 else \
        [freq1 + (freq2 - freq1) * i / (nfreq - 1) for i in range(nfreq)]

    blob = bytearray()
    for s in range(nsig):
        b = bytearray(rec)
        # --- header A ---
        b[0:4] = _be_i4(nbytesb)
        b[4:8] = _be_i4(nbytesd)
        b[10:11] = bytes([ord("1") if flags["edge_diff"] else ord("0")])
        b[384:388] = _be_i4(1 if flags["iqmatrix"] else 0)
        b[388:392] = _be_i4(int(flags["ibspsave"]))
        # --- header C at size_a + hdrbsize ---
        cb = size_a + hdrbsize
        def ci(n, v): b[cb + coff(n):cb + coff(n) + 4] = _be_i4(v)
        def cf(n, v): b[cb + coff(n):cb + coff(n) + 4] = _be_f4(v)
        ci("maxfreq", nfreq); ci("nfreq", nfreq); ci("ifreq", ifreq)
        cf("freq1", freq1); cf("freq2", freq2)
        ci("imono", 1 if mode == "incident" else 2)
        ci("maxaspects", nsig); ci("maxang", nsig); ci("maxfreqang", nfreq * nsig)
        # --- explicit freq block (ifreq==2), right before header-D ---
        if ifreq == 2:
            fb = cb + size_c
            for i, fv in enumerate(expl_freq):
                b[fb + 4 * i:fb + 4 * i + 4] = _be_f4(fv)
        # --- header D at nbytesb ---
        d = nbytesb
        if mode == "incident":
            azi = eli = sweep[s]; azo = elo = 0.0
        else:  # bistatic: incident pinned at 0/360, observation carries the sweep
            azi = eli = (0.0 if s % 2 == 0 else 360.0)
            azo = elo = sweep[s]
        b[d + 280:d + 284] = _be_f4(azi)
        b[d + 284:d + 288] = _be_f4(eli)
        b[d + 288:d + 292] = _be_f4(azo)
        b[d + 292:d + 296] = _be_f4(elo)
        # --- data: per freq f, pols (vv,vh,hv,hh) = (10/20/30/40 + f) + 1j*s ---
        dd = nbytesb + 408
        for f in range(nfreq):
            for p, base in enumerate((10, 20, 30, 40)):
                re_off = dd + (8 * f + 2 * p) * 4
                im_off = re_off + 4
                b[re_off:re_off + 4] = _be_f4(base + f)
                b[im_off:im_off + 4] = _be_f4(s)
        blob += b
    with open(path, "wb") as fh:
        fh.write(blob)
    return dict(hdrbsize=hdrbsize, nbytesb=nbytesb, nbytesd=nbytesd)


def check(name, nsig, nfreq, ifreq, f1, f2, flags, mode):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.ss")
        meta = build_ss(p, nsig, nfreq, ifreq, f1, f2, flags, mode)
        r = R.read_ss(p, verbose=False)

        errs = []
        # frequency axis
        exp_freq = np.linspace(f1, f2, nfreq)
        if not np.allclose(r["freq"], exp_freq, rtol=1e-5, atol=1e-4):
            errs.append(f"freq {np.round(r['freq'],3)} != {np.round(exp_freq,3)}")
        if r["num_freqs"] != nfreq:
            errs.append(f"num_freqs {r['num_freqs']} != {nfreq}")
        if not r["freq_axis_ok"]:
            errs.append("freq_axis_ok False (header-C mislocated)")
        # angle source + axis
        exp_src = "incident" if mode == "incident" else "observation"
        if r["angle_source"] != exp_src:
            errs.append(f"angle_source {r['angle_source']} != {exp_src}")
        az_uniq = np.unique(np.round(r["az"], 4))
        if az_uniq.size != nsig:
            errs.append(f"az has {az_uniq.size} uniq, expected {nsig}: {az_uniq}")
        if r["imono"] != (1 if mode == "incident" else 2):
            errs.append(f"imono {r['imono']}")
        # pols (check a couple of cells)
        for s in (0, nsig - 1):
            if not np.isclose(r["vv"][s][0], (10 + 0) + 1j * s):
                errs.append(f"vv[{s}][0]={r['vv'][s][0]} != {10+1j*s}")
            if nfreq > 1 and not np.isclose(r["hh"][s][nfreq - 1], (40 + nfreq - 1) + 1j * s):
                errs.append(f"hh[{s}][-1]={r['hh'][s][-1]}")

        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {name}  (hdrbsize={meta['hdrbsize']}, src={r['angle_source']}, "
              f"freq=[{r['freq'][0]:.2f}..{r['freq'][-1]:.2f}], imono={r['imono']})")
        for e in errs:
            print("        -", e)
        return not errs


if __name__ == "__main__":
    ok = True
    # baseline: monostatic, uniform freq, single CAD slot (the original synthetic case)
    ok &= check("monostatic/uniform/hdrb=256", nsig=5, nfreq=8, ifreq=1,
                f1=8.0, f2=12.0, flags=dict(edge_diff=False, iqmatrix=True, ibspsave=1),
                mode="incident")
    # header-B == 0 (all flags off): old code assumed 256 and mislocated header-C
    ok &= check("monostatic/uniform/hdrb=0", nsig=7, nfreq=16, ifreq=1,
                f1=2.0, f2=18.0, flags=dict(edge_diff=False, iqmatrix=False, ibspsave=1),
                mode="incident")
    # THE REAL CASE: bistatic sweep in observation + discrete freqs + hdrb=768
    ok &= check("bistatic/discrete/hdrb=768", nsig=9, nfreq=8, ifreq=2,
                f1=8.0, f2=12.0, flags=dict(edge_diff=True, iqmatrix=True, ibspsave=3),
                mode="observation")
    # bistatic + uniform + hdrb=512
    ok &= check("bistatic/uniform/hdrb=512", nsig=13, nfreq=4, ifreq=1,
                f1=9.0, f2=10.0, flags=dict(edge_diff=True, iqmatrix=False, ibspsave=2),
                mode="observation")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)
