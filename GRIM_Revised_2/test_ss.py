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
import unittest
import numpy as np
import read_ss as R
from grim_dataset import RcsGrid


def _be_i4(v):
    return int(v).to_bytes(4, "big", signed=True)


def _be_f4(v):
    return struct.pack(">f", float(v))


def build_ss(
    path,
    nsig,
    nfreq,
    ifreq,
    freq1,
    freq2,
    flags,
    mode,
    sweep_values=None,
    pre_frequency_padding=0,
):
    """Write a synthetic .ss. `flags` -> header-B size; `mode` in {incident, observation}."""
    size_a = R._table_bytes(R.HDRA)            # 648
    size_c = R._table_bytes(R.HDRC)
    hdrbsize = 256 * (int(bool(flags["edge_diff"]))
                      + int(bool(flags["iqmatrix"]))
                      + int(flags["ibspsave"] > 1))
    freqblock = 4 * nfreq if ifreq == 2 else 0
    nbytesb = (
        size_a + hdrbsize + size_c + int(pre_frequency_padding) + freqblock
    )
    nbytesd = 408 + nfreq * 32
    rec = nbytesb + nbytesd

    coff = lambda n: R._field_offset(R.HDRC, n)
    if sweep_values is None:
        sweep = [0.0] * nsig if nsig == 1 else [
            360.0 * i / (nsig - 1) for i in range(nsig)
        ]
    else:
        sweep = [float(value) for value in sweep_values]
        if len(sweep) != nsig:
            raise ValueError("sweep_values length must equal nsig")
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
            fb = nbytesb - freqblock
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


class TestSsParsing(unittest.TestCase):
    def test_monostatic_uniform_header_b_256(self):
        self.assertTrue(check(
            "monostatic/uniform/hdrb=256", 5, 8, 1, 8.0, 12.0,
            dict(edge_diff=False, iqmatrix=True, ibspsave=1), "incident",
        ))

    def test_monostatic_uniform_header_b_0(self):
        self.assertTrue(check(
            "monostatic/uniform/hdrb=0", 7, 16, 1, 2.0, 18.0,
            dict(edge_diff=False, iqmatrix=False, ibspsave=1), "incident",
        ))

    def test_bistatic_discrete_header_b_768(self):
        self.assertTrue(check(
            "bistatic/discrete/hdrb=768", 9, 8, 2, 8.0, 12.0,
            dict(edge_diff=True, iqmatrix=True, ibspsave=3), "observation",
        ))

    def test_bistatic_uniform_header_b_512(self):
        self.assertTrue(check(
            "bistatic/uniform/hdrb=512", 13, 4, 1, 9.0, 10.0,
            dict(edge_diff=True, iqmatrix=False, ibspsave=2), "observation",
        ))

    def test_rejects_nonintegral_record_framing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad-frame.ss")
            meta = build_ss(
                path, 2, 4, 1, 9.0, 10.0,
                dict(edge_diff=False, iqmatrix=False, ibspsave=1), "incident",
            )
            with open(path, "r+b") as stream:
                stream.seek(4)
                stream.write(_be_i4(meta["nbytesd"] + 1))
            with self.assertRaisesRegex(ValueError, r"exactly nbytesd=408\+32\*N"):
                R.read_ss(path, verbose=False)

    def test_rejects_truncated_final_record_instead_of_returning_partial_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.ss")
            build_ss(
                path, 3, 4, 1, 9.0, 10.0,
                dict(edge_diff=False, iqmatrix=True, ibspsave=1), "incident",
            )
            with open(path, "r+b") as stream:
                stream.seek(0, os.SEEK_END)
                stream.truncate(stream.tell() - 1)
            with self.assertRaisesRegex(ValueError, "truncated EOF"):
                R.read_ss(path, verbose=False)

    def test_rejects_header_c_count_mismatch_instead_of_scanning_a_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad-header.ss")
            meta = build_ss(
                path, 2, 4, 1, 9.0, 10.0,
                dict(edge_diff=True, iqmatrix=False, ibspsave=1), "incident",
            )
            header_c = R._table_bytes(R.HDRA) + meta["hdrbsize"]
            nfreq_offset = R._field_offset(R.HDRC, "nfreq")
            with open(path, "r+b") as stream:
                stream.seek(header_c + nfreq_offset)
                stream.write(_be_i4(99))
            with self.assertRaisesRegex(ValueError, "header-C validation failed"):
                R.read_ss(path, verbose=False)

    def test_accepts_variable_blocks_between_header_c_and_frequency_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "variable-blocks.ss")
            build_ss(
                path, 2, 4, 2, 9.0, 10.0,
                dict(edge_diff=False, iqmatrix=False, ibspsave=1),
                "incident",
                pre_frequency_padding=173,
            )
            parsed = R.read_ss(path, verbose=False)
            np.testing.assert_allclose(parsed["freq"], np.linspace(9.0, 10.0, 4))

    def test_rejects_valid_looking_header_c_at_an_illegal_byte_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "header-c-collision.ss")
            size_c = R._table_bytes(R.HDRC)
            meta = build_ss(
                path, 1, 4, 1, 9.0, 10.0,
                dict(edge_diff=False, iqmatrix=False, ibspsave=1),
                "incident",
                pre_frequency_padding=size_c + 32,
            )
            header_c = R._table_bytes(R.HDRA) + meta["hdrbsize"]
            with open(path, "r+b") as stream:
                stream.seek(header_c)
                valid_header = stream.read(size_c)
                stream.seek(header_c + size_c + 8)
                stream.write(valid_header)
                stream.seek(header_c + R._field_offset(R.HDRC, "nfreq"))
                stream.write(_be_i4(99))
            with self.assertRaisesRegex(ValueError, "header-C validation failed"):
                R.read_ss(path, verbose=False)

    def test_grid_loader_rejects_duplicate_angular_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "collision.ss")
            build_ss(
                path, 2, 4, 1, 9.0, 10.0,
                dict(edge_diff=False, iqmatrix=False, ibspsave=1),
                "incident",
                sweep_values=[15.0, 15.0],
            )
            with self.assertRaisesRegex(ValueError, "angular coordinate collision"):
                RcsGrid.load_ss(path)

    def test_ss_to_grim_round_trip_preserves_ghz_axes_and_complex_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.ss")
            build_ss(
                source, 2, 3, 2, 9.0, 11.0,
                dict(edge_diff=True, iqmatrix=True, ibspsave=3),
                "incident",
                sweep_values=[10.0, 20.0],
                pre_frequency_padding=37,
            )

            converted = RcsGrid.load_ss(source)
            np.testing.assert_allclose(converted.frequencies, [9.0, 10.0, 11.0])
            np.testing.assert_allclose(converted.azimuths, [10.0, 20.0])
            np.testing.assert_allclose(converted.elevations, [10.0, 20.0])
            np.testing.assert_array_equal(converted.polarizations, ["VV", "VH", "HV", "HH"])
            self.assertEqual(converted.units["frequency"], "GHz")
            self.assertEqual(converted.linear_quantity(), "sigma_3d")

            expected_first = np.asarray([10.0, 11.0, 12.0], dtype=np.complex64)
            expected_second = expected_first + np.complex64(1j)
            np.testing.assert_allclose(converted.rcs[0, 0, :, 0], expected_first)
            np.testing.assert_allclose(converted.rcs[1, 1, :, 0], expected_second)

            output = converted.save(os.path.join(tmp, "converted"))
            restored = RcsGrid.load(output)
            np.testing.assert_allclose(restored.frequencies, converted.frequencies)
            np.testing.assert_array_equal(restored.polarizations, converted.polarizations)
            np.testing.assert_allclose(restored.rcs, converted.rcs, equal_nan=True)
            self.assertEqual(restored.units["frequency"], "GHz")


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
