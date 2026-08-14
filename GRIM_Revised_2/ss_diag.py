"""
ss_diag.py - READ-ONLY diagnostic dump for an Xpatch .ss file.

Does NOT modify anything. Reuses the tables/helpers in read_ss.py and adds:
  * the FULL header-D parse (read_ss only reads the minimal 408-byte form),
    so we can see azinc/elinc vs azobs/elobs vs azstart/elstart/azstop/elstop;
  * per-field variation across ALL signals -> reveals which field actually sweeps;
  * header-C location candidates + the nfreq-vs-maxfreq-vs-framing comparison;
  * a raw dump of the float region just before header-D (the ifreq==2 freq block).

usage:  python ss_diag.py FILE.ss
"""

import sys
import numpy as np
import read_ss as R

# Full header-D ('D' case) from xpheaders.m, lines 226-263.  Total = 408 bytes,
# so it occupies exactly the same span as the minimal 'd' form read_ss uses.
HDRD_FULL = [
    ("int", "simDate", 3), ("float", "stime", 1), ("float", "run_time_used", 1),
    ("int", "node_number_used", 1), ("char", "modelTitle", 256),
    ("float", "azinc", 1), ("float", "elinc", 1),
    ("float", "azobs", 1), ("float", "elobs", 1),
    ("int", "kaztot5nscale", 1), ("int", "keltot5nscale", 1),
    ("int", "kaztot", 1), ("int", "keltot", 1),
    ("float", "deltax", 1), ("float", "deltay", 1),
    ("float", "azstart", 1), ("float", "elstart", 1),
    ("float", "azstop", 1), ("float", "elstop", 1),
    ("int", "mtot9", 1), ("int", "mout9", 1), ("int", "mmiss9", 1),
    ("int", "m2many9", 1), ("int", "mabsorb9", 1), ("int", "mexit9", 1),
    ("int", "jb01", 1), ("int", "jb02", 1), ("int", "jb03", 1), ("int", "jb04", 1),
    ("int", "jb05", 1), ("int", "jb10", 1), ("int", "jb15", 1), ("int", "jb20", 1),
    ("int", "jb30", 1), ("int", "jb40", 1), ("int", "jb50", 1), ("int", "mhit9", 1),
]

SWEEP_FIELDS = ["azinc", "elinc", "azobs", "elobs",
                "azstart", "elstart", "azstop", "elstop"]


def summarize(name, vals):
    v = np.asarray(vals, dtype=float)
    u = np.unique(np.round(v, 4))
    head = ", ".join(f"{x:.4g}" for x in v[:6])
    ushow = ", ".join(f"{x:.4g}" for x in u[:8]) + (" ..." if u.size > 8 else "")
    print(f"  {name:<9} n_unique={u.size:<5} range=[{v.min():.4g} .. {v.max():.4g}]"
          f"  first6=[{head}]  uniq=[{ushow}]")


def main(path):
    raw = np.fromfile(path, dtype=np.uint8)
    print(f"file: {path}   size={raw.size} bytes\n")

    size_a = R._table_bytes(R.HDRA)
    nbytesb0 = R._i4(raw, 0)
    nbytesd0 = R._i4(raw, 4)
    num_freqs0 = (nbytesd0 - 408) // 32
    print(f"record0 framing:  nbytesb={nbytesb0}  nbytesd={nbytesd0}"
          f"  -> num_freqs(framing)={num_freqs0}\n")

    # ---- header-C: flag-derived offset (the corrected method) + candidates ----
    rel_mf = R._field_offset(R.HDRC, "maxfreq")
    edge = int(raw[R._field_offset(R.HDRA, "edge_diff")])
    iqm = R._i4(raw, R._field_offset(R.HDRA, "iqmatrix"))
    ibsp = R._i4(raw, R._field_offset(R.HDRA, "ibspsave"))
    hdrbsize = R._hdrb_size(raw)
    flag_off = size_a + hdrbsize
    cands = R.scan_hdrc_offset(raw, num_freqs0)
    edge_c = chr(edge) if 32 <= edge < 127 else "?"
    print(f"header-A flags: edge_diff={edge}('{edge_c}')  iqmatrix={iqm}  ibspsave={ibsp}"
          f"   -> hdrbsize={hdrbsize} ({hdrbsize // 256} slots)")
    print(f"header-C: flag-derived offset={flag_off}; maxfreq@flag={R._i4(raw, flag_off + rel_mf)}"
          f"  (want {num_freqs0})")
    print(f"          offsets where int32==num_freqs (maxfreq anchor): {cands[:16]}"
          + (" ..." if len(cands) > 16 else ""))

    # parse header-C at whatever read_ss would choose, and show the freq-relevant fields
    chosen = flag_off
    if R._i4(raw, flag_off + rel_mf) != num_freqs0 and cands:
        chosen = min(cands, key=lambda o: abs(o - flag_off))
    hdrc = R._parse_table(raw[chosen:chosen + R._table_bytes(R.HDRC)], R.HDRC)
    print(f"\nheader-C parsed at offset {chosen}:")
    for k in ("ifreq", "maxfreq", "nfreq", "maxfreqbin", "maxang", "maxaspects",
              "maxfreqang", "imono", "freq1", "freq2", "range1", "range2"):
        print(f"    {k:<11} = {R._fmt_field(hdrc[k])}")
    print(f"    -> compare: num_freqs(framing)={num_freqs0}  maxfreq={int(hdrc['maxfreq'][0])}"
          f"  nfreq={int(hdrc['nfreq'][0])}")

    # ---- walk records, collect FULL header-D fields ----
    fields = {k: [] for k in SWEEP_FIELDS}
    p, n = 0, 0
    first_dumps = []
    while p + 8 <= raw.size:
        nbytesb = R._i4(raw, p)
        nbytesd = R._i4(raw, p + 4)
        if nbytesb <= 0 or nbytesd <= 408:
            break
        d = R._parse_table(raw[p + nbytesb: p + nbytesb + 408], HDRD_FULL)
        for k in SWEEP_FIELDS:
            fields[k].append(float(d[k][0]))
        if n < 3:
            first_dumps.append((n, p, nbytesb, nbytesd, d))
        last = (n, p, nbytesb, nbytesd, d)
        n += 1
        p += nbytesb + nbytesd

    print(f"\nsignals walked: {n}\n")
    print("per-field variation across ALL signals (which one is the real sweep?):")
    for k in SWEEP_FIELDS:
        summarize(k, fields[k])

    print("\nfull header-D for first/last signals:")
    for (idx, pp, nb, nd, d) in first_dumps + [("last",) + last[1:]]:
        vals = {k: float(d[k][0]) for k in SWEEP_FIELDS}
        print(f"  sig {idx}: " + "  ".join(f"{k}={vals[k]:.4g}" for k in SWEEP_FIELDS))

    # ---- raw float region just before header-D of record 0 (ifreq==2 block) ----
    mf = int(hdrc["maxfreq"][0])
    win = max(mf, num_freqs0) + 6
    start = max(0, nbytesb0 - 4 * win)
    floats = np.frombuffer(raw[start:nbytesb0], R.BE_F4)
    print(f"\nfloat region [{start}..{nbytesb0}) just before header-D (last {floats.size} f32):")
    print("  tail:", np.array2string(floats[-min(12, floats.size):],
                                      precision=4, max_line_width=120))
    print(f"  (if ifreq==2, the freq axis is the last num_freqs={num_freqs0} of these)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ss_diag.py FILE.ss")
        sys.exit(1)
    main(sys.argv[1])
