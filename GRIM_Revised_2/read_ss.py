"""
read_ss.py - standalone reader for Xpatch .ss signature files.

Ported from the hand-transcribed MATLAB ssread.m / xpheaders.m in this project.
READ ONLY (no write support). Pulls exactly what GRIM needs: the complex
scattering data (VV/VH/HV/HH), the frequency axis, and the azimuth/elevation of
each signal.

How it works -- the .ss format is self-framing. Every "signal" record starts
with two big-endian int32s, nbytesb and nbytesd:

    record_size = nbytesb + nbytesd
    nbytesb     = byte offset from record start to header-D
    nbytesd     = header-D (408 bytes) + complex data block
                  => num_freqs = (nbytesd - 408) / 32
                     (32 bytes per freq = 4 pols x complex64)

That framing lets us reach header-D (az/el) and the data block WITHOUT parsing
the advanced / raytrace / materials blocks that ssread.m walks through (their
contents are unused here). The only table-derived read is header-C (the
frequency axis). Its offset depends on the *variable-length* header-B, whose
size we compute from the header-A flags (edge_diff/iqmatrix/ibspsave) exactly as
ssread.m's readheaderb does -- header-B is 256 bytes per enabled CAD-file slot
(0/256/512/768 total), NOT a fixed 256. We then cross-check header-C by requiring
maxfreq == framing num_freqs (and ifreq in {1,2}); if that fails we scan for the
real offset. az/el/data stay correct regardless because they are pinned by the
framing ints, not by the field tables.

Az/el: header-D carries incident (azinc/elinc) AND observation (azobs/elobs)
angles. ssread.m returns incident, which is right for monostatic data but stays
fixed for bistatic runs (imono==2), where the sweep is in observation. We pick
whichever pair actually varies across signals so both cases load correctly.

VERIFY the printed numbers against MATLAB `ssread` output before trusting this.
"""

import sys
import numpy as np

BE_I4 = np.dtype(">i4")   # big-endian int32  (MATLAB '*int',  'ieee-be')
BE_F4 = np.dtype(">f4")   # big-endian single (MATLAB '*float', 'ieee-be')

_TYPE_BYTES = {"int": 4, "float": 4, "char": 1, "": 1}

# --- field tables, ported verbatim from xpheaders.m -------------------------
# (type, name, count).  type "" / name "" means: skip `count` BYTES.

HDRA = [
    ("int", "nbytesb", 1), ("int", "nbytesd", 1),
    ("char", "bin_head_form", 1), ("char", "method", 1),
    ("char", "edge_diff", 1), ("char", "polar", 1),
    ("char", "x_version_num", 16), ("char", "hardware", 8),
    ("char", "host_machine", 16), ("char", "op_system", 16),
    ("char", "op_release", 8), ("char", "op_version", 8),
    ("char", "mem_version", 8), ("int", "num_tasks", 1),
    ("char", "simTitle", 256), ("int", "simDate", 3),
    ("int", "restart_Date", 3), ("int", "restart_count", 1),
    ("int", "ipoedge", 1), ("int", "iqmatrix", 1),
    ("int", "ibspsave", 1), ("char", "acadfct", 256),
]

HDRB_SLOT = 256   # one CAD-file slot; header-B is N*256 (N enabled slots) -- see _hdrb_size()

# the 'C' header == SsStandardC: the standard-parameters block (read at hdrc_off)
HDRC = [
    ("int","maxlay",1),("int","maxrstep",1),("int","maxchild",1),("int","maxson",1),
    ("int","maxram",1),("int","maxband",1),("int","maxpixx",1),("int","maxpixy",1),
    ("int","max114knot",1),("int","igui",1),("float","safenuss",1),("float","edgeblockwave",1),
    ("int","maxfiles",1),("int","maxaspects",1),("int","maxang",1),("int","maxfreqbin",1),
    ("int","maxbncpl",1),("int","maxcoat",1),("int","maxbulkcoat",1),("int","maxexpnuss",1),
    ("int","maxfreq",1),("int","maxedge",1),("int","maxfreqang",1),("int","maxramf",1),
    ("int","maxrangestep",1),("int","maxstackson",1),("int","iramtot",1),("int","maxnfct",1),
    ("int","maxnnod",1),("char","modelTitle",256),("float","model_roll_angle",1),("float","bmin",3),
    ("float","bmax",3),("int","mctot",1),("int","mcbadtot",1),("int","mabsorbtot",1),
    ("float","areatot",1),("int","itracetype",1),("int","iunit",1),("int","ifreq",1),
    ("float","freq1",1),("float","freq2",1),("int","nfreq",1),("int","inorange",1),
    ("float","range1",1),("float","range2",1),("int","nrange",1),("int","imono",1),
    ("float","rt071",1),("float","rt072",1),("int","nrt07",1),("float","rp071",1),
    ("float","rp072",1),("int","nrp07",1),("float","theob1",1),("float","theob2",1),
    ("int","ntheob",1),("float","phiob1",1),("float","phiob2",1),("int","nphiob",1),
    ("int","ioutformat",1),("int","iaddedge",1),("int","ipozbuff",1),("float","cellmax",1),
    ("float","blockangle",1),("int","irightnormal",1),("float","pixsize",1),("int","ipixout",1),
    ("int","maxvoxdepth",1),("int","maxvoxl",1),("int","maxbncin",1),("float","raywvel",1),
    ("float","raywvaz",1),("float","nscale",1),("int","icoatabsorb",1),("int","ipec",1),
    ("float","pecfudge",1),("float","delf9",1),("int","maxang_in",1),("int","num_advanced",1),
]

# header-D minimal form (the 'd' case): 280 skip, az/el/azobs/elobs, 112 skip
# 280 + 4*4 + 112 = 408  <- matches the magic number in ssread.m
HDRDMIN = [
    ("", "", 280),
    ("float", "azinc", 1), ("float", "elinc", 1),
    ("float", "azobs", 1), ("float", "elobs", 1),
    ("", "", 112),
]


def _table_bytes(table):
    return sum(_TYPE_BYTES[t] * n for t, _, n in table)


def _parse_table(buf, table):
    """Parse a packed big-endian record `buf` per `table`; return name -> value(s)."""
    out, off = {}, 0
    for typ, name, count in table:
        nbytes = _TYPE_BYTES[typ] * count
        chunk = buf[off:off + nbytes]
        if typ == "int":
            out[name] = np.frombuffer(chunk, BE_I4, count)
        elif typ == "float":
            out[name] = np.frombuffer(chunk, BE_F4, count)
        elif typ == "char":
            out[name] = bytes(chunk)
        off += nbytes
    return out


def _i4(buf, off):
    return int(np.frombuffer(buf[off:off + 4], BE_I4)[0])


def _fmt_field(v):
    """Format one parsed header field: bytes -> string, numbers -> scalar or list."""
    if isinstance(v, (bytes, bytearray)):
        return repr(v.decode("ascii", errors="replace").rstrip("\x00").rstrip())
    if v.dtype.kind == "f":
        vals = [f"{float(x):.6g}" for x in v.reshape(-1)]
    else:
        vals = [str(int(x)) for x in v.reshape(-1)]
    return vals[0] if len(vals) == 1 else "[" + ", ".join(vals) + "]"


def print_struct(name, hdr, table=None, base=0):
    """Pretty-print a parsed header dict as `@off field = value`, in table order.

    `@off` is the byte offset of the field within the struct; if `base` is given
    it's added so the number is the absolute file offset. Use it to line fields
    up against the reference reader and spot where the layout drifts.
    """
    offsets = {}
    if table is not None:
        off = 0
        for typ, fname, count in table:
            if fname:
                offsets[fname] = off
            off += _TYPE_BYTES[typ] * count
    print(f"{name}:" + (f"   (struct starts at file offset {base})" if base else ""))
    width = max((len(k) for k in hdr), default=0)
    for k, v in hdr.items():
        tag = f"@{base + offsets[k]:>5}" if k in offsets else "      "
        print(f"  {tag}  {k:<{width}} = {_fmt_field(v)}")


def _field_offset(table, name):
    """Byte offset of a field within its packed struct table."""
    off = 0
    for typ, fname, count in table:
        if fname == name:
            return off
        off += _TYPE_BYTES[typ] * count
    raise KeyError(name)


def _hdrb_size(raw):
    """Header-B length in bytes (it is variable, not fixed).

    Per xpheaders.m readheaderb, header-B holds one 256-byte CAD-file name per
    enabled slot, gated by these header-A flags:
        edge_diff == '1'   (acadedge)
        iqmatrix  == 1     (acadcurv)
        ibspsave   > 1     (acadbsp)
    So the block is 0/256/512/768 bytes. Computing it here (instead of assuming
    256) puts header-C at the correct offset, which is what fixes the freq axis.
    """
    edge_diff = int(raw[_field_offset(HDRA, "edge_diff")])      # 1 char byte
    iqmatrix = _i4(raw, _field_offset(HDRA, "iqmatrix"))
    ibspsave = _i4(raw, _field_offset(HDRA, "ibspsave"))
    n = (edge_diff == ord("1")) + (iqmatrix == 1) + (ibspsave > 1)
    return int(n) * HDRB_SLOT


def _self_check():
    """Fail loudly at import on the table edits that DO corrupt offsets.

    Field offsets come from type*count, never from the name spelling, so a typo
    in a name the reader doesn't query is harmless. But a wrong count/type
    silently shifts every following field, and a typo in a load-bearing name
    breaks a lookup mid-parse. This asserts the two fixed-size headers and that
    every name we look up by name still resolves -- turning both into an
    immediate, explicit error instead of a subtle mis-read.
    """
    for tbl, name, want in (("HDRA", HDRA, 648), ("HDRDMIN", HDRDMIN, 408)):
        got = _table_bytes(name)
        if got != want:
            raise ValueError(f"{tbl} must be {want} bytes but sums to {got} -- "
                             "check a count/type column (not a name typo).")
    looked_up = {
        "HDRA": (HDRA, ("edge_diff", "iqmatrix", "ibspsave")),
        "HDRC": (HDRC, ("maxfreq", "ifreq", "nfreq", "imono", "freq1", "freq2")),
        "HDRDMIN": (HDRDMIN, ("azinc", "elinc", "azobs", "elobs")),
    }
    for tbl, (table, names) in looked_up.items():
        for nm in names:
            try:
                _field_offset(table, nm)
            except KeyError:
                raise ValueError(f"{tbl} is missing load-bearing field '{nm}' "
                                 "(name typo in the table?).") from None


_self_check()


def scan_hdrc_offset(raw, num_freqs, lo=600, hi=2400):
    """Candidate header-C offsets: where the int32 at (offset + maxfreq_rel) equals
    num_freqs. maxfreq must equal the framing-derived freq count, so a match flags a
    plausible header-C start (and thus the real hdrbsize = offset - len(header-A))."""
    rel = _field_offset(HDRC, "maxfreq")
    hi = min(hi, raw.size - rel - 4)
    return [o for o in range(lo, hi) if _i4(raw, o + rel) == num_freqs]


def read_ss(path, verbose=True):
    raw = np.fromfile(path, dtype=np.uint8)
    filesize = raw.size
    if filesize < 8:
        raise ValueError(f"{path}: too small to be a .ss file ({filesize} bytes)")

    size_a = _table_bytes(HDRA)        # = 648

    # frequency count from the first record's framing: nbytesd = n_freqs*32 + 408
    nbytesd0 = _i4(raw, 4)
    if nbytesd0 <= 408:
        raise ValueError(f"{path}: first nbytesd={nbytesd0} (<=408); not a .ss file?")
    num_freqs0 = (nbytesd0 - 408) // 32

    # header-C starts at size_a + hdrbsize (ssread order A,B,C). header-B is
    # variable-length, so derive its size from the header-A flags rather than
    # guessing -- this is what places header-C (and the freq axis) correctly.
    rel_maxfreq = _field_offset(HDRC, "maxfreq")
    rel_ifreq = _field_offset(HDRC, "ifreq")

    def _hdrc_ok(off):
        if off < 0 or off + _table_bytes(HDRC) > raw.size:
            return False
        if _i4(raw, off + rel_maxfreq) != num_freqs0:   # maxfreq must == framing count
            return False
        return _i4(raw, off + rel_ifreq) in (1, 2)      # ifreq is a 1/2 spacing flag

    hdrbsize = _hdrb_size(raw)
    hdrc_off = size_a + hdrbsize
    if not _hdrc_ok(hdrc_off):
        cands = [o for o in scan_hdrc_offset(raw, num_freqs0) if _hdrc_ok(o)] \
            or scan_hdrc_offset(raw, num_freqs0)
        if verbose:
            print(f"  note: flag-derived header-C@{hdrc_off} (hdrbsize={hdrbsize}) "
                  f"failed validation; scan candidates={cands[:8]}")
        if cands:
            hdrc_off = min(cands, key=lambda o: abs(o - (size_a + hdrbsize)))

    # --- header C (frequency axis); read once from the first record ----------
    hdrc = _parse_table(raw[hdrc_off:hdrc_off + _table_bytes(HDRC)], HDRC)
    ifreq = int(hdrc["ifreq"][0])
    maxfreq = int(hdrc["maxfreq"][0])
    freq1 = float(hdrc["freq1"][0])
    freq2 = float(hdrc["freq2"][0])
    imono = int(hdrc["imono"][0])              # 1: mono-static, 2: bistatic

    # --- walk records by framing --------------------------------------------
    # header-D carries both incident (azinc/elinc) and observation (azobs/elobs)
    # angles; collect both and decide which is the real sweep after the walk.
    ang = {"azinc": [], "elinc": [], "azobs": [], "elobs": []}
    pol = {"vv": [], "vh": [], "hv": [], "hh": []}
    p, nsig, num_freqs_global = 0, 0, None
    while p + 8 <= filesize:
        nbytesb = _i4(raw, p)
        nbytesd = _i4(raw, p + 4)
        if nbytesb <= 0 or nbytesd <= 408:
            if verbose:
                print(f"  record {nsig}: framing not set (nbytesb={nbytesb}, "
                      f"nbytesd={nbytesd}); stopping")
            break
        num_freqs = (nbytesd - 408) // 32
        if num_freqs_global is None:
            num_freqs_global = num_freqs

        # header-D (minimal): incident/observation az/el at p + nbytesb
        d = _parse_table(raw[p + nbytesb: p + nbytesb + 408], HDRDMIN)
        for k in ang:
            ang[k].append(float(d[k][0]))

        # complex data: 32*num_freqs bytes at p + nbytesb + 408
        dstart = p + nbytesb + 408
        nbytes = 32 * num_freqs
        if dstart + nbytes > filesize:
            if verbose:
                print(f"  record {nsig}: data block truncated; stopping")
            break
        chunk = np.frombuffer(raw[dstart:dstart + nbytes], BE_F4, 8 * num_freqs)
        c = chunk[0::2] + 1j * chunk[1::2]      # num_freqs*4 complex
        pol["vv"].append(c[0::4]); pol["vh"].append(c[1::4])
        pol["hv"].append(c[2::4]); pol["hh"].append(c[3::4])

        nsig += 1
        p += nbytesb + nbytesd

    if nsig == 0:
        raise ValueError(f"{path}: no readable signal records")

    # --- frequency axis ------------------------------------------------------
    # Length is pinned to the framing count (num_freqs_global), which always
    # matches the per-signal data block; maxfreq equals it when header-C is right.
    nfa = num_freqs_global
    if ifreq == 2:
        # explicit freqs: nfa float32 immediately before header-D of record 0
        nbytesb0 = _i4(raw, 0)
        fstart = nbytesb0 - 4 * nfa
        freqdata = np.frombuffer(raw[fstart:fstart + 4 * nfa], BE_F4, nfa).copy()
    else:
        freqdata = np.linspace(freq1, freq2, nfa)

    # --- choose incident vs observation angles ------------------------------
    # Monostatic (imono==1): incident == observation. Bistatic (imono==2): the
    # sweep is usually in observation while incident stays fixed (0/360). Pick
    # whichever pair actually varies, so both run types load with a real axis.
    def _nuniq(vals):
        return int(np.unique(np.round(np.asarray(vals, float), 4)).size)
    az_inc, el_inc = np.asarray(ang["azinc"]), np.asarray(ang["elinc"])
    az_obs, el_obs = np.asarray(ang["azobs"]), np.asarray(ang["elobs"])
    n_inc = max(_nuniq(az_inc), _nuniq(el_inc))
    n_obs = max(_nuniq(az_obs), _nuniq(el_obs))
    if n_obs > n_inc:
        az, el, angle_source = az_obs, el_obs, "observation"
    else:
        az, el, angle_source = az_inc, el_inc, "incident"

    match = (num_freqs_global == maxfreq)
    result = {
        "az": np.asarray(az), "el": np.asarray(el),
        "freq": freqdata, "num_freqs": num_freqs_global,
        "maxfreq": maxfreq, "ifreq": ifreq, "imono": imono,
        "az_inc": az_inc, "el_inc": el_inc, "az_obs": az_obs, "el_obs": el_obs,
        "angle_source": angle_source,
        "vv": np.asarray(pol["vv"]), "vh": np.asarray(pol["vh"]),   # (nsig, num_freqs)
        "hv": np.asarray(pol["hv"]), "hh": np.asarray(pol["hh"]),
        "header_c": hdrc, "freq_axis_ok": match,
    }

    if verbose:
        print(f"  signals           : {nsig}")
        print(f"  num_freqs (framing): {num_freqs_global}    maxfreq (header C): {maxfreq}    match: {match}")
        print(f"  header-C offset   : {hdrc_off}  (size_a={_table_bytes(HDRA)} + hdrbsize={hdrbsize})")
        if not match:
            print("  !! header-C mismatch -> wrong offset; FREQ AXIS SUSPECT")
            print("     (az/el/data are framing-pinned and still trustworthy)")
            rel = _field_offset(HDRC, "maxfreq")
            for hb in (0, HDRB_SLOT, 2 * HDRB_SLOT, 3 * HDRB_SLOT):
                o = _table_bytes(HDRA) + hb
                mf = _i4(raw, o + rel) if o + rel + 4 <= raw.size else None
                flag = "  <- matches num_freqs!" if mf == num_freqs_global else ""
                print(f"     hdrbsize={hb:>4} -> header-C@{o:<5} maxfreq={mf}{flag}")
            cands = scan_hdrc_offset(raw, num_freqs_global)
            print(f"     offsets giving maxfreq=={num_freqs_global}: {cands[:16]}"
                  + (" ..." if len(cands) > 16 else ""))
        print(f"  ifreq             : {ifreq}   freq1={freq1:.6g}  freq2={freq2:.6g}   imono={imono}")
        print(f"  angle source      : {angle_source}  (incident n_uniq={n_inc}, observation n_uniq={n_obs})")
        print(f"    azinc {_nuniq(az_inc):>4} uniq [{az_inc.min():.3f}..{az_inc.max():.3f}]   "
              f"azobs {_nuniq(az_obs):>4} uniq [{az_obs.min():.3f}..{az_obs.max():.3f}]")
        print(f"    elinc {_nuniq(el_inc):>4} uniq [{el_inc.min():.3f}..{el_inc.max():.3f}]   "
              f"elobs {_nuniq(el_obs):>4} uniq [{el_obs.min():.3f}..{el_obs.max():.3f}]")
        print(f"  az range (chosen) : {az.min():.4f} .. {az.max():.4f}")
        print(f"  el range (chosen) : {el.min():.4f} .. {el.max():.4f}")
        print(f"  freq[:3]          : {np.round(freqdata[:3], 6)}")
        print(f"  vv[sig0][:2]      : {result['vv'][0][:2]}")
        print()
        print_struct("SsStandardC", hdrc, HDRC, base=hdrc_off)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python read_ss.py FILE.ss")
        sys.exit(1)
    print(f"reading {sys.argv[1]}")
    read_ss(sys.argv[1])
