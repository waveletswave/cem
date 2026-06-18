#!/usr/bin/env python3
"""
coastsat_single_site.py
=======================

Extract ONE shoreline for ONE site and save it as a two-column ``.xy`` file
(projected metres) for use as an initial shoreline in the Coastline Evolution
Model (CEM).

This is the lightweight, single-site counterpart to the regional
``drawShorelines`` notebook. It has two interchangeable sources -- pick one with
a subcommand (this is the "switch"):

  zenodo   Reconstruct a shoreline for a chosen YEAR from the pre-processed
           CoastSat "US East Coast" time-series on Zenodo
           (doi:10.5281/zenodo.15626279, latest record 18435286).
           * No Google Earth Engine, no imagery download, fully non-interactive.
           * Covers US Atlantic + Gulf sandy coasts only (Delaware -> Texas).

  imagery  Run CoastSat on satellite imagery for a small ROI and digitise one
           reference shoreline (this is exactly the path that produced the
           Erie_spit_ESPIn.xy file in espin-2021/coastal).
           * Works anywhere in the world.
           * Needs the CoastSat toolbox importable AND an authenticated Google
             Earth Engine session. The reference-shoreline step is INTERACTIVE
             (you click the shoreline on one image); it is cached afterwards.

Output (both routes): an ``.xy`` file of ``x y`` rows in projected metres (UTM
by default), the format CEM's ``shorelinetogrid()`` reads with ``np.loadtxt``.

Scope: this tool stops at the ``.xy``. Turning the ``.xy`` into a Dean-profile
depth grid (``shorelinetogrid``) and feeding CEM is the CEM-side step and is out
of scope here.

Examples
--------
# Route 1 (no GEE). You have unzipped the Zenodo shoreline_data/ folder.
python coastsat_single_site.py zenodo \
    --site usa_NC0030 --year 2020 \
    --data-dir shoreline_data \
    --out usa_NC0030_2020.xy

# Route 2 (needs GEE + CoastSat on the path). ROI as "lon,lat lon,lat ...".
python coastsat_single_site.py imagery \
    --polygon "-80.1818,42.1755 -80.0534,42.1759 -80.0553,42.0998 -80.1780,42.1001" \
    --dates 1985-04-01 1985-09-01 --sat L5 \
    --sitename Erie_new --filepath ./data \
    --out Erie_new.xy
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import namedtuple


# --------------------------------------------------------------------------- #
# Shared helpers (pure, no GEE / no heavy geo deps)                            #
# --------------------------------------------------------------------------- #
def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _transformer(src_epsg: int, dst_epsg: int):
    from pyproj import Transformer
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}",
                                always_xy=True)


def reproject(xs, ys, src_epsg: int, dst_epsg: int):
    """Reproject parallel x/y sequences from src EPSG to dst EPSG."""
    xs = list(xs)
    ys = list(ys)
    if int(src_epsg) == int(dst_epsg):
        return xs, ys
    tr = _transformer(src_epsg, dst_epsg)
    X, Y = tr.transform(xs, ys)
    # pyproj returns scalars for length-1 input in some versions; normalise
    if not hasattr(X, "__len__"):
        X, Y = [X], [Y]
    return list(X), list(Y)


def is_projected(epsg: int) -> bool:
    """True if the EPSG code is a projected (metric) CRS, not geographic."""
    from pyproj import CRS
    return CRS.from_epsg(int(epsg)).is_projected


def write_xy(xs, ys, path: str, close_ring: bool = False,
             precision: int = 6, header=None) -> int:
    """Write 'x y' rows (projected metres) to *path*. Returns point count.

    *header* (an iterable of strings) is written as leading ``# ...`` comment
    lines -- np.loadtxt skips these, so a downstream notebook can recover, e.g.,
    the land direction and EPSG from the file instead of guessing them.
    """
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    if not xs:
        raise ValueError("no points to write")
    if close_ring and (xs[0], ys[0]) != (xs[-1], ys[-1]):
        xs.append(xs[0])
        ys.append(ys[0])
    fmt = "%.{}f".format(int(precision))
    with open(path, "w") as fh:
        for line in (header or []):
            fh.write(f"# {line}\n")
        for x, y in zip(xs, ys):
            fh.write(f"{fmt % x} {fmt % y}\n")
    return len(xs)


def _natural_key(tid: str):
    """Sort key that orders transect ids by their numeric parts."""
    nums = re.findall(r"\d+", str(tid))
    return tuple(int(n) for n in nums) if nums else (str(tid),)


# --------------------------------------------------------------------------- #
# Route 1 -- reconstruct from the Zenodo US-East-Coast time-series             #
# --------------------------------------------------------------------------- #
def extract_zenodo(site: str,
                   year: int,
                   transects_path: str,
                   timeseries_path: str,
                   target_epsg: int | None = None,
                   src_epsg: int = 4326,
                   month: int = 6,
                   day: int = 30,
                   smooth_days: int = 180,
                   order: str = "ascending"):
    """
    Reconstruct one shoreline for *site* at *year* from the pre-processed
    CoastSat time-series.

    For every cross-shore transect of the site we read its chainage (distance
    from the landward, non-erodible transect origin to the shoreline) time
    series, interpolate it in time to <year>-<month>-<day>, then place a point
    that far seaward along the transect. The ordered set of points is the
    reconstructed shoreline.

    Crucially the placement is done in a PROJECTED (metric) CRS, because the
    chainages are in metres -- never in lon/lat degrees.

    Returns (xs, ys, out_epsg, n_points) with xs/ys in *target_epsg*.
    """
    import json
    import numpy as np
    import pandas as pd

    # -- transects for this site -------------------------------------------- #
    with open(transects_path) as fh:
        gj = json.load(fh)
    feats = [ft for ft in gj.get("features", [])
             if str(ft.get("properties", {}).get("site_id")) == str(site)]
    if not feats:
        raise SystemExit(
            f"No transects with site_id == {site!r} found in {transects_path}.\n"
            f"Check the site id (current Zenodo format looks like 'usa_NC0030')."
        )

    # choose the output CRS now (must be metric) so we can place points in metres
    lons, lats = [], []
    for ft in feats:
        c0 = ft["geometry"]["coordinates"][0]
        lons.append(c0[0])
        lats.append(c0[1])
    if target_epsg is None:
        # transect coords are lon/lat by default -> derive UTM from centroid
        if int(src_epsg) == 4326:
            clon = sum(lons) / len(lons)
            clat = sum(lats) / len(lats)
        else:
            clon_l, clat_l = reproject([sum(lons) / len(lons)],
                                       [sum(lats) / len(lats)], src_epsg, 4326)
            clon, clat = clon_l[0], clat_l[0]
        target_epsg = utm_epsg(clon, clat)
    if not is_projected(target_epsg):
        raise SystemExit(
            f"Output EPSG {target_epsg} is geographic (lon/lat). The .xy must be "
            f"in projected metres -- pass a UTM/metric --epsg, or omit it to auto-pick UTM."
        )

    tr = None if int(src_epsg) == int(target_epsg) else _transformer(src_epsg, target_epsg)

    # transect id -> (origin_x, origin_y, unit_x, unit_y) all in target CRS metres
    transects: dict[str, tuple] = {}
    for ft in feats:
        tid = str(ft.get("properties", {}).get("id"))
        coords = ft["geometry"]["coordinates"]
        (x0, y0) = coords[0][:2]
        (x1, y1) = coords[-1][:2]
        if tr is not None:
            (x0, x1), (y0, y1) = tr.transform([x0, x1], [y0, y1])
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        transects[tid] = (x0, y0, dx / length, dy / length)

    # -- chainage time-series for this site --------------------------------- #
    df = pd.read_csv(timeseries_path)
    date_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("dates", "date")), None)
    if date_col is None:  # fall back: first column that parses as datetime
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise SystemExit(f"No dates column found in {timeseries_path}")

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    tcols = [c for c in df.columns if str(c) in transects]
    if not tcols:
        raise SystemExit(
            "No overlap between the time-series transect columns and the "
            "transect ids in the geojson. The site id matches but the per-"
            "transect keys do not -- the two files are probably from different "
            "dataset versions."
        )
    df = df[tcols].apply(pd.to_numeric, errors="coerce")

    # -- interpolate to the target date ------------------------------------- #
    target = pd.Timestamp(year=int(year), month=int(month), day=int(day), tz="UTC")
    df = df.reindex(df.index.union(pd.DatetimeIndex([target]))).sort_index()
    df = df.interpolate(method="time", limit_direction="both")
    if smooth_days and smooth_days > 0:
        df = df.rolling(f"{int(smooth_days)}D", min_periods=1, center=True).mean()
    row = df.loc[target]

    # -- place a point on each transect, in along-shore order --------------- #
    valid = [t for t in tcols if not (isinstance(row[t], float) and math.isnan(row[t]))]
    valid = sorted(valid, key=_natural_key)
    if order.startswith("desc") or order.startswith("rev"):
        valid = valid[::-1]

    xs, ys = [], []
    for tid in valid:
        x0, y0, ux, uy = transects[str(tid)]
        chain = float(row[tid])
        xs.append(x0 + chain * ux)
        ys.append(y0 + chain * uy)
    if not xs:
        raise SystemExit(
            f"No valid shoreline points at {target.date()} for site {site}. "
            f"Try a different --year (data range is ~1984-2025)."
        )
    # Mean seaward direction (every transect unit vector points seaward) -> lets
    # downstream CEM gridding know which side is land (land_dir = -sea_dir).
    svx = sum(transects[str(t)][2] for t in valid) / len(valid)
    svy = sum(transects[str(t)][3] for t in valid) / len(valid)
    snorm = math.hypot(svx, svy) or 1.0
    sea_dir = (svx / snorm, svy / snorm)
    return xs, ys, int(target_epsg), len(xs), sea_dir


def find_zenodo_files(site: str, data_dir: str | None,
                      transects_path: str | None,
                      timeseries_path: str | None):
    """Resolve the transects geojson and the per-site time-series CSV.

    Layout expected under --data-dir (the unzipped Zenodo shoreline_data/):
        <data_dir>/transects.geojson
        <data_dir>/<site>/time_series_tidally_corrected.csv
    Explicit --transects / --timeseries override the auto layout.
    """
    if timeseries_path is None:
        if data_dir is None:
            raise SystemExit("Provide --data-dir (the unzipped Zenodo "
                             "shoreline_data folder) or --timeseries.")
        cand = os.path.join(data_dir, site, "time_series_tidally_corrected.csv")
        if not os.path.exists(cand):
            raise SystemExit(f"Time-series not found: {cand}\n"
                             f"Expected <data_dir>/<site>/time_series_tidally_corrected.csv")
        timeseries_path = cand
    if transects_path is None:
        if data_dir is None:
            raise SystemExit("Provide --transects, or --data-dir so "
                             "transects.geojson can be found.")
        cand = os.path.join(data_dir, "transects.geojson")
        if not os.path.exists(cand):
            raise SystemExit(f"transects.geojson not found in {data_dir}. "
                             f"Pass --transects explicitly.")
        transects_path = cand
    return transects_path, timeseries_path


# A small result bundle for notebook / library use.
ShorelineForCem = namedtuple(
    "ShorelineForCem", "x y epsg n land_dir sea_dir site year")


def shoreline_for_cem(site, year, data_dir=None, transects=None, timeseries=None,
                      epsg=None, src_epsg=4326, month=6, day=30,
                      smooth_days=180, order="ascending"):
    """High-level entry point for notebooks/scripts (zenodo route).

    Resolves the data files, reconstructs the shoreline for <site>/<year>, and
    also returns the LAND direction (opposite the mean seaward transect
    direction) so a CEM grid builder can mask land automatically -- no need to
    eyeball which way the coast faces.

    Returns a ShorelineForCem namedtuple: (x, y, epsg, n, land_dir, sea_dir,
    site, year), where x/y are lists in EPSG metres and land_dir/sea_dir are
    (ux, uy) unit vectors in that same CRS.
    """
    tpath, cpath = find_zenodo_files(site, data_dir, transects, timeseries)
    xs, ys, out_epsg, n, sea_dir = extract_zenodo(
        site=site, year=year, transects_path=tpath, timeseries_path=cpath,
        target_epsg=epsg, src_epsg=src_epsg, month=month, day=day,
        smooth_days=smooth_days, order=order)
    land_dir = (-sea_dir[0], -sea_dir[1]) if sea_dir else None
    return ShorelineForCem(xs, ys, out_epsg, n, land_dir, sea_dir, site, year)


# --------------------------------------------------------------------------- #
# Multi-site stitching (build a LONG shoreline by joining adjacent sites)      #
# --------------------------------------------------------------------------- #
def list_sites(data_dir):
    """Every site id under *data_dir* that has a time-series CSV (i.e. is
    extractable). Sorted by natural order so numbering runs along the coast."""
    out = []
    if data_dir and os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            if os.path.exists(os.path.join(
                    data_dir, name, "time_series_tidally_corrected.csv")):
                out.append(name)
    return sorted(out, key=_natural_key)


def _site_num(site):
    """Split a site id into (prefix, digits) on its trailing number, e.g.
    'usa_NC_0030' -> ('usa_NC_', '0030'). Returns None if it has no number."""
    m = re.match(r"^(.*?)(\d+)$", str(site))
    return (m.group(1), m.group(2)) if m else None


def neighbor_sites(seed, n, data_dir):
    """*seed* plus up to *n* sites on each side, by trailing number, keeping
    only those that actually exist under *data_dir*. Ordered along the coast."""
    parsed = _site_num(seed)
    if parsed is None or n <= 0:
        return [seed]
    prefix, digits = parsed
    width, base = len(digits), int(digits)
    have = set(list_sites(data_dir))
    out = [f"{prefix}{k:0{width}d}" for k in range(base - n, base + n + 1) if k >= 0]
    out = [s for s in out if s in have]
    return out or [seed]


def _common_epsg(sites, transects_path, src_epsg=4326):
    """Pick ONE metric (UTM) EPSG from the centroid of all transects of *sites*,
    so every stitched site is placed in the same CRS."""
    import json
    with open(transects_path) as fh:
        gj = json.load(fh)
    sel = {str(s) for s in sites}
    lons, lats = [], []
    for ft in gj.get("features", []):
        if str(ft.get("properties", {}).get("site_id")) in sel:
            c0 = ft["geometry"]["coordinates"][0]
            lons.append(c0[0]); lats.append(c0[1])
    if not lons:
        return None
    clon, clat = sum(lons) / len(lons), sum(lats) / len(lats)
    if int(src_epsg) != 4326:
        (clon,), (clat,) = reproject([clon], [clat], src_epsg, 4326)
    return utm_epsg(clon, clat)


def _chain_segments(segments):
    """Join a list of (xs, ys) polylines into ONE continuous line.

    Sites come back internally ordered but with no guarantee that site A's tail
    meets site B's head -- a naive concatenation would zig-zag at every join.
    So we (1) use a PCA axis over all points to pick a natural starting end,
    then (2) greedily append whichever remaining segment has an endpoint nearest
    the current tail, reversing it if its *far* end is the nearer one. This is
    robust to site order, per-site direction, gaps, and gentle coast curvature.
    """
    import numpy as np
    segs = [(np.asarray(xs, float), np.asarray(ys, float))
            for xs, ys in segments if len(xs) > 0]
    if not segs:
        return [], []
    if len(segs) == 1:
        return list(segs[0][0]), list(segs[0][1])

    allx = np.concatenate([s[0] for s in segs])
    ally = np.concatenate([s[1] for s in segs])
    pca = np.column_stack([allx - allx.mean(), ally - ally.mean()])
    axis = np.linalg.svd(pca, full_matrices=False)[2][0]
    proj = lambda px, py: px * axis[0] + py * axis[1]

    # start at the segment reaching furthest "back" along the axis, pointed +axis
    i0 = int(np.argmin([min(proj(s[0][0], s[1][0]), proj(s[0][-1], s[1][-1]))
                        for s in segs]))
    xs, ys = segs[i0]
    if proj(xs[0], ys[0]) > proj(xs[-1], ys[-1]):
        xs, ys = xs[::-1], ys[::-1]
    chainx, chainy = list(xs), list(ys)
    used = {i0}

    while len(used) < len(segs):
        tx, ty = chainx[-1], chainy[-1]
        best, rev, bestd = None, False, None
        for j, (xs, ys) in enumerate(segs):
            if j in used:
                continue
            dhead = (xs[0] - tx) ** 2 + (ys[0] - ty) ** 2
            dtail = (xs[-1] - tx) ** 2 + (ys[-1] - ty) ** 2
            d, r = (dhead, False) if dhead <= dtail else (dtail, True)
            if bestd is None or d < bestd:
                bestd, best, rev = d, j, r
        xs, ys = segs[best]
        if rev:
            xs, ys = xs[::-1], ys[::-1]
        chainx += list(xs); chainy += list(ys)
        used.add(best)
    return chainx, chainy


def extract_zenodo_multi(sites, year, transects_path, data_dir,
                         target_epsg=None, src_epsg=4326, month=6, day=30,
                         smooth_days=180, order="ascending"):
    """Reconstruct and stitch several sites into ONE long shoreline.

    Reuses the single-site reconstruction per site (each site has its own
    time-series CSV, but they share one transects.geojson), forces a common CRS,
    then chains the segments head-to-tail. Returns
    (xs, ys, epsg, n, sea_dir, per_site) where per_site is [(site, n_points), ...].
    """
    if not sites:
        raise SystemExit("No sites given to stitch.")
    if target_epsg is None:
        target_epsg = _common_epsg(sites, transects_path, src_epsg)

    segments, sea_acc, per_site = [], [0.0, 0.0], []
    for s in sites:
        _, cpath = find_zenodo_files(s, data_dir, transects_path, None)
        xs, ys, ep, n, sea = extract_zenodo(
            site=s, year=year, transects_path=transects_path,
            timeseries_path=cpath, target_epsg=target_epsg, src_epsg=src_epsg,
            month=month, day=day, smooth_days=smooth_days, order=order)
        target_epsg = ep                       # lock CRS after the first
        segments.append((xs, ys))
        sea_acc[0] += sea[0] * n; sea_acc[1] += sea[1] * n
        per_site.append((s, n))

    xs, ys = _chain_segments(segments)
    snorm = math.hypot(*sea_acc) or 1.0
    sea_dir = (sea_acc[0] / snorm, sea_acc[1] / snorm)   # point-count weighted
    return xs, ys, int(target_epsg), len(xs), sea_dir, per_site


def shoreline_for_cem_multi(sites, year, data_dir=None, transects=None,
                            epsg=None, src_epsg=4326, month=6, day=30,
                            smooth_days=180, order="ascending"):
    """High-level multi-site entry for notebooks (zenodo route).

    Same contract as ``shoreline_for_cem`` but for a LIST of adjacent sites,
    returning one stitched shoreline. ``site`` in the result is '+'.join(sites).
    """
    if isinstance(sites, str):
        sites = [sites]
    tpath, _ = find_zenodo_files(sites[0], data_dir, transects, None)
    xs, ys, out_epsg, n, sea_dir, _per = extract_zenodo_multi(
        sites, year, tpath, data_dir, target_epsg=epsg, src_epsg=src_epsg,
        month=month, day=day, smooth_days=smooth_days, order=order)
    land_dir = (-sea_dir[0], -sea_dir[1]) if sea_dir else None
    return ShorelineForCem(xs, ys, out_epsg, n, land_dir, sea_dir,
                           "+".join(sites), year)


# --------------------------------------------------------------------------- #
# Portable input discovery (so notebooks need no hard-coded absolute paths)    #
# --------------------------------------------------------------------------- #
def _candidate_roots(search_roots=None, max_up: int = 5):
    """cwd plus its parents (or an explicit list), as absolute paths."""
    if search_roots:
        return [os.path.abspath(r) for r in search_roots]
    here = os.path.abspath(os.getcwd())
    roots = [here]
    p = here
    for _ in range(max_up):
        parent = os.path.dirname(p)
        if not parent or parent == p:
            break
        roots.append(parent)
        p = parent
    return roots


def _has_site_folders(d: str) -> bool:
    import glob
    return bool(glob.glob(os.path.join(d, "usa_*")))


def find_site_data(search_roots=None, max_up: int = 5):
    """Locate the folder that directly contains the usa_* site folders.

    Handles the common layouts: shoreline_data/ with the site folders directly
    inside, an extra wrapper folder (e.g. csv_run7/), or a data/ prefix. Returns
    an absolute path or None.
    """
    roots = _candidate_roots(search_roots, max_up)
    rels = ["shoreline_data/csv_run7", "shoreline_data",
            "data/shoreline_data/csv_run7", "data/shoreline_data",
            "csv_run7", "."]
    for base in roots:
        for rel in rels:
            d = os.path.join(base, rel)
            if os.path.isdir(d) and _has_site_folders(d):
                return os.path.abspath(d)
        # unknown-named wrapper folder directly under shoreline_data/
        for sdrel in ("shoreline_data", "data/shoreline_data"):
            sd = os.path.join(base, sdrel)
            if os.path.isdir(sd):
                for name in sorted(os.listdir(sd)):
                    sub = os.path.join(sd, name)
                    if os.path.isdir(sub) and _has_site_folders(sub):
                        return os.path.abspath(sub)
    return None


def find_transects_file(search_roots=None, max_up: int = 5):
    """Locate a transects geojson (US_East_transects.geojson or transects.geojson).
    Returns an absolute path or None."""
    import glob
    roots = _candidate_roots(search_roots, max_up)
    names = ["US_East_transects.geojson", "transects.geojson"]
    rels = [".", "shoreline_data", "data/shoreline_data", "data"]
    for base in roots:
        for rel in rels:
            for nm in names:
                f = os.path.join(base, rel, nm)
                if os.path.isfile(f):
                    return os.path.abspath(f)
    # last resort: any *transects*.geojson under a shoreline_data/ dir
    for base in roots:
        for sdrel in ("shoreline_data", "data/shoreline_data"):
            sd = os.path.join(base, sdrel)
            if os.path.isdir(sd):
                hits = sorted(glob.glob(os.path.join(sd, "*transects*.geojson")))
                if hits:
                    return os.path.abspath(hits[0])
    return None


# --------------------------------------------------------------------------- #
# Route 2 -- digitise one shoreline from imagery with CoastSat                 #
# --------------------------------------------------------------------------- #
def parse_polygon(spec: str):
    """'lon,lat lon,lat ...' -> [[[lon,lat], ...]] (GeoJSON-style ring)."""
    ring = []
    for tok in spec.replace(";", " ").split():
        lon_s, lat_s = tok.split(",")
        ring.append([float(lon_s), float(lat_s)])
    if len(ring) < 3:
        raise SystemExit("--polygon needs at least 3 'lon,lat' points.")
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return [ring]


def extract_imagery(polygon,
                    dates,
                    sat_list,
                    sitename: str,
                    filepath: str,
                    target_epsg: int | None = None,
                    cloud_thresh: float = 0.5,
                    skip_download: bool = False,
                    settings_extra: dict | None = None):
    """
    Download a few images for *polygon* over *dates* and digitise one reference
    shoreline with CoastSat. Returns (xs, ys, out_epsg, n_points) in target_epsg.

    The reference-shoreline step (SDS_preprocess.get_reference_sl) is interactive
    on first run -- a window opens and you click the shoreline. It is cached to
    <sitename>_reference_shoreline.pkl, so re-runs are non-interactive.
    """
    import numpy as np
    try:
        from coastsat import SDS_download, SDS_preprocess, SDS_tools
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise SystemExit(
            "Route 'imagery' needs the CoastSat toolbox importable (e.g. put the "
            "coastsat/ folder from espin-2021/coastal on your PYTHONPATH) and an "
            "authenticated Google Earth Engine session ('earthengine authenticate').\n"
            f"Import failed: {exc}"
        )

    polygon = SDS_tools.smallest_rectangle(polygon)
    if target_epsg is None:
        ring = polygon[0]
        lon = sum(p[0] for p in ring) / len(ring)
        lat = sum(p[1] for p in ring) / len(ring)
        target_epsg = utm_epsg(lon, lat)
    if not is_projected(target_epsg):
        raise SystemExit(f"Output EPSG {target_epsg} is geographic; pass a UTM/metric --epsg.")

    inputs = {"polygon": polygon, "dates": list(dates),
              "sat_list": list(sat_list), "sitename": sitename,
              "filepath": filepath}
    SDS_download.check_images_available(inputs)
    metadata = (SDS_download.get_metadata(inputs) if skip_download
                else SDS_download.retrieve_images(inputs))

    settings = {
        "cloud_thresh": float(cloud_thresh),
        "output_epsg": int(target_epsg),
        "check_detection": True,
        "adjust_detection": False,
        "save_figure": True,
        "min_beach_area": 4500,
        "buffer_size": 150,
        "min_length_sl": 200,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "inputs": inputs,
    }
    if settings_extra:
        settings.update(settings_extra)

    ref = np.asarray(SDS_preprocess.get_reference_sl(metadata, settings))
    if ref.ndim != 2 or ref.shape[1] < 2 or ref.shape[0] < 2:
        raise SystemExit("Reference shoreline came back empty/!2D -- nothing digitised?")
    return ref[:, 0].tolist(), ref[:, 1].tolist(), int(target_epsg), ref.shape[0], None


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coastsat_single_site.py",
        description="Extract one shoreline for one site as a CEM-ready .xy file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="source", required=True,
                           metavar="{zenodo,zenodo-multi,imagery}")

    # shared output options factored into a parent parser
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="output .xy path")
    common.add_argument("--epsg", type=int, default=None,
                        help="output EPSG (projected/metric). Default: auto UTM zone.")
    common.add_argument("--close-ring", action="store_true",
                        help="repeat the first point at the end (closed loop, "
                             "like the Erie spit example). Default: open line.")

    z = sub.add_parser("zenodo", parents=[common],
                       help="reconstruct from the Zenodo US-East-Coast time-series (no GEE)")
    z.add_argument("--site", required=True,
                   help="site id / folder name, e.g. usa_NC0030")
    z.add_argument("--year", type=int, required=True, help="target year (~1984-2025)")
    z.add_argument("--data-dir", default=None,
                   help="unzipped Zenodo shoreline_data/ folder")
    z.add_argument("--transects", default=None, help="override path to transects.geojson")
    z.add_argument("--timeseries", default=None,
                   help="override path to the site's time_series_tidally_corrected.csv")
    z.add_argument("--src-epsg", type=int, default=4326,
                   help="CRS of the transects geojson coordinates (default 4326)")
    z.add_argument("--month", type=int, default=6, help="target month (default 6)")
    z.add_argument("--day", type=int, default=30, help="target day (default 30)")
    z.add_argument("--smooth-days", type=int, default=180,
                   help="centred rolling-mean window in days to denoise (0 = off, default 180)")
    z.add_argument("--order", choices=["ascending", "descending"], default="ascending",
                   help="along-shore vertex order by transect id")

    # ---- multi-site stitching (one LONG .xy from several adjacent sites) ---- #
    zm = sub.add_parser(
        "zenodo-multi",
        help="stitch several adjacent Zenodo sites into one long .xy")
    zm.add_argument("--out", default=None,
                    help="output .xy path (required unless --list-sites)")
    zm.add_argument("--epsg", type=int, default=None,
                    help="output EPSG (metric). Default: one UTM zone for the whole span.")
    zm.add_argument("--close-ring", action="store_true",
                    help="repeat the first point at the end (default: open line)")
    zm.add_argument("--sites", nargs="+", default=None,
                    help="explicit site ids, e.g. usa_NC_0029 usa_NC_0030 usa_NC_0031")
    zm.add_argument("--site", default=None,
                    help="seed site id (combine with --neighbors)")
    zm.add_argument("--neighbors", type=int, default=0,
                    help="with --site: also take N sites on EACH side (e.g. 3 -> 7 total)")
    zm.add_argument("--list-sites", action="store_true",
                    help="just print the extractable site ids under --data-dir and exit")
    zm.add_argument("--year", type=int, default=None, help="target year (~1984-2025)")
    zm.add_argument("--data-dir", default=None, required=True,
                    help="unzipped Zenodo shoreline_data/ folder")
    zm.add_argument("--transects", default=None, help="override path to transects.geojson")
    zm.add_argument("--src-epsg", type=int, default=4326,
                    help="CRS of the transects geojson coordinates (default 4326)")
    zm.add_argument("--month", type=int, default=6, help="target month (default 6)")
    zm.add_argument("--day", type=int, default=30, help="target day (default 30)")
    zm.add_argument("--smooth-days", type=int, default=180,
                    help="centred rolling-mean window in days (0 = off, default 180)")
    zm.add_argument("--order", choices=["ascending", "descending"], default="ascending",
                    help="along-shore vertex order within each site")

    m = sub.add_parser("imagery", parents=[common],
                       help="digitise one shoreline from imagery with CoastSat (needs GEE)")
    m.add_argument("--polygon", required=True,
                   help="ROI as 'lon,lat lon,lat ...' (<100 km^2)")
    m.add_argument("--dates", nargs=2, metavar=("START", "END"), required=True,
                   help="date range, e.g. 1985-04-01 1985-09-01")
    m.add_argument("--sat", nargs="+", default=["L5"],
                   help="satellite missions, e.g. L5 L7 L8 S2 (default L5)")
    m.add_argument("--sitename", required=True, help="site name (also the cache key)")
    m.add_argument("--filepath", default="./data", help="where CoastSat stores data")
    m.add_argument("--cloud-thresh", type=float, default=0.5)
    m.add_argument("--skip-download", action="store_true",
                   help="reuse already-downloaded imagery (get_metadata instead of retrieve)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.source == "zenodo":
        transects_path, timeseries_path = find_zenodo_files(
            args.site, args.data_dir, args.transects, args.timeseries)
        xs, ys, out_epsg, n, _sea = extract_zenodo(
            site=args.site, year=args.year,
            transects_path=transects_path, timeseries_path=timeseries_path,
            target_epsg=args.epsg, src_epsg=args.src_epsg,
            month=args.month, day=args.day, smooth_days=args.smooth_days,
            order=args.order)
    elif args.source == "zenodo-multi":
        if args.list_sites:
            sites = list_sites(args.data_dir)
            print(f"{len(sites)} extractable sites under {args.data_dir}:")
            for s in sites:
                print(" ", s)
            return 0
        if not args.year:
            raise SystemExit("--year is required.")
        if args.sites:
            sites = list(args.sites)
        elif args.site:
            sites = neighbor_sites(args.site, args.neighbors, args.data_dir)
        else:
            raise SystemExit("Provide --sites a b c, or --site SEED [--neighbors N], "
                             "or --list-sites.")
        if not args.out:
            raise SystemExit("--out is required (unless --list-sites).")
        tpath, _ = find_zenodo_files(sites[0], args.data_dir, args.transects, None)
        xs, ys, out_epsg, n, sea_dir, per = extract_zenodo_multi(
            sites, args.year, tpath, args.data_dir, target_epsg=args.epsg,
            src_epsg=args.src_epsg, month=args.month, day=args.day,
            smooth_days=args.smooth_days, order=args.order)
        ld = (-sea_dir[0], -sea_dir[1])
        written = write_xy(xs, ys, args.out, close_ring=args.close_ring,
                           header=[f"land_dir {ld[0]:.4f} {ld[1]:.4f}",
                                   f"epsg {out_epsg}",
                                   f"sites {'+'.join(s for s, _ in per)}",
                                   f"year {args.year}"])
        try:
            import numpy as _np
            length_km = float(_np.hypot(_np.diff(xs), _np.diff(ys)).sum()) / 1000.0
        except Exception:
            length_km = float("nan")
        print(f"Stitched {len(sites)} sites -> {args.out}")
        print("  sites:", ", ".join(f"{s}({c})" for s, c in per))
        print(f"  {written} points, EPSG:{out_epsg}, length ~{length_km:.1f} km, "
              f"land_dir~=({ld[0]:.2f}, {ld[1]:.2f})")
        print(f"  -> in cem_run_xy set XY_PATH=\"{os.path.basename(args.out)}\", "
              f"LAND_DIR=({round(ld[0])}, {round(ld[1])})")
        return 0
    else:  # imagery
        xs, ys, out_epsg, n, _sea = extract_imagery(
            polygon=parse_polygon(args.polygon), dates=args.dates,
            sat_list=args.sat, sitename=args.sitename, filepath=args.filepath,
            target_epsg=args.epsg, cloud_thresh=args.cloud_thresh,
            skip_download=args.skip_download)

    hdr = [f"epsg {out_epsg}"]
    if _sea is not None:
        hdr.insert(0, f"land_dir {-_sea[0]:.4f} {-_sea[1]:.4f}")
    written = write_xy(xs, ys, args.out, close_ring=args.close_ring, header=hdr)
    print(f"Wrote {written} points to {args.out} (EPSG:{out_epsg}, "
          f"{'closed ring' if args.close_ring else 'open line'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
