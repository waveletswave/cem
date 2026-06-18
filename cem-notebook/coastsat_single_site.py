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
             precision: int = 6) -> int:
    """Write 'x y' rows (projected metres) to *path*. Returns point count."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    if not xs:
        raise ValueError("no points to write")
    if close_ring and (xs[0], ys[0]) != (xs[-1], ys[-1]):
        xs.append(xs[0])
        ys.append(ys[0])
    fmt = "%.{}f".format(int(precision))
    with open(path, "w") as fh:
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
                           metavar="{zenodo,imagery}")

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
    else:  # imagery
        xs, ys, out_epsg, n, _sea = extract_imagery(
            polygon=parse_polygon(args.polygon), dates=args.dates,
            sat_list=args.sat, sitename=args.sitename, filepath=args.filepath,
            target_epsg=args.epsg, cloud_thresh=args.cloud_thresh,
            skip_download=args.skip_download)

    written = write_xy(xs, ys, args.out, close_ring=args.close_ring)
    print(f"Wrote {written} points to {args.out} (EPSG:{out_epsg}, "
          f"{'closed ring' if args.close_ring else 'open line'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
