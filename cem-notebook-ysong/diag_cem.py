#!/usr/bin/env python3
"""
diag_cem.py  --  reproduce the CEM run in a TERMINAL so the real error shows.

The notebook hides C-level crashes (you only get "kernel died"). Running the
same steps as a plain script prints whatever the kernel was hiding -- e.g.
`xtest is uninitialized!` (the v0 shadow bug) or a segfault -- and shows exactly
which step dies.

Run it in a Terminal (not the notebook):

    python diag_cem.py                         # default: usa_NC_0030, 2020, aligned, 10 steps
    python diag_cem.py --site usa_NC_0018      # try a different site
    python diag_cem.py --no-align              # without coast rotation
    python diag_cem.py --xy Erie_spit_ESPIn.xy --land-dir 0 1   # test a raw .xy (e.g. Erie)
    python diag_cem.py --steps 5

Then paste the FULL terminal output back -- especially the last lines before any
crash.
"""
import argparse
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np


def find_extractor_dir():
    p = Path(os.getcwd()).resolve(); bases = [p]
    for _ in range(5):
        if p.parent == p:
            break
        p = p.parent; bases.append(p)
    for b in bases:
        if (b / "coastsat_single_site.py").is_file():
            return str(b)
    for b in bases:
        hits = sorted(glob.glob(str(b / "*" / "coastsat_single_site.py")))
        if hits:
            return str(Path(hits[0]).parent)
    return None


def align(x, y, land_dir):
    # rotate coast to ~horizontal AND put LAND at LOW cross-shore rows (CEM puts
    # land at row 0 / ocean at high rows; backwards -> the x>= crash).
    x = np.asarray(x, float); y = np.asarray(y, float)
    cx, cy = x.mean(), y.mean()
    X = np.column_stack([x - cx, y - cy])
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    th = math.atan2(vt[0, 1], vt[0, 0])
    c, s = math.cos(-th), math.sin(-th)
    XY = (np.array([[c, -s], [s, c]]) @ X.T).T
    xr, yr = XY[:, 0] + cx, XY[:, 1] + cy
    ld = np.array([[c, -s], [s, c]]) @ np.asarray(land_dir, float)
    if ld[1] >= 0:
        xr = 2 * xr.mean() - xr          # 180-deg rotation -> land at LOW y, no mirror
        yr = 2 * yr.mean() - yr
    return xr, yr, (0.0, -1.0), math.degrees(th)


def build_grid(x, y, dx, dy, land_dir, pad_x=0, pad_y=30):
    from matplotlib import path
    from scipy.spatial import cKDTree
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = 0.1; rng = 100000
    x0 = int(np.floor(min(x)/dx)*dx) - pad_x*dx
    y0 = int(np.floor(min(y)/dy)*dy) - pad_y*dy
    x1 = int(np.ceil(max(x)/dx)*dx) + pad_x*dx
    y1 = int(np.ceil(max(y)/dy)*dy) + pad_y*dy
    xg, yg = np.meshgrid(np.arange(x0, x1, dx), np.arange(y0, y1, dy),
                         sparse=False, indexing="ij")
    tree = cKDTree(np.column_stack([x, y]))
    nn, _ = tree.query(np.column_stack([xg.ravel(), yg.ravel()]), k=1,
                       distance_upper_bound=rng)
    nn[np.isinf(nn)] = np.sqrt(1.e10)
    dist = nn.reshape(xg.shape)
    zg = -A * dist**(2/3)                                    # Dean depth (water < 0)
    # column-wise SEAWARD-ENVELOPE land fill: single-valued by construction and robust
    # to a folding/cuspate coast (no polygon to self-intersect). Kept in sync with the
    # notebook's shorelinetogrid_open. land_dir[1]'s sign picks the land (cross-shore) end.
    land_val = float(np.min(A * dist**(2/3))) + 1
    land_low = np.asarray(land_dir, float)[1] < 0
    band = 1.5 * max(dx, dy)
    for i in range(xg.shape[0]):
        near = np.where(dist[i, :] < band)[0]
        if near.size == 0:
            near = np.array([int(np.argmin(dist[i, :]))])
        if land_low:
            zg[i, :near.max() + 1] = land_val
        else:
            zg[i, near.min():] = land_val
    return zg * -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="usa_NC_0030")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--xy", default=None, help="load a raw .xy instead of generating")
    ap.add_argument("--land-dir", nargs=2, type=float, default=None,
                    metavar=("UX", "UY"), help="land direction for --xy mode")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dx", type=int, default=100)
    args = ap.parse_args()

    sys.path.insert(0, find_extractor_dir() or ".")
    import coastsat_single_site as css

    # --- shoreline ---
    if args.xy:
        data = np.loadtxt(args.xy)
        x, y = data[:, 0], data[:, 1]
        land_dir = tuple(args.land_dir) if args.land_dir else (0.0, 1.0)
        print(f"loaded {len(x)} pts from {args.xy}", flush=True)
    else:
        DATA_DIR = css.find_site_data(); TRANSECTS = css.find_transects_file()
        print("DATA_DIR:", DATA_DIR, "\nTRANSECTS:", TRANSECTS, flush=True)
        r = css.shoreline_for_cem(args.site, args.year, data_dir=DATA_DIR,
                                  transects=TRANSECTS)
        x, y = np.asarray(r.x), np.asarray(r.y)
        land_dir = r.land_dir
        print(f"{args.site} {args.year}: {r.n} pts, land_dir={tuple(round(v,2) for v in land_dir)}",
              flush=True)

    if not args.no_align:
        x, y, land_dir, ang = align(x, y, land_dir)
        print(f"aligned coast (rotated {ang:.1f} deg), land_dir -> {land_dir}", flush=True)

    # --- grid + sanity ---
    zg = build_grid(x, y, args.dx, args.dx, land_dir)
    domain = -1.0 * zg.T
    nr, nc = domain.shape
    print(f"\nDOMAIN  rows x cols = {nr} x {nc}  (cells: {domain.size})", flush=True)
    print(f"  land cells: {(domain>0).sum()} ({100*(domain>0).mean():.0f}%) | "
          f"sea cells: {(domain<0).sum()}", flush=True)
    print(f"  domain min/max: {domain.min():.2f} / {domain.max():.2f} | "
          f"NaN: {np.isnan(domain).any()} | inf: {np.isinf(domain).any()}", flush=True)

    # --- CEM ---
    try:
        from pymt.models import Cem, Waves
    except Exception as e:
        print("\nCannot import pymt CEM here:", e, flush=True)
        print("(Run this on the OpenEarthscape hub / your CEM env.)", flush=True)
        return

    waves = Waves(); cem = Cem()
    args_setup = cem.setup(number_of_rows=nr, number_of_cols=nc,
                           grid_spacing=float(args.dx))
    waves.initialize(*waves.setup()); cem.initialize(*args_setup)
    for k, v in [("sea_surface_water_wave__height", 1.5),
                 ("sea_surface_water_wave__period", 7.0)]:
        waves.set_value(k, v); cem.set_value(k, v)
    waves.set_value("sea_shoreline_wave~incoming~deepwater__ashton_et_al_approach_angle_highness_parameter", 0.5)
    waves.set_value("sea_shoreline_wave~incoming~deepwater__ashton_et_al_approach_angle_asymmetry_parameter", 0.5)
    cem.set_value("land_surface__elevation", domain.flatten())
    print("\nCEM initialized; stepping (watch for a C error / segfault below)...", flush=True)

    alpha = "sea_surface_water_wave__azimuth_angle_of_opposite_of_phase_velocity"
    qs = np.zeros((nr, nc)); qs[0, nc // 2] = 500.0
    for t in range(args.steps):
        waves.update()
        cem.set_value(alpha, waves.get_value(alpha))
        cem.set_value("land_surface_water_sediment~bedload__mass_flow_rate", qs)
        cem.update()
        print(f"  step {t} OK", flush=True)
    print(f"\nDONE: {args.steps} steps without crashing.", flush=True)


if __name__ == "__main__":
    main()
