#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import argparse
import warnings
from pathlib import Path
# Helpers for animations
import warnings
import matplotlib.animation as animation  # safe in headless when using Agg

def _parse_clim(s):
    if not s:
        return None
    try:
        a, b = s.split(",")
        return (float(a), float(b))
    except Exception:
        raise argparse.ArgumentTypeError(f"Bad clim '{s}'. Use 'vmin,vmax'.")

def _fmt_sec(sec):
    import datetime
    try:
        return str(datetime.timedelta(seconds=float(sec)))
    except Exception:
        return str(sec)

# Headless safeguard before importing pyplot
HEADLESS = os.environ.get("HEADLESS", "1") == "1" or not os.environ.get("DISPLAY")
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    from netCDF4 import Dataset
except Exception as e:
    print("ERROR: netCDF4 is required to read .sww files. conda install -c conda-forge netcdf4")
    raise

try:
    import pyproj
except Exception:
    pyproj = None

from netCDF4 import Dataset
import numpy as np

def load_sww(sww_path):
    ds = Dataset(sww_path, 'r')  # don't leave the 'with' until arrays are copied
    try:
        # Required vars in ANUGA .sww (names may vary slightly)
        x  = ds.variables['x'][:].astype(np.float64).copy()
        y  = ds.variables['y'][:].astype(np.float64).copy()
        t  = ds.variables['time'][:].astype(np.float64).copy()

        # Tri connectivity may be named 'volumes' or 'triangles'
        if 'volumes' in ds.variables:
            tris = ds.variables['volumes'][:].astype(np.int64).copy()
        else:
            tris = ds.variables['triangles'][:].astype(np.int64).copy()

        # Elevation/stage (shape usually (Nt, N) or (N,) depending on writer)
        elev = ds.variables['elevation'][:].astype(np.float64).copy()
        stage = ds.variables['stage'][:].astype(np.float64).copy()

        # Optional velocities, depth, etc.
        ux = ds.variables['xmomentum'][:].astype(np.float64).copy() if 'xmomentum' in ds.variables else None
        uy = ds.variables['ymomentum'][:].astype(np.float64).copy() if 'ymomentum' in ds.variables else None

        return {
            "x": x, "y": y, "time": t, "tris": tris,
            "elevation": elev, "stage": stage,
            "xmomentum": ux, "ymomentum": uy,
            "attrs": dict(ds.__dict__),
        }
    finally:
        ds.close()

def compute_depth(stage, elev):
    depth = stage - elev[None, :]
    depth[depth < 0] = 0.0
    return depth

def compute_velocity(umom, vmom, depth):
    if umom is None or vmom is None:
        return None
    # velocity = momentum / depth (avoid division by 0)
    d = depth.copy()
    eps = 1e-12
    d[d < eps] = np.nan
    u = umom / d
    v = vmom / d
    speed = np.sqrt(u*u + v*v)
    # replace NaNs (dry) with 0
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    return speed

def save_summary(meta, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(meta, indent=2))

import numpy as np
import matplotlib.tri as mtri

import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

def plot_field_triangulation(x, y, tri, field, title, outpng,
                             cmap="viridis", vmin=None, vmax=None, dpi=150,
                             reduce="max"):  # reduce over time: "max" | "min" | "mean" | None
    triang = mtri.Triangulation(x, y, tri)

    F = np.asarray(field)
    # If time x nodes/triangles, reduce along time
    if F.ndim == 2:
        if reduce == "max":
            F = np.nanmax(F, axis=0)
        elif reduce == "min":
            F = np.nanmin(F, axis=0)
        elif reduce == "mean":
            F = np.nanmean(F, axis=0)
        else:
            raise ValueError("field is 2D; set reduce to 'max'|'min'|'mean'")

    # Clean NaNs/Infs
    F = F.astype(float, copy=False)
    F[~np.isfinite(F)] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    if F.size == x.size:
        # Node-based
        pc = ax.tripcolor(triang, F, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    elif F.size == tri.shape[0]:
        # Triangle-based
        pc = ax.tripcolor(triang, facecolors=F, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"Field length {F.size} doesn't match nodes ({x.size}) or triangles ({tri.shape[0]}).")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(pc, ax=ax, label=title)
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

def nearest_node(x, y, lon, lat, mesh_crs="EPSG:26914"):
    """Return nearest node index to given lon/lat (or x/y if mesh_crs is None)."""
    if pyproj is None:
        raise RuntimeError("pyproj required for coordinate transforms")
    # Assume input is lon/lat WGS84
    wgs84 = pyproj.CRS.from_epsg(4326)
    dst = pyproj.CRS.from_user_input(mesh_crs)
    tfm = pyproj.Transformer.from_crs(wgs84, dst, always_xy=True).transform
    X, Y = tfm(lon, lat)
    dx = x - X
    dy = y - Y
    idx = np.argmin(dx*dx + dy*dy)
    return idx, X, Y

def animate_water_level_video(
    x, y, tri, elev, stage, t, out_mp4,
    fps=10, every=1, dpi=150, clim=None, title_prefix="Water level"
  ):
    """
    Stage animation with flat-shaded triangle faces.
    Keeps color limits fixed across frames for visual consistency.
    """
    import matplotlib.tri as mtri

    T, N = stage.shape
    frames = np.arange(0, T, max(1, int(every)))
    if frames.size == 0:
        frames = np.array([0])

    triang = mtri.Triangulation(x, y, tri)

    # Initial frame -> face values
    st0 = stage[frames[0], :]
    depth0 = st0 - elev
    st0 = np.where(depth0 <= 0.05, np.nan, st0)
    st0_faces = np.nanmean(st0[tri], axis=1)

    # Color limits
    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin = float(np.nanquantile(st0_faces, 0.05))
        vmax = float(np.nanquantile(st0_faces, 0.95))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
            mid = 0.5 * (vmin + vmax)
            vmin, vmax = mid - 0.05, mid + 0.05

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("slategrey")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title_prefix} at t={_fmt_sec(t[frames[0]])}")
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    pc = ax.tripcolor(triang, facecolors=st0_faces, cmap="turbo",
                      vmin=vmin, vmax=vmax, shading="flat")
    fig.colorbar(pc, ax=ax, label="Water level [m]")

    def _update(k):
        fi = frames[k]
        st = stage[fi, :]
        st = np.where((st - elev) <= 0.05, np.nan, st)
        faces = np.nanmean(st[tri], axis=1)
        pc.set_array(faces)
        ax.set_title(f"{title_prefix} at t={_fmt_sec(t[fi])}")
        return (pc,)

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), blit=False)

    # Prefer ffmpeg; fallback to Pillow
    try:
        ani.save(out_mp4, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception as e:
        warnings.warn(f"ffmpeg unavailable ({e}); falling back to Pillow writer (slower/larger).")
        ani.save(out_mp4, writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return out_mp4


def animate_depth_video(
    x, y, tri, elev, stage, t, out_mp4,
    fps=10, every=1, dpi=150, clim=None, gamma=0.6, dry_thresh=0.01,
    title_prefix="Water depth"
):
    """
    Depth animation: depth = max(stage - elevation, 0).
    Uses PowerNorm to enhance shallow areas.
    """
    import matplotlib.tri as mtri
    from matplotlib import colors

    T, N = stage.shape
    frames = np.arange(0, T, max(1, int(every)))
    if frames.size == 0:
        frames = np.array([0])

    triang = mtri.Triangulation(x, y, tri)

    st0 = stage[frames[0], :]
    d0 = np.clip(st0 - elev, 0.0, None)
    d0[d0 <= dry_thresh] = np.nan
    d0_faces = np.nanmean(d0[tri], axis=1)

    # Color limits
    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        if np.isfinite(d0_faces).any():
            vmin = float(np.nanquantile(d0_faces, 0.05))
            vmax = float(np.nanquantile(d0_faces, 0.95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
                vmin, vmax = 0.0, max(1.0, float(np.nanmax(d0_faces)))
        else:
            vmin, vmax = 0.0, 1.0

    norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("slategrey")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title_prefix} at t={_fmt_sec(t[frames[0]])}")
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    pc = ax.tripcolor(triang, facecolors=d0_faces, cmap="Blues",
                      shading="flat", norm=norm)
    fig.colorbar(pc, ax=ax, label="Water depth [m]")

    def _update(k):
        fi = frames[k]
        d = np.clip(stage[fi, :] - elev, 0.0, None)
        d[d <= dry_thresh] = np.nan
        faces = np.nanmean(d[tri], axis=1)
        pc.set_array(faces)
        ax.set_title(f"{title_prefix} at t={_fmt_sec(t[fi])}")
        return (pc,)

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), blit=False)

    try:
        ani.save(out_mp4, writer="ffmpeg", fps=fps, dpi=dpi)
    except Exception as e:
        warnings.warn(f"ffmpeg unavailable ({e}); falling back to Pillow writer (slower/larger).")
        ani.save(out_mp4, writer=animation.PillowWriter(fps=fps))

    plt.close(fig)
    return out_mp4
    
def main():
    
    p = argparse.ArgumentParser(description="ANUGA .sww output analysis")
    p.add_argument("--make-video", choices=["none", "both", "stage", "depth"], default="none",
               help="Write MP4 animations (water level, depth, or both)")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for MP4")
    p.add_argument("--every", type=int, default=1, help="Use every Nth time step")
    p.add_argument("--clim-stage", default=None,
               help="Stage color limits as 'vmin,vmax' (e.g., 410.7,411.3)")
    p.add_argument("--clim-depth", default=None,
               help="Depth color limits as 'vmin,vmax' (e.g., 0,2)")
    p.add_argument("--gamma", type=float, default=0.6, help="Gamma for depth PowerNorm")
    p.add_argument("--dry-thresh", type=float, default=0.01, help="Depth ≤ this is dry (NaN)")

    p.add_argument("--sww", required=True, help="Path to .sww file")
    p.add_argument("--out", default="./analysis", help="Output directory")
    p.add_argument("--frames", type=int, default=6, help="How many sample frames to export (evenly spaced)")
    p.add_argument("--mesh-crs", default="EPSG:26914", help="Mesh CRS for sampling lon/lat points")
    p.add_argument("--sample", action="append", default=[],
                   help='Sample lon,lat[:name] points (e.g., "50.993889,-101.287222:Russell")')
    args = p.parse_args()
    clim_stage = _parse_clim(args.clim_stage)
    clim_depth = _parse_clim(args.clim_depth)

    sww_path = Path(args.sww)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {sww_path}")
    
    dat = load_sww(sww_path)
    x   = dat.get("x")
    y   = dat.get("y")
    tri = dat.get("tri", dat.get("tris"))
    t   = dat.get("t",   dat.get("time"))
    elev = dat["elevation"]; stage = dat["stage"]; um = dat["xmomentum"]; vm = dat["ymomentum"]

    # Basic meta
    meta = {
        "sww": str(sww_path),
        "n_nodes": int(x.shape[0]),
        "n_tris": int(tri.shape[0]),
        "n_steps": int(stage.shape[0]),
        "t_min": float(t.min() if t.size else 0.0),
        "t_max": float(t.max() if t.size else 0.0),
        "has_momentum": bool(um is not None and vm is not None),
    }
    save_summary(meta, outdir)
    print("Summary:", json.dumps(meta, indent=2))

    depth = compute_depth(stage, elev)                           # (T,N)
    max_depth = depth.max(axis=0)                                # (N,)
    plot_field_triangulation(x, y, tri, max_depth, "Max Depth (m)",
                             outdir / "max_depth.png", cmap="cmo.haline" if "cmo" in plt.colormaps() else "viridis")

    # Velocity if present
    if um is not None and vm is not None:
        speed = compute_velocity(um, vm, depth)                  # (T,N)
        max_speed = speed.max(axis=0)
        plot_field_triangulation(x, y, tri, max_speed, "Max Speed (m/s)",
                                 outdir / "max_speed.png", cmap="cmo.speed" if "cmo" in plt.colormaps() else "plasma")
    else:
        print("Momentum not found; skipping velocity plots.")

    # Export a few frames
    nT = stage.shape[0]
    n_frames = max(1, min(args.frames, nT))
    idxs = np.unique(np.linspace(0, nT-1, n_frames, dtype=int))
    for k, ti in enumerate(idxs):
        d = depth[ti, :]
        plot_field_triangulation(x, y, tri, d, f"Depth at t={t[ti]:.1f} s",
                                 outdir / f"depth_t{ti:04d}.png", cmap="cmo.haline" if "cmo" in plt.colormaps() else "viridis")

    # Optional sampling at lon/lat points
    if args.sample:
        import csv
        spath = outdir / "samples_timeseries.csv"
        with open(spath, "w", newline="") as f:
            w = csv.writer(f)
            header = ["name", "lon", "lat", "X", "Y"] + [f"t={float(tt):.1f}s" for tt in t]
            w.writerow(header)
            for s in args.sample:
                try:
                    coord, *name_part = s.split(":")
                    name = name_part[0] if name_part else "pt"
                    lon, lat = map(float, coord.split(","))
                except Exception:
                    print(f"Skipping malformed --sample '{s}' (use 'lon,lat[:name]')")
                    continue
                idx, Xp, Yp = nearest_node(x, y, lon, lat, mesh_crs=args.mesh_crs)
                dts = depth[:, idx]  # depth timeseries (T,)
                w.writerow([name, lon, lat, Xp, Yp] + [float(val) for val in dts])
        print(f"Wrote samples: {spath}")
    
           # ---- MP4 animations (optional) ----
    if args.make_video in ("both", "stage"):
     out_mp4 = outdir / "water_level.mp4"
     print(f"Making stage video → {out_mp4}")
     animate_water_level_video(
        x=x, y=y, tri=tri, elev=elev, stage=stage, t=t,
        out_mp4=str(out_mp4),
        fps=args.fps, every=args.every, dpi=150,
        clim=clim_stage, title_prefix="Water level"
     )

    if args.make_video in ("both", "depth"):
     out_mp4 = outdir / "water_depth.mp4"
     print(f"Making depth video → {out_mp4}")
     animate_depth_video(
        x=x, y=y, tri=tri, elev=elev, stage=stage, t=t,
        out_mp4=str(out_mp4),
        fps=args.fps, every=args.every, dpi=150,
        clim=clim_depth, gamma=args.gamma, dry_thresh=args.dry_thresh,
        title_prefix="Water depth"
     )
       
    print("Done. Outputs in:", outdir)

if __name__ == "__main__":
    main()