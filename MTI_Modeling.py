#!/usr/bin/env python
# coding: utf-8

import os

#headless mode before importing pyplot
HEADLESS = os.environ.get("HEADLESS", "1") == "1" or not os.environ.get("DISPLAY")
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

import sys
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
if HEADLESS:
    plt.show = lambda *a, **kw: None
    
import cmocean
import shutil
from pathlib import Path
from tqdm import tqdm
from utils import data_processing_tools as dpt

import anuga
MYID, NUMPROCS = anuga.myid, anuga.numprocs
ISROOT = (MYID == 0)

def finish_plot(path=None, dpi=150):
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    elif not HEADLESS:
        plt.show()
    plt.close()

workshop_dir = os.getcwd()
#workshop_dir = '/path/to/1_HydrodynamicModeling_ANUGA'
data_dir = os.path.join(workshop_dir, 'data')
model_inputs_dir = os.path.join(workshop_dir, 'model_inputs')
model_outputs_dir = os.path.join(workshop_dir, 'model_outputs')
if 'google.colab' in sys.modules:
    data_dir = os.path.join(data_dir, 'collab')
    model_inputs_dir = os.path.join(model_inputs_dir, 'collab')
    model_outputs_dir = os.path.join(model_outputs_dir, 'collab')
model_visuals_dir = os.path.join(workshop_dir, 'visuals')
model_validation_dir = os.path.join(workshop_dir, 'validation')

for d in [model_inputs_dir, model_outputs_dir, model_visuals_dir, model_validation_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)
        


# In[5]:


## Import the datasets
# DEM
f_DEM_tif = os.path.join(data_dir, 'DEM_MTI_PART.tif')
if 'google.colab' not in sys.modules:
    DEM_src = rio.open(f_DEM_tif)
    DEM = DEM_src.read(1)
    resolution = DEM_src.res[0]

extent = [DEM_src.bounds.left, DEM_src.bounds.right,
          DEM_src.bounds.bottom, DEM_src.bounds.top]
# Background imagery
f_bg_img_tif = os.path.join(data_dir, 'Landsat_B6.tif')
bg_img_src = rio.open(f_bg_img_tif)
bg_img = np.stack([bg_img_src.read(i) for i in range(1, bg_img_src.count+1)], axis=2)
bg_img_extent = [bg_img_src.bounds.left, bg_img_src.bounds.right,
                 bg_img_src.bounds.bottom, bg_img_src.bounds.top]


# In[6]:


# Read DEM with mask (nodata auto-masked)
DEM = DEM_src.read(1, masked=True).astype("float32")
nd  = DEM_src.nodata  # e.g., -32767 or -9999

# (Optional) also mask any clearly impossible elevations to be safe
import numpy as np
DEM = np.ma.masked_where((~DEM.mask) & (DEM < 0), DEM)  # adjust threshold to your region

# Nice plotting: masked cells transparent
import cmocean
cmap = cmocean.cm.topo.copy()
cmap.set_bad(alpha=0.0)

fig, ax = plt.subplots(1,1, figsize=(8,4), dpi=200)
ax.imshow(bg_img, extent=bg_img_extent)
im1 = ax.imshow(DEM, cmap=cmap, extent=extent)  # DEM is a masked array now
plt.colorbar(im1)
ax.set_title('DEM (nodata masked)')
#plt.savefig(os.path.join(model_visuals_dir, 'DEM.png'))
#plt.show()
if ISROOT:  # only rank 0 writes files
    finish_plot(os.path.join(model_visuals_dir, "DEM.png"))
# sanity check (ignores masked cells)
print("Range:", np.nanmin(DEM), np.nanmax(DEM), "| nodata:", nd)


# In[7]:


# Output .asc file name
f_edited_DEM_asc = os.path.join(
    data_dir, os.path.basename(f_DEM_tif).replace('.tif', '_edited.asc')
)
if os.path.exists(f_edited_DEM_asc):
    os.remove(f_edited_DEM_asc)

with rio.open(f_DEM_tif) as src:
    # read with mask so nodata becomes masked automatically
    arr = src.read(1, masked=True).astype('float32')
    src_nd = src.nodata

# fill masked (nodata) with -9999; also catch any stray -32767
A = np.ma.filled(arr, fill_value=-9999.0)
A[A == -32767] = -9999.0

# build a clean AAIGrid profile
profile = {
    'driver': 'AAIGrid',
    'dtype': 'float32',          # avoid int16 sentinels
    'width': A.shape[1],
    'height': A.shape[0],
    'count': 1,
    'crs': src.crs,              # reuse CRS from the GeoTIFF
    'transform': src.transform,  # and geotransform
    'nodata': -9999.0
}

# write ASCII grid
with rio.open(f_edited_DEM_asc, 'w', **profile) as dst:
    dst.write(A, 1)

# quick sanity check
print("Wrote:", f_edited_DEM_asc)
print("Min/Max (ignoring -9999):",
      np.nanmin(np.where(A==-9999, np.nan, A)),
      np.nanmax(np.where(A==-9999, np.nan, A)))
print("Counts -> -32767:", np.sum(A == -32767), " -9999:", np.sum(A == -9999))


# In[11]:


# --- Load upstream boundary line ---
f_US_BC = os.path.join(data_dir, 'ShMouth_US_BC.shp')
us_gdf = gpd.read_file(f_US_BC)

# --- Load downstream boundary line ---
f_DS_BC = os.path.join(data_dir, 'Russel_DS_Boundary.shp')
ds_gdf = gpd.read_file(f_DS_BC)


# --- Reproject everything to match DEM CRS ---
if us_gdf.crs != DEM_src.crs:
    us_gdf = us_gdf.to_crs(DEM_src.crs)

if ds_gdf.crs != DEM_src.crs:
    ds_gdf = ds_gdf.to_crs(DEM_src.crs)


# --- Extract coordinates from line shapefiles ---
us_bc_line = np.array(us_gdf.geometry.iloc[0].coords)
ds_bc_line = np.array(ds_gdf.geometry.iloc[0].coords)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

# Plot DEM
im = ax.imshow(DEM, cmap='cmo.topo', extent=extent, zorder=1)

# Overlay upstream and downstream boundary lines
ax.plot(us_bc_line[:, 0], us_bc_line[:, 1], color='red', linewidth=2, label='Upstream BC Line', zorder=2)
ax.plot(ds_bc_line[:, 0], ds_bc_line[:, 1], color='blue', linewidth=2, label='Downstream BC Line', zorder=2)


# Formatting
ax.set_title('DEM with Boundary Lines')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_aspect('equal')
ax.legend(loc='lower right')

# Colorbar
plt.colorbar(im, ax=ax, label='Elevation (m)')
plt.tight_layout()
if ISROOT:  # only rank 0 writes files
    finish_plot(os.path.join(model_visuals_dir, "DEM_with_BC_Lines.png"))

#plt.savefig(os.path.join(model_visuals_dir, 'DEM_with_BC_Lines.png'))
#plt.show()


# In[12]:


from dataretrieval import nwis
import noaa_coops as noaa
from tqdm import notebook 
import matplotlib
import anuga
from anuga import Set_stage, Reflective_boundary
from anuga.structures.inlet_operator import Inlet_operator

#from utils.anuga_tools.baptist_operator import Baptist_operator
from utils.anuga_tools import anuga_tools as at
from utils import data_processing_tools as dpt

# Define the path to scripts and data
workshop_dir = os.getcwd()
# # Alternatively:
# workshop_dir = '/path/to/1_HydrodynamicModeling_ANUGA'
data_dir = os.path.join(workshop_dir, 'data')
model_inputs_dir = os.path.join(workshop_dir, 'model_inputs')
model_outputs_dir = os.path.join(workshop_dir, 'model_outputs')
if 'google.colab' in sys.modules:
    data_dir = os.path.join(data_dir, 'collab')
    model_inputs_dir = os.path.join(model_inputs_dir, 'collab')
    model_outputs_dir = os.path.join(model_outputs_dir, 'collab')
model_visuals_dir = os.path.join(workshop_dir, 'visuals')
model_validation_dir = os.path.join(workshop_dir, 'validation')

for d in [model_inputs_dir, model_outputs_dir, model_visuals_dir, model_validation_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)
        
# Install custom anuga modules
#f_py_install = os.path.join(workshop_dir, 'utils/anuga_tools/install.py')
#get_ipython().system('python $f_py_install')

import subprocess, sys
#subprocess.check_call([sys.executable, f_py_install])

# Check if the user operating system is windows (useful )
is_windows = sys.platform.startswith('win')


# In[18]:


import math
import fiona
from shapely.geometry import shape
import rasterio

mesh_tri_shp = os.path.join(data_dir, "DEM_MTI_PART_USM.shp")  # your mesh triangles shapefile
pts_path  = os.path.join(data_dir, "mesh_mti_shp_pts.npy")
tris_path = os.path.join(data_dir, "mesh_mti_shp_tris.npy")

# Tolerance for merging nearly-identical node coordinates (in CRS units; meters for UTM)
dedup_tol = 1e-6  # try 1e-6 to 1e-3 depending on your CRS precision
# ------------------------------------------------

def decimals_from_tol(tol: float) -> int:
    if tol <= 0:
        return 8
    # e.g., tol=1e-6 -> 6 decimals, tol=0.01 -> 2 decimals
    return max(0, int(round(-math.log10(tol))))

def unique_indexer(decimals: int):
    """Return a function that maps (x,y) to a stable key with rounding."""
    def keyfun(x, y):
        return (round(float(x), decimals), round(float(y), decimals))
    return keyfun

def add_point_get_index(x, y, keyfun, node_index, nodes):
    k = keyfun(x, y)
    idx = node_index.get(k)
    if idx is None:
        idx = len(nodes)
        nodes.append([float(k[0]), float(k[1])])
        node_index[k] = idx
    return idx

def extract_triangles_from_geometry(geom):
    tris = []
    if geom.geom_type == "Polygon":
        rings = [geom.exterior]
    elif geom.geom_type == "MultiPolygon":
        rings = [g.exterior for g in geom.geoms]
    else:
        return tris

    for ring in rings:
        coords = list(ring.coords)
        # drop closing duplicate if present
        if len(coords) >= 2 and (coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]):
            coords = coords[:-1]

        # some writers store triangles with 3 distinct points; others might repeat one
        # keep first 3 unique points in order
        uniq = []
        seen = set()
        for (x, y) in coords:
            p = (x, y)
            if p not in seen:
                uniq.append(p)
                seen.add(p)
            if len(uniq) == 3:
                break

        if len(uniq) == 3:
            tris.append(uniq)
        # else: not a triangle; skip silently
    return tris

decimals = decimals_from_tol(dedup_tol)
keyfun   = unique_indexer(decimals)

nodes = []             
node_index = {}          
tri_indices = []          # [[i,j,k], ...]

with fiona.open(mesh_tri_shp) as src:
    # Optional: sanity on CRS
    crs_info = src.crs_wkt or src.crs
    print("Mesh shapefile CRS:", crs_info)

    n_feat = 0
    n_tri  = 0
    for ft in src:
        n_feat += 1
        geom = shape(ft["geometry"]) if ft["geometry"] else None
        if geom is None:
            continue
        tri_rings = extract_triangles_from_geometry(geom)
        for tri in tri_rings:
            # each tri is [(x1,y1),(x2,y2),(x3,y3)]
            i = add_point_get_index(tri[0][0], tri[0][1], keyfun, node_index, nodes)
            j = add_point_get_index(tri[1][0], tri[1][1], keyfun, node_index, nodes)
            k = add_point_get_index(tri[2][0], tri[2][1], keyfun, node_index, nodes)
            tri_indices.append([i, j, k])
            n_tri += 1

print(f"Features read: {n_feat} | Triangles extracted: {n_tri} | Unique nodes: {len(nodes)}")

pts  = np.asarray(nodes, dtype=np.float64)     # shape (N,2)
tris = np.asarray(tri_indices, dtype=np.int32) # shape (M,3)

# Basic validation
if pts.size == 0 or tris.size == 0:
    raise RuntimeError("No points/triangles extracted. Ensure the shapefile contains triangle polygons.")

# Save
np.save(pts_path,  pts)
np.save(tris_path, tris)

# Optional CSVs for inspection
np.savetxt(os.path.splitext(pts_path)[0] + ".csv",  pts,  delimiter=",", header="x,y", comments="")
np.savetxt(os.path.splitext(tris_path)[0]+ ".csv", tris, delimiter=",", header="i,j,k", fmt="%d", comments="")

print("Saved:")
print(" -", pts_path)
print(" -", tris_path)


# In[19]:


import pyproj
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
import fiona
import matplotlib.tri as mtri

# ---------- helpers ----------
def reproject_points_xy(pts_xy: np.ndarray, src_crs, dst_crs):
    """Reproject Nx2 array of XY coordinates from src_crs -> dst_crs."""
    if src_crs is None or dst_crs is None:
        raise ValueError("CRS missing for reprojection")
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return pts_xy
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    x, y = T(pts_xy[:,0], pts_xy[:,1])
    out = pts_xy.copy()
    out[:,0] = x
    out[:,1] = y
    return out

def read_first_linestring_with_crs(path):
    with fiona.open(path) as src:
        crs = src.crs_wkt or src.crs
        for ft in src:
            g = ft["geometry"]
            if not g: continue
            if g["type"] == "LineString":
                return LineString(g["coordinates"]), crs
            if g["type"] == "MultiLineString":
                parts = [LineString(c) for c in g["coordinates"]]
                coords = [xy for part in parts for xy in part.coords]
                return LineString(coords), crs
    raise RuntimeError(f"No LineString in {path}")

def reproject_linestring(ls, src_crs, dst_crs):
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return ls
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    return shp_transform(T, ls)

# ---------- load your arrays ----------
pts  = np.load(pts_path,  allow_pickle=False)          # Nx2 float
tris = np.load(tris_path, allow_pickle=True)           # Mx3 int
if not np.issubdtype(tris.dtype, np.integer):
    tris = tris.astype(np.int64, copy=False)

# If you know the CRS of pts/tris (e.g., EPSG:26914 or 3158), set it here:
MESH_SRC_CRS = "EPSG:3158"   # <-- set to what your pts/tris are actually in

# Optional: read DS/US lines now so they plot in the same CRS as DEM
us_line_shp = os.path.join(data_dir, "ShMouth_US_BC.shp")
ds_line_shp = os.path.join(data_dir, "Russel_DS_Boundary.shp")
US_raw, US_crs = read_first_linestring_with_crs(us_line_shp)
DS_raw, DS_crs = read_first_linestring_with_crs(ds_line_shp)

# ---------- DEM overlay ----------
asc_dem = os.path.join(data_dir, os.path.basename(f_DEM_tif).replace('.tif', '_edited.asc'))
out_over_dem_png = os.path.join(data_dir, "mesh_over_dem_preview.png")

try:
    with rio.open(asc_dem) as r:
        dem = r.read(1, masked=True)
        dem_crs = r.crs  # rasterio CRS object
        bounds = r.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Reproject pts to DEM CRS if needed
    pts_plot = reproject_points_xy(pts, MESH_SRC_CRS, dem_crs)

    # Reproject lines to DEM CRS for visual QA
    US_plot = reproject_linestring(US_raw, US_crs, dem_crs)
    DS_plot = reproject_linestring(DS_raw, DS_crs, dem_crs)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    im = ax.imshow(dem, extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation")

    triang = mtri.Triangulation(pts_plot[:, 0], pts_plot[:, 1], triangles=tris)
    ax.triplot(triang, linewidth=0.25)

    # Draw US/DS lines (now in same CRS as DEM)
    ax.plot(*US_plot.xy, lw=1.5, label="US line")
    ax.plot(*DS_plot.xy, lw=1.5, label="DS line")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mesh over DEM (all in DEM CRS)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_over_dem_png, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_over_dem_png}")

except Exception as e:
    print(f"(Skipping DEM overlay: {e})")

# From here onward, CRS  of the DEM is used

# In[20]:

#1) Domain built
#CHANGING TRIAGLES FROM Clock-Wise TO Counter CW

# 2) If triangles are object or float, sanitize to pure int
def sanitize_tris(arr):
    arr = np.asarray(arr)
    # If it's already a proper int array with shape (M,3), return
    if arr.ndim == 2 and arr.shape[1] == 3 and np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False), {}

    issues = {}
    # If ragged (1D object), try to stack into (M,3)
    if arr.dtype == object and arr.ndim == 1:
        try:
            arr = np.vstack(arr)  # raise if ragged
        except Exception as e:
            raise ValueError(f"Triangles look ragged; cannot form (M,3): {e}")

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Triangles must be (M,3); got {arr.shape} and dtype={arr.dtype}")

    # Convert anything not int to float to detect NaN/inf, then to int
    if not np.issubdtype(arr.dtype, np.integer):
        arr_float = arr.astype(float, copy=False)
        bad_nan_inf = ~np.isfinite(arr_float).all(axis=1)
        if bad_nan_inf.any():
            issues["nan_inf_rows"] = int(bad_nan_inf.sum())
            arr = arr[~bad_nan_inf]
            arr_float = arr_float[~bad_nan_inf]
        arr = arr_float.astype(np.int64, copy=False)

    # Remove rows with negatives
    neg = (arr < 0).any(axis=1)
    if neg.any():
        issues["neg_index_rows"] = int(neg.sum())
        arr = arr[~neg]
    oob = (arr.max(axis=1) >= len(pts))
    if oob.any():
        issues["oob_index_rows"] = int(oob.sum())
        arr = arr[~oob]

    # Remove degenerate triangles
    deg = (arr[:,0]==arr[:,1]) | (arr[:,1]==arr[:,2]) | (arr[:,0]==arr[:,2])
    if deg.any():
        issues["degenerate_rows"] = int(deg.sum())
        arr = arr[~deg]

    # Drop exact duplicate triangles
    if arr.size:
        before = arr.shape[0]
        arr = np.unique(arr, axis=0)
        if arr.shape[0] != before:
            issues["duplicate_rows_removed"] = int(before - arr.shape[0])

    return arr, issues

tris, issues = sanitize_tris(tris)
print("Sanitized tris:", tris.shape, tris.dtype, "| issues:", issues)

# 3) Contiguity for ANUGA and flip to CCW
pts  = np.ascontiguousarray(pts,  dtype=np.float64)
tris = np.ascontiguousarray(tris, dtype=np.int64)

def signed_area2(p, t):
    a,b,c = p[t[0]], p[t[1]], p[t[2]]
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

flip = np.array([signed_area2(pts, t) < 0 for t in tris])
if flip.any():
    tmp = tris[flip, 1].copy()
    tris[flip, 1] = tris[flip, 2]
    tris[flip, 2] = tmp
    print(f"Flipped {flip.sum()} triangles to CCW")

# 4) Final assertions before ANUGA
assert tris.ndim == 2 and tris.shape[1] == 3
assert np.issubdtype(tris.dtype, np.integer)
assert (tris.min() >= 0) and (tris.max() < len(pts))


# In[21]:


# Check current dtypes 
print("pts:", getattr(pts, "dtype", None), "itemsize:", getattr(getattr(pts,"dtype",None), "itemsize", None))
print("tris:", getattr(tris, "dtype", None), "itemsize:", getattr(getattr(tris,"dtype",None), "itemsize", None))

# Force correct dtypes + contiguity 
pts  = np.ascontiguousarray(pts,  dtype=np.float64)   # coordinates in float64
tris = np.ascontiguousarray(tris, dtype=np.int64)     # triangles in int64

# If pandas DataFrame, convert .to_numpy:

# Checks
assert pts.ndim == 2 and pts.shape[1] == 2, f"pts must be (N,2), got {pts.shape}"
assert tris.ndim == 2 and tris.shape[1] == 3, f"tris must be (M,3), got {tris.shape}"
assert tris.dtype == np.int64, f"tris dtype must be int64, got {tris.dtype}"
assert pts.dtype  == np.float64, f"pts dtype must be float64, got {pts.dtype}"
assert tris.min() >= 0, "tris contains negative indices"
assert tris.max() < len(pts), "tris index exceeds number of points"

def signed_area2(p, t):
    a,b,c = p[t[0]], p[t[1]], p[t[2]]
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
flip = np.fromiter((signed_area2(pts,t) < 0 for t in tris), count=tris.shape[0], dtype=bool)
if flip.any():
    tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()

# --- 3) Build domain safely
from anuga.shallow_water.boundaries import Reflective_boundary
domain = anuga.Domain(coordinates=pts, vertices=tris, verbose=False)
domain.set_boundary({'exterior': Reflective_boundary(domain)})


# In[22]:


import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import anuga
from anuga.shallow_water.boundaries import Reflective_boundary

# 1) domain is built from pts, tris
# 2) DEM Sampling at vertex coords for ANUGA ---

dem_path = f_DEM_tif   #  TIFF DEM

with rio.open(dem_path) as r:
    print("DEM CRS:", r.crs)
    L,B,R,T = r.bounds
    inside = (pts[:,0] >= L) & (pts[:,0] <= R) & (pts[:,1] >= B) & (pts[:,1] <= T)
    if not inside.all():
        raise RuntimeError("Mesh is outside DEM bounds. Reproject one to match the other.")

    # elevations at vertices
    Zv = np.array([v[0] for v in r.sample(pts)], dtype=float)
    if r.nodata is not None:
        Zv[Zv == r.nodata] = np.nan
    # fill NaN with nearest)
    if np.isnan(Zv).any():
        Zv[np.isnan(Zv)] = np.nanmin(Zv)

# set domain elevation on vertices
q = domain.quantities['elevation']
q.set_values(Zv, location='vertices')  

elev_v = domain.quantities['elevation'].vertex_values
print("Domain elevation stats (m):", float(elev_v.min()), float(elev_v.max()))

# Plot elevation on mesh over DEM background and save as image
triang = mtri.Triangulation(pts[:,0], pts[:,1], tris)
z_cell = domain.get_quantity('elevation').centroid_values

with rio.open(dem_path) as r:
    dem = r.read(1, masked=True)
    extent = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)

fig, ax = plt.subplots(figsize=(8,7), dpi=200)
ax.imshow(dem, extent=extent, origin="upper", alpha=0.5)
pc = ax.tripcolor(triang, facecolors=z_cell, cmap="terrain", shading="flat")
ax.triplot(triang, linewidth=0.15, alpha=0.35, color='k')
plt.colorbar(pc, ax=ax, fraction=0.04, label="Elevation (m)")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Interpolated Elevation on Mesh")
plt.tight_layout()

if ISROOT:  # only rank 0 writes files
    finish_plot(os.path.join(model_visuals_dir, "DEM_meshed.png"))
#plt.savefig(os.path.join(model_visuals_dir, "DEM_meshed.png"), bbox_inches="tight")
#plt.show()


# In[23]:

# In[24]:
# In[25]:

import rasterio as rio, numpy as np
with rio.open(f_DEM_tif) as r:        # tiff or asc_dem if you used the ASC
    z = r.read(1, masked=True)
    print("DEM stats:", float(z.min()), float(z.max()))


# In[26]:
#writefile model_settings.py
# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import utm
import sys

# Define the path to scripts and data
workshop_dir = os.getcwd()
data_dir = os.path.join(workshop_dir, 'data')
model_inputs_dir = os.path.join(workshop_dir, 'model_inputs')
model_outputs_dir = os.path.join(workshop_dir, 'model_outputs')
if 'google.colab' in sys.modules:
    data_dir = os.path.join(data_dir, 'collab')
    model_inputs_dir = os.path.join(model_inputs_dir, 'collab')
    model_outputs_dir = os.path.join(model_outputs_dir, 'collab')
model_visuals_dir = os.path.join(workshop_dir, 'visuals')
model_validation_dir = os.path.join(workshop_dir, 'validation')

sim_starttime = pd.Timestamp("2014-07-02 00:00:00", tz="UTC")
sim_endtime   = pd.Timestamp("2014-07-14 00:00:00", tz="UTC")

#sim_starttime = pd.to_datetime('2014-07-02 00:00:00', format="%Y-%m-%d %H:%M:%S", utc=True)
#sim_endtime = pd.to_datetime('2014-07-14 00:00:00', format="%Y-%m-%d %H:%M:%S", utc=True)
#sim_endtime   = sim_starttime + pd.to_timedelta(2, 'd')
sim_timestep = pd.to_timedelta(1800, 's')
sim_total_duration = (sim_endtime-sim_starttime)
data_download_dates = (str(sim_starttime-pd.to_timedelta(1, 'd'))[0:10].replace(' ', '').replace(':', ''),
                       str(sim_endtime+pd.to_timedelta(1, 'd'))[0:10].replace(' ', '').replace(':', ''))

#sim_time = np.arange(sim_starttime, sim_endtime+sim_timestep, sim_timestep)
sim_time = pd.date_range(sim_starttime, sim_endtime, freq=sim_timestep)
sim_starttime_str = str(sim_starttime)[0:19].replace('-', '').replace(' ', '').replace(':', '')

# Output file naming
domain_name = 'Shellmouth_flood'
model_name = f'{sim_starttime_str}_{domain_name}_{sim_total_duration.days}_days'

# Boundary Conditions
discharge_gauge_x, discharge_gauge_y = utm.from_latlon(50.960278, -101.412222)[0:2] # Lake of Prairies
discharge_gauge_ID = ('05MD009', 'DAM', discharge_gauge_x, discharge_gauge_y)

level_gauge_x, level_gauge_y = utm.from_latlon(50.993889, -101.287222)[0:2] # Shellmouth bridge
level_gauge_ID = ('05MD801', 'Russel', level_gauge_x, level_gauge_y)

# Gauges filepath
f_discharge =  os.path.join(model_inputs_dir, 'Discharge_at_%s_%s-%s.csv' % (
               discharge_gauge_ID[1],
               data_download_dates[0].replace('-', ''), 
               data_download_dates[1].replace('-', '')))

f_level =  os.path.join(model_inputs_dir, 'Level_at_%s_%s-%s.csv' % (
              level_gauge_ID[1],
              data_download_dates[0].replace('-', ''), 
              data_download_dates[1].replace('-', '')))
# In[27]:
# --- Load boundary-condition CSVs and parse time to UTC ---
df_q  = pd.read_csv(f_discharge)  # columns expected: time, Q
df_wl = pd.read_csv(f_level)      # columns expected: time, Level (or similar)

# Robust time parsing (handles ...Z or ...+00:00; no format= needed)
df_q["time"]  = pd.to_datetime(df_q["time"],  utc=True)
df_wl["time"] = pd.to_datetime(df_wl["time"], utc=True)

# (optional) set index if you resample/join on time later
# df_q  = df_q.set_index("time")
# df_wl = df_wl.set_index("time")


# Save settings to input directory
f_model_settings_in = os.path.join(workshop_dir, 'model_settings.py')
f_model_settings_out = os.path.join(model_inputs_dir, 'model_settings.py')
if os.path.exists(f_model_settings_in):
    os.replace(f_model_settings_in, f_model_settings_out)

# Import all settings as global variables
if 'google.colab' not in sys.modules:
    from model_inputs.model_settings import *
else:
    from model_inputs.collab.model_settings import *


# In[28]:


#get_ipython().run_line_magic('matplotlib', 'inline')

# Write “clean” files without offset in the time strings
f_discharge_clean = os.path.join(model_inputs_dir, "Discharge_at_DAM_clean.csv")
f_level_clean     = os.path.join(model_inputs_dir, "Level_at_Russel_clean.csv")

df_q.assign(time=df_q["time"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(f_discharge_clean, index=False)
df_wl.assign(time=df_wl["time"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(f_level_clean, index=False)


# Prepare boundary condition functions for Anuga
discharge_function = at.GenerateHydrograph(filename = f_discharge_clean,
                                            smoothing = False, 
                                            t_start=sim_starttime, 
                                            t_end=sim_endtime, 
                                            t_step=sim_timestep,
                                            progressive=False) 

level_function = at.GenerateTideGauge(filename = f_level_clean,
                                      t_start = sim_starttime, 
                                      t_end = sim_endtime, 
                                      t_step = sim_timestep,
                                      offset = 0,
                                      smoothing = False,
                                      smoothing_span = .1)


# Data visualization
t = (sim_time-sim_time[0]).astype('timedelta64[s]').astype(float)
discharge_ts = [discharge_function(i) for i in t]
level_ts = [level_function(i) for i in t]

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6,4), dpi=100)
ax1.plot(sim_time, discharge_ts, label=discharge_gauge_ID[1])
ax1.legend()
ax1.grid('on')
ax1.set_ylabel('Discharge [m$^3$/s]')
ax1.set_xticklabels([])

ax2.plot(sim_time, level_ts, label=level_gauge_ID[1])
ax2.legend()
ax2.grid('on')
ax2.set_xlabel('Date [yyyy-mm-dd]')
ax2.set_ylabel('Water Level [m]')
for tick in ax2.get_xticklabels():
    tick.set_rotation(30)


if ISROOT:  # only rank 0 writes files
    finish_plot(os.path.join(model_visuals_dir, "hydrodynamic_inputs.png"))
#plt.savefig(os.path.join(model_visuals_dir, 'hydrodynamic_inputs.png'))
#plt.show()


# In[29]:


# Fill elevation NaNs if any after DEM interpolation
elev = domain.get_quantity('elevation').get_values(location='vertices')
if np.isnan(elev).any():
    print(f" Found {np.isnan(elev).sum()} NaNs in interpolated elevation — replacing with mean elevation.")
    elev[np.isnan(elev)] = np.nanmean(elev)
    domain.set_quantity('elevation', elev, location='centroids')


# In[30]:


# Set up the model
# Initial water level
domain.set_quantity('elevation', Zv, location='vertices') 
domain.set_quantity('stage', expression='elevation')  # 10 cm
# Initial time
domain.set_starttime(0) 
# Computational parameters
domain.set_flow_algorithm('DE1')
domain.set_name(model_name)
domain.set_low_froude(1) # Use low-froude DE1 to reduce flux damping
domain.set_minimum_allowed_height(0.1) # Only store heights > the given value 10 cm


# In[31]:

# Load discharge transect US/BC
discharge_transect_gdf = gpd.read_file(f_US_BC)
discharge_loc = np.asarray(discharge_transect_gdf.geometry[0].xy).T.tolist()


# In[34]:
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform as shp_transform
from functools import partial
import pyproj

# -------------------- CONFIG --------------------
MESH_CRS_EPSG = 26914
us_line_shp = os.path.join(data_dir, "ShMouth_US_BC.shp")
ds_line_shp = os.path.join(data_dir, "Russel_DS_Boundary.shp")
BASE_TOL = 20.0  # meters (used as an upper cap; code will auto-scale from edge lengths)

# -------------------- HELPERS --------------------
def read_first_linestring_with_crs(path):
    with fiona.open(path) as src:
        crs = src.crs_wkt or src.crs
        for ft in src:
            g = ft["geometry"]
            if not g: continue
            if g["type"] == "LineString":
                return LineString(g["coordinates"]), crs
            if g["type"] == "MultiLineString":
                parts = [LineString(c) for c in g["coordinates"]]
                coords = [xy for part in parts for xy in part.coords]
                return LineString(coords), crs
    raise RuntimeError(f"No LineString in {path}")

def reproject_linestring(ls, src_crs, dst_epsg):
    if not src_crs:
        raise RuntimeError("Source shapefile has no CRS; set or re-save it with a defined CRS.")
    try:
        # fiona gives dict or WKT; pyproj can handle both
        dst_crs = pyproj.CRS.from_epsg(dst_epsg)
        src_crs = pyproj.CRS.from_user_input(src_crs)
        proj = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
        return shp_transform(proj, ls)
    except Exception as e:
        raise RuntimeError(f"CRS transform failed: {e}")

def edge_nodes(v_idx, e):
    a,b,c = v_idx
    return [(b,c),(c,a),(a,b)][e]

def boundary_edges_segments(pts, tris):
    """Return list of ((tri_id, e), seg_line, seg_len, mid_xy) for exterior edges."""
    tmp = anuga.Domain(coordinates=pts, vertices=tris)
    out = []
    for (tri_id, e) in tmp.boundary.keys():
        v = tris[tri_id]
        i,j = edge_nodes(v, e)
        p1 = tuple(pts[i]); p2 = tuple(pts[j])
        seg = LineString([p1, p2])
        L = float(seg.length)
        mid = (0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]))
        out.append(((tri_id, e), seg, L, mid))
    return out

# -------------------- 1) Read & reproject lines --------------------
US_raw, US_crs = read_first_linestring_with_crs(us_line_shp)
DS_raw, DS_crs = read_first_linestring_with_crs(ds_line_shp)

US = reproject_linestring(US_raw, US_crs, MESH_CRS_EPSG)
DS = reproject_linestring(DS_raw, DS_crs, MESH_CRS_EPSG)

# -------------------- 2) Quick sanity prints --------------------
mesh_xmin, mesh_ymin = np.min(pts, axis=0)
mesh_xmax, mesh_ymax = np.max(pts, axis=0)
print(f"Mesh bounds:   x[{mesh_xmin:.1f}, {mesh_xmax:.1f}]  y[{mesh_ymin:.1f}, {mesh_ymax:.1f}]")
print(f"US bounds:     {US.bounds}")
print(f"DS bounds:     {DS.bounds}")

# Check that line bounds overlap mesh bbox at least partially
def overlaps(b1, b2):
    (x1min,y1min,x1max,y1max) = b1
    (x2min,y2min,x2max,y2max) = b2
    return (x1min <= x2max and x2min <= x1max and y1min <= y2max and y2min <= y1max)

if not overlaps((mesh_xmin,mesh_ymin,mesh_xmax,mesh_ymax), DS.bounds):
    print("WARNING: DS line bbox does not overlap mesh bbox. CRS or location may be wrong.")

# -------------------- 3) Build exterior segments & compute distances --------------------
items = boundary_edges_segments(pts, tris)
edge_lengths = np.array([L for _,_,L,_ in items])
auto_tol = (3.0 * np.percentile(edge_lengths, 10)) if len(edge_lengths) else 5.0
tol = min(BASE_TOL, max(auto_tol, 1.0))  # meters
print(f"Auto tolerance (capped): {tol:.2f} m")

# Compute minimum distances from DS to edges (segment-to-line distance)
dists = np.array([seg.distance(DS) for _, seg, _, _ in items])
print(f"Min dist DS->any exterior edge: {dists.min():.3f} m (tagging tol = {tol:.3f} m)")

# -------------------- 4) Visualize edges near DS --------------------
near_mask = dists <= tol
print(f"Edges within tol of DS: {near_mask.sum()}")

# (Optional plot)
fig, ax = plt.subplots(figsize=(6,6), dpi=140)
ax.scatter(pts[:,0], pts[:,1], s=1, alpha=0.2)
ax.plot(*US.xy, lw=2, label='US line')
ax.plot(*DS.xy, lw=2, label='DS line')
# draw some exterior edges
for k,( (tri_e), seg, L, mid) in enumerate(items[::max(1,len(items)//500)]):  # subsample for speed
    x,y = seg.xy
    ax.plot(x,y, lw=0.5, alpha=0.2, color='gray')
# highlight edges within tol
for (ok, (_, seg, _, _)) in zip(near_mask, items):
    if ok:
        x,y = seg.xy
        ax.plot(x,y, lw=2.0)
ax.legend(); ax.set_title("Exterior edges near DS")
#plt.show()
fig.tight_layout()

if ISROOT:  # only rank 0 writes files
    finish_plot(os.path.join(model_visuals_dir, "exterior_edges_near_DS.png"))

# -------------------- 5) Build boundary map --------------------
boundary_map = {}
n_inlet = n_outlet = 0
# Uses boundary segment distance
for ((tri_id, e), seg, _, _) in items:
    dUS = seg.distance(US)
    dDS = seg.distance(DS)
    if dUS <= tol:
        boundary_map[(tri_id, e)] = "inlet";  n_inlet += 1
    elif dDS <= tol:
        boundary_map[(tri_id, e)] = "outlet"; n_outlet += 1
    else:
        boundary_map[(tri_id, e)] = "sides"

print(f"Tagged edges -> inlet: {n_inlet}, outlet: {n_outlet}, sides: {len(items)-n_inlet-n_outlet}")

if n_outlet == 0:
    tol2 = max(tol, min(BASE_TOL, dists.min()*1.25 + 1.0))
    print(f"Retry with larger tol = {tol2:.2f} m")
    boundary_map.clear(); n_inlet = n_outlet = 0
    for ((tri_id, e), seg, _, _) in items:
        dUS = seg.distance(US)
        dDS = seg.distance(DS)
        if dUS <= tol2:
            boundary_map[(tri_id, e)] = "inlet";  n_inlet += 1
        elif dDS <= tol2:
            boundary_map[(tri_id, e)] = "outlet"; n_outlet += 1
        else:
            boundary_map[(tri_id, e)] = "sides"
    

    if n_outlet == 0:
        raise RuntimeError(
            "no 'outlet', line not touching domain, extend to cross the domain boundary"
            )


# In[35]:

# In[36]:


# Tag boundaries
from collections import defaultdict
import numpy as np
import anuga

# 1) declare the names and stable integer ids
boundary_tags = {"inlet": 1, "outlet": 2, "sides": 3}

# 2) build tagged_elements: int id -> list[ (tri_id, edge_idx) ]
tagged_elements = {i: [] for i in boundary_tags.values()}

for (tri_e, name) in boundary_map.items():
    tri_id, e = tri_e
    # be strict about Python int (not numpy types)
    tri_id = int(tri_id); e = int(e)
    if name not in boundary_tags:
        raise ValueError(f"Unknown tag name in boundary_map: {name}")
    tagged_elements[boundary_tags[name]].append((tri_id, e))

assert pts.ndim == 2 and pts.shape[1] == 2
assert tris.ndim == 2 and tris.shape[1] == 3
if not np.issubdtype(tris.dtype, np.integer):
    tris = tris.astype(np.int64, copy=False)

# boundary_map: {(tri_id:int, edge:int[0..2]) : 'inlet'/'outlet'/'sides'}
assert isinstance(boundary_map, dict) and len(boundary_map) > 0

# Build the Domain WITH boundary only 
domain.boundary = boundary_map

# boundary tag names from the map
existing_tags = sorted(set(boundary_map.values()))
print("Existing boundary tags on domain:", existing_tags)

# Build BC dict only for available tags 
Bc = {}
if "outlet" in existing_tags:
    Bc["outlet"] = anuga.Transmissive_stage_zero_momentum_boundary(domain)
if "inlet" in existing_tags:
    pass
if "sides" in existing_tags:
    Bc["sides"] = anuga.Reflective_boundary(domain)

if not Bc and "exterior" in existing_tags:
    Bc["exterior"] = anuga.Reflective_boundary(domain)

#Apply
if "outlet" not in existing_tags:
    print("WARNING: outlet tag not available")
domain.set_boundary(Bc)
print("boundary applied.")



# In[37]:
# Initialize discharge inlet
inlet_ATC = Inlet_operator(domain, discharge_loc, Q = discharge_function(0))
# Assign Friction
domain.set_quantity('friction', 0.03, location = 'centroids') # Manning N

# In[38]:
#print interpolated levels
level_ts


# In[39]:

allowed_tags = sorted(set(domain.boundary.values()))
print("Allowed tags for this domain:", allowed_tags)

Bc = {}

# Outlet: time-varying stage, zero momentum
def stage_func(t):
    return float(level_function(t))  
try:
    outlet_bc = anuga.shallow_water.boundaries.Time_stage_zero_momentum_boundary(domain, stage_func)
except AttributeError:
    outlet_bc = anuga.shallow_water.boundaries.Time_stage_zero_momentum_boundary(domain, function=stage_func)

# Sides are reflective
from anuga.shallow_water.boundaries import Reflective_boundary
sides_bc = Reflective_boundary(domain)

if 'outlet' in allowed_tags:
    Bc['outlet'] = outlet_bc
if 'sides' in allowed_tags:
    Bc['sides'] = sides_bc

domain.set_boundary(Bc)


# In[40]:
# Run the model
timestep = sim_timestep.total_seconds()
finaltime = (sim_time.shape[0]-1)*timestep
total_number_of_steps = sim_time.shape[0]
for n, t in tqdm(enumerate(domain.evolve(yieldstep=timestep, finaltime=finaltime)), 
                          total=total_number_of_steps):
#for n, t in enumerate(domain.evolve(yieldstep=timestep, finaltime=finaltime)):
    
    # Update discharge value at the inlet
    inlet_ATC.Q = discharge_function(t)

    # Optional, report volume conservation at some intervals
    if n % 2 == 0:
        print(f"\nTime = {t:.2f} s (step {n})")
        domain.print_timestepping_statistics()
        domain.report_water_volume_statistics()

        # Check for negative depths
        shallow = domain.quantities['stage'].centroid_values - domain.quantities['elevation'].centroid_values
        if (shallow < 0).any():
            print("Warning: Negative water depths detected.")

    # or log to a file for warnings


# In[42]:


model_name


# In[41]:

import shutil
# Save output to the dedicated directory
f_anuga_output_in = os.path.join(workshop_dir, "20140702000000_Shellmouth_flood_12_days.sww")
f_anuga_output_out = os.path.join(model_outputs_dir, "20140702000000_Shellmouth_flood_12_days.sww")

shutil.copy2(f_anuga_output_in, f_anuga_output_out)

# In[ ]:




