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
