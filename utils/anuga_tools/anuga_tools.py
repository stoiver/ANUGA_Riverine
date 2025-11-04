import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from osgeo import gdal
import pandas as pd
import numpy as np

def _read_timeseries_any_tz(csv_path, time_col="time"):
    df = pd.read_csv(csv_path)
    # find the time column case-insensitively if needed
    if time_col not in df.columns:
        lc = {c.lower(): c for c in df.columns}
        if time_col.lower() in lc:
            time_col = lc[time_col.lower()]
        else:
            raise KeyError(f"'{time_col}' not found in {csv_path}. Columns: {list(df.columns)}")

    s = df[time_col].astype(str).str.replace("Z", "+00:00", regex=False)

    # Accept both “YYYY-mm-dd HH:MM:SS” and “...+00:00” without specifying a format
    try:
        t = pd.to_datetime(s, utc=True)               # works for both with/without TZ
    except Exception:
        # Pandas ≥2 has ISO8601 parser; keep as a fallback
        t = pd.to_datetime(s, format="ISO8601", utc=True)

    df[time_col] = t

    # Make common value column names easy to use
    # Discharge
    for cand in ["Q", "Discharge", "water_level", "discharge", "flow", "Flow"]:
        if cand in df.columns:
            df["__value__"] = pd.to_numeric(df[cand], errors="coerce")
            break
    # Level (if not set already)
    if "__value__" not in df.columns:
        for cand in ["Level", "level", "WSE", "stage", "Stage"]:
            if cand in df.columns:
                df["__value__"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "__value__" not in df.columns:
        raise KeyError(f"Could not find a value column (tried Q/Discharge/Level/WSE/Stage) in {csv_path}")

    df = df.dropna(subset=[time_col, "__value__"])
    return df.rename(columns={time_col: "time"})


def GenerateTideGauge(
    filename,
    t_start,
    t_end,
    t_step,
    offset=0.0,
    smoothing=False,
    smoothing_span=0.1,
    hot_start=False,
    last_frame=0,
):
    """
    Build a time-varying stage function from a CSV.
    Accepts datetime strings with or without timezone (+00:00 / Z).
    Expects a value column among: WL, Level, WSE, stage, Stage (case-insensitive).
    """
    # Parse times & value column robustly
    df = _read_timeseries_any_tz(filename, time_col="time")
    if "__value__" not in df.columns:
        # prefer common water-level names
        for cand in ["WL", "Level", "WSE", "water_level", "stage", "Stage", "level"]:
            if cand in df.columns:
                df["__value__"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "__value__" not in df.columns:
        raise KeyError("No water-level column found (looked for WL/Level/WSE/stage).")

    # limit to sim window
    if hot_start:
        t_start = t_start + (t_step * last_frame)
    mask = (df["time"] >= t_start) & (df["time"] <= t_end)
    df = df.loc[mask].sort_values("time")

    # seconds since t_start
    ts = (df["time"] - t_start).dt.total_seconds().to_numpy()

    # values (+ offset), optional smoothing in time
    vals = (df["__value__"].to_numpy(dtype=float) + float(offset))
    if smoothing:
        vals = lowess(vals, df["time"].view("i8"), is_sorted=True, frac=smoothing_span, it=1)[:, 1]

    # interpolation function
    def fBC_tides(t):
        # extrapolate flat at ends
        return float(np.interp(t, ts, vals, left=vals[0], right=vals[-1]))

    return fBC_tides


def GenerateHydrograph(
    filename,
    t_start,
    t_end,
    t_step,
    offset=0.0,
    smoothing=False,
    smoothing_span=0.25,
    progressive=True,
    hot_start=False,
    last_frame=0,
):
    """
    Build a discharge function from a CSV.
    Accepts datetime strings with or without timezone (+00:00 / Z).
    Expects a value column among: Q, Discharge, flow (case-insensitive).
    """
    # Parse times & value column robustly
    df = _read_timeseries_any_tz(filename, time_col="time")
    if "__value__" not in df.columns:
        for cand in ["Q", "Discharge", "discharge", "flow", "Flow"]:
            if cand in df.columns:
                df["__value__"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "__value__" not in df.columns:
        raise KeyError("No discharge column found (looked for Q/Discharge/flow).")

    if hot_start:
        t_start = t_start + (t_step * last_frame)

    # limit to sim window
    mask = (df["time"] >= t_start) & (df["time"] <= t_end)
    df = df.loc[mask].sort_values("time")

    ts = (df["time"] - t_start).dt.total_seconds().to_numpy()
    vals = (df["__value__"].to_numpy(dtype=float) + float(offset))

    if smoothing:
        vals = lowess(vals, df["time"].view("i8"), is_sorted=True, frac=smoothing_span, it=1)[:, 1]

    # Progressive ramp (up to first 100 points, but safe if shorter)
    if progressive and vals.size:
        n_ramp = min(100, vals.size)
        ramp = (np.arange(1, n_ramp + 1) / n_ramp)
        vals[:n_ramp] = vals[:n_ramp] * ramp

    def fQ(t):
        return float(np.interp(t, ts, vals, left=vals[0], right=vals[-1]))

    return fQ


def GenerateTideCosine(amplitude=0.25, period=7.2722e-5, phase=0, offset=0.26):
    """
    Function to generate an artificial tidal BC from a sinusoid. Default is
    cosine with same tidal range as LAWMA, freq of N2.
    Returns a function of time.
    """
    # Default is pseudo tide with same tidal range as LAWMA, freq of N2. Time in sec
    def fBC_tides(t):
        return amplitude*np.cos(t*period + phase) + offset
    return fBC_tides

# ------------------------------------------------------------------------------
# Domain settings functions
# ------------------------------------------------------------------------------
def Raster2Mesh(meshX, meshY, raster_filename):
    """
    Function to grab raster value at each mesh cell centroid.
    Used for friction assignment. Specify raster in settings.py.
    Returns a numpy.ndarray
    """
    src = gdal.Open(raster_filename)
    data_array = np.array(src.GetRasterBand(1).ReadAsArray())
    # Get geographic info
    transform = src.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Loop and grab all values at mesh coords
    meshVal = np.zeros(len(meshX), dtype=int)
    for ii in list(range(0, len(meshX))):
        col = int((meshX[ii] - xOrigin) / pixelWidth)
        row = int((yOrigin - meshY[ii] ) / pixelHeight)
        try:
            meshVal[ii] = data_array[row][col]
        except IndexError:
            meshVal[ii] = 0
    # Return values
    return meshVal

def AssignFricValue(FricVal, n_array, m_array, h_array, D_array):
    """
    Using Friction ID (output of Raster2Mesh), assign friction parameters.
    All Mannings (n) and Baptist (m, hv, D) values returned.
    Specify coefficients for each class in settings.py
    """
    FricVal = FricVal.astype(int)-1 # Subtract 1 assuming map ID's start at 1
    n = np.zeros_like(FricVal, dtype=float)
    m = np.zeros_like(FricVal, dtype=float)
    hv = np.zeros_like(FricVal, dtype=float)
    D = np.zeros_like(FricVal, dtype=float)
    
    for ii in list(range(len(FricVal))):
        n[ii] = n_array[FricVal[ii]]
        m[ii] = m_array[FricVal[ii]]
        hv[ii] = h_array[FricVal[ii]]
        D[ii] = D_array[FricVal[ii]]
    return n, m, hv, D