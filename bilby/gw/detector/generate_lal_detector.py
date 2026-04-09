#!/usr/bin/env python3
"""
bilby_to_lal_detector.py
------------------------
Convert an arbitrary bilby interferometer definition file into LAL-style
#define constants suitable for LALDetectors.h / BayesWave.

Supports:
  shape = 'L'        -> one detector block  (standard IFO, e.g. LIGO, CE)
  shape = 'Triangle' -> three sub-detectors + null stream (e.g. Einstein Telescope)

Usage:
    python bilby_to_lal_detector.py <detector.cfg> [PREFIX]

    detector.cfg  -- path to a bilby detector definition file
    PREFIX        -- optional 2-3 char LAL prefix to use instead of auto-derived one
                     For Triangle detectors this is the base (e.g. "EM" -> EM1/EM2/EM3/EM0)
                     For L-shaped detectors this is used directly (e.g. "CE")

Example bilby files
-------------------
L-shaped (Cosmic Explorer style):
    name = 'CE'
    latitude  = 46.455
    longitude = -119.408
    elevation = 0.0
    xarm_azimuth = 135.0
    yarm_azimuth = 225.0
    length = 40

Triangle (Einstein Telescope EMR site):
    name = 'ET_D_EMR'
    latitude  = 50 + 43./60 + 50.30/3600
    longitude = 5  + 54./60 +  2.50/3600
    elevation = 0.0
    xarm_azimuth = 70.5674
    yarm_azimuth = 130.5674
    shape = 'Triangle'
    length = 10

Notes
-----
- Bilby azimuths are clockwise from North (geographic), as are LAL azimuths.
- latitude / longitude can be plain floats or simple arithmetic expressions
  (e.g. "50 + 43./60 + 50.30/3600") -- both are evaluated safely.
- For Triangle detectors, sub-detectors 2 and 3 are the stated arms rotated
  by +120 and +240 degrees respectively. All three share the same vertex.
  The null-stream (index 0) has zeroed arm vectors.
- The LAL prefix for L-shaped detectors defaults to the first 2 chars of
  `name` uppercased; for Triangle it appends 1/2/3/0.
"""

import sys
import re
import ast
import numpy as np
from bilby.core.utils.constants import radius_of_earth as R_EARTH

# ---------------------------------------------------------------------------
# WGS-84 constants
# ---------------------------------------------------------------------------
WGS84_A  = 6_378_137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_B  = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 1 - (WGS84_B / WGS84_A) ** 2

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg, lon_deg, elev_m):
    """Geodetic (WGS-84) -> Earth-centred Earth-fixed Cartesian (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + elev_m) * np.cos(lat) * np.cos(lon)
    y = (N + elev_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E2) + elev_m) * np.sin(lat)
    return np.array([x, y, z])


def az_alt_to_ecef(lat_deg, lon_deg, az_deg, alt_deg=0.0):
    """
    Convert arm azimuth + altitude (degrees, CW from North) to an ECEF
    unit vector via the local East-North-Up frame.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az  = np.radians(az_deg)
    alt = np.radians(alt_deg)

    # ENU components of the arm direction
    e_enu = np.sin(az) * np.cos(alt)
    n_enu = np.cos(az) * np.cos(alt)
    u_enu = np.sin(alt)

    # ENU basis vectors in ECEF
    east  = np.array([-np.sin(lon),  np.cos(lon),  0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)])
    up    = np.array([ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon),  np.sin(lat)])

    return e_enu * east + n_enu * north + u_enu * up

# ---------------------------------------------------------------------------
# Bilby file parser
# ---------------------------------------------------------------------------

def _safe_eval_number(expr):
    """
    Safely evaluate a numeric expression string such as
    "50 + 43./60 + 50.30/3600" or "-119.408".
    Only allows numbers and basic arithmetic operators.
    """
    expr = re.sub(r'#.*', '', expr).strip().strip("'\"")
    if re.search(r'[^0-9eE+\-*/().\s]', expr):
        raise ValueError(f"Unsafe expression in detector file: {expr!r}")
    return float(eval(compile(ast.parse(expr, mode='eval'), '<string>', 'eval')))  # noqa: S307


def parse_bilby_file(path):
    """
    Parse a bilby detector definition file and return a dict of parameters.

    Recognised keys (case-insensitive):
        name, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
        shape, length (arm length in km)
    """
    params = {}
    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key   = key.strip().lower()
            value = re.sub(r'\s*#.*', '', value).strip()  # drop inline comments

            if key in ('name', 'shape', 'power_spectral_density'):
                params[key] = value.strip("'\"")
            elif key in ('latitude', 'longitude', 'elevation',
                         'xarm_azimuth', 'yarm_azimuth', 'length',
                         'minimum_frequency', 'maximum_frequency'):
                try:
                    params[key] = _safe_eval_number(value)
                except Exception as exc:
                    raise ValueError(f"Cannot parse '{key} = {value}': {exc}") from exc
    return params


def validate_params(p):
    """Check required keys are present and fill sensible defaults."""
    for k in ('name', 'latitude', 'longitude', 'xarm_azimuth', 'yarm_azimuth'):
        if k not in p:
            raise KeyError(f"Required key '{k}' missing from detector file.")

    p.setdefault('elevation', 0.0)
    p.setdefault('length', None)
    p.setdefault('shape', 'L')

    shape = p['shape'].strip("'\" ").capitalize()
    if shape not in ('L', 'Triangle'):
        raise ValueError(f"Unsupported shape '{shape}'. Must be 'L' or 'Triangle'.")
    p['shape'] = shape
    return p

# ---------------------------------------------------------------------------
# Detector builder
# ---------------------------------------------------------------------------

def build_detectors(p, prefix_override=None):
    """
    Given validated bilby parameters, return a list of detector dicts
    ready for LAL output.
    """
    name     = p['name']
    lat      = p['latitude']
    lon      = p['longitude']
    elev     = p['elevation']
    xaz      = p['xarm_azimuth']
    yaz      = p['yarm_azimuth']
    arm_len  = p['length'] * 1000.0 if p['length'] is not None else None  # km -> m
    midpoint = arm_len / 2.0 if arm_len is not None else 0.0
    shape    = p['shape']

    vertex  = geodetic_to_ecef(lat, lon, elev)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Derive LAL prefix
    if prefix_override:
        base_prefix = prefix_override.upper()
    else:
        alnum = re.sub(r'[^A-Za-z0-9]', '', name)
        base_prefix = alnum[:2].upper().ljust(2, 'X')

    detectors = []

    if shape == 'L':
        detectors.append({
            "idx":        None,
            "name":       name,
            "prefix":     base_prefix,
            "lon_rad":    lon_rad,
            "lat_rad":    lat_rad,
            "elev_m":     elev,
            "xarm_az":    np.radians(xaz),
            "yarm_az":    np.radians(yaz),
            "xarm_alt":   0.0,
            "yarm_alt":   0.0,
            "midpoint_m": midpoint,
            "vertex":     vertex,
            "x_dir":      az_alt_to_ecef(lat, lon, xaz),
            "y_dir":      az_alt_to_ecef(lat, lon, yaz),
        })

    else:  # Triangle
        # Replicate bilby's TriangularInterferometer vertex-walking logic exactly.
        #
        # bilby converts xarm_azimuth (CW from North) to a geodesic bearing via:
        #   brng = 90 - xarm_azimuth   (converts to CCW-from-East / math angle)
        # Then after each sub-detector it advances the vertex by `length` km along
        # that bearing on the sphere, and rotates both bearing and azimuths by +240°,
        # which closes an equilateral triangle after three steps.
        #
        # The geodesic step uses the standard spherical-Earth forward formula:
        #   phi2 = arcsin( sin(phi1)*cos(d/R) + cos(phi1)*sin(d/R)*cos(brng) )
        #   lam2 = lam1 + arctan2( sin(brng)*sin(d/R)*cos(phi1),
        #                           cos(d/R) - sin(phi1)*sin(phi2) )
        # where d = arm_length in metres, R = radius of Earth.

        cur_lat = lat
        cur_lon = lon
        cur_xaz = xaz
        cur_yaz = yaz
        cur_brng = xaz   # bilby: brng = 90 - xarm_azimuth

        for i in range(1, 4):
            vertex_i  = geodetic_to_ecef(cur_lat, cur_lon, elev)
            lat_rad_i = np.radians(cur_lat)
            lon_rad_i = np.radians(cur_lon)

            detectors.append({
                "idx":        i,
                "name":       f"{name}_{i}_T1400308",
                "prefix":     f"{base_prefix}{i}",
                "lon_rad":    lon_rad_i,
                "lat_rad":    lat_rad_i,
                "elev_m":     elev,
                "xarm_az":    np.radians(cur_xaz % 360.0),
                "yarm_az":    np.radians(cur_yaz % 360.0),
                "xarm_alt":   0.0,
                "yarm_alt":   0.0,
                "midpoint_m": midpoint,
                "vertex":     vertex_i,
                "x_dir":      az_alt_to_ecef(cur_lat, cur_lon, cur_xaz % 360.0),
                "y_dir":      az_alt_to_ecef(cur_lat, cur_lon, cur_yaz % 360.0),
            })

            # Walk the vertex forward along cur_brng by arm_length metres
            d      = arm_len if arm_len is not None else 0.0
            phi1   = np.radians(cur_lat)
            lam1   = np.radians(cur_lon)
            brng_r = np.radians(cur_brng)

            phi2 = np.arcsin(
                np.sin(phi1) * np.cos(d / R_EARTH) +
                np.cos(phi1) * np.sin(d / R_EARTH) * np.cos(brng_r)
            )
            lam2 = lam1 + np.arctan2(
                np.sin(brng_r) * np.sin(d / R_EARTH) * np.cos(phi1),
                np.cos(d / R_EARTH) - np.sin(phi1) * np.sin(phi2)
            )

            cur_lat  = np.degrees(phi2)
            cur_lon  = np.degrees(lam2)
            cur_brng -= 240.0
            cur_xaz  -= 240.0
            cur_yaz  -= 240.0

        # Null stream: vertex at the original (first) location, arms zeroed
        detectors.append({
            "idx":        0,
            "name":       f"{name}_0_T1400308",
            "prefix":     f"{base_prefix}0",
            "lon_rad":    lon_rad,
            "lat_rad":    lat_rad,
            "elev_m":     elev,
            "xarm_az":    0.0,
            "yarm_az":    0.0,
            "xarm_alt":   0.0,
            "yarm_alt":   0.0,
            "midpoint_m": 0.0,
            "vertex":     geodetic_to_ecef(lat, lon, elev),
            "x_dir":      np.zeros(3),
            "y_dir":      np.zeros(3),
        })

    return detectors

# ---------------------------------------------------------------------------
# LAL #define output
# ---------------------------------------------------------------------------

def fmt_f(v, d=11):
    return f"{v:.{d}f}"


def detector_to_defines(d):
    tag = f"LAL_{d['prefix']}"
    pfx = d['prefix']
    lines = ["/** @{ */"]

    def macro(suffix, value, comment):
        m = f"#define {tag}_DETECTOR_{suffix}"
        return f"{m:<60}\t{value}\t/**< {comment} */"

    def macro_vec(suffix, value, comment):
        m = f"#define {tag}_{suffix}"
        return f"{m:<60}\t{value}\t/**< {comment} */"

    lines.append(macro("NAME",            f'"{d["name"]}"',          f'{pfx} detector name string'))
    lines.append(macro("PREFIX",          f'"{pfx}"',                f'{pfx} detector prefix string'))
    lines.append(macro("LONGITUDE_RAD",   fmt_f(d["lon_rad"]),       f'{pfx} vertex longitude (rad)'))
    lines.append(macro("LATITUDE_RAD",    fmt_f(d["lat_rad"]),       f'{pfx} vertex latitude (rad)'))
    lines.append(macro("ELEVATION_SI",    fmt_f(d["elev_m"], 3),     f'{pfx} vertex elevation (m)'))
    lines.append(macro("ARM_X_AZIMUTH_RAD",  fmt_f(d["xarm_az"]),   f'{pfx} x arm azimuth (rad)'))
    lines.append(macro("ARM_Y_AZIMUTH_RAD",  fmt_f(d["yarm_az"]),   f'{pfx} y arm azimuth (rad)'))
    lines.append(macro("ARM_X_ALTITUDE_RAD", fmt_f(d["xarm_alt"]),  f'{pfx} x arm altitude (rad)'))
    lines.append(macro("ARM_Y_ALTITUDE_RAD", fmt_f(d["yarm_alt"]),  f'{pfx} y arm altitude (rad)'))
    lines.append(macro("ARM_X_MIDPOINT_SI",  fmt_f(d["midpoint_m"]),f'{pfx} x arm midpoint (m)'))
    lines.append(macro("ARM_Y_MIDPOINT_SI",  fmt_f(d["midpoint_m"]),f'{pfx} y arm midpoint (m)'))

    vx, vy, vz = d["vertex"]
    lines.append(macro_vec("VERTEX_LOCATION_X_SI", f"{vx:.5e}", f'{pfx} x-component of vertex location in Earth-centered frame (m)'))
    lines.append(macro_vec("VERTEX_LOCATION_Y_SI", f"{vy:.5e}", f'{pfx} y-component of vertex location in Earth-centered frame (m)'))
    lines.append(macro_vec("VERTEX_LOCATION_Z_SI", f"{vz:.5e}", f'{pfx} z-component of vertex location in Earth-centered frame (m)'))

    xx, xy, xz = d["x_dir"]
    lines.append(macro_vec("ARM_X_DIRECTION_X", fmt_f(xx), f'{pfx} x-component of unit vector along x arm in Earth-centered frame'))
    lines.append(macro_vec("ARM_X_DIRECTION_Y", fmt_f(xy), f'{pfx} y-component of unit vector along x arm in Earth-centered frame'))
    lines.append(macro_vec("ARM_X_DIRECTION_Z", fmt_f(xz), f'{pfx} z-component of unit vector along x arm in Earth-centered frame'))

    yx, yy, yz = d["y_dir"]
    lines.append(macro_vec("ARM_Y_DIRECTION_X", fmt_f(yx), f'{pfx} x-component of unit vector along y arm in Earth-centered frame'))
    lines.append(macro_vec("ARM_Y_DIRECTION_Y", fmt_f(yy), f'{pfx} y-component of unit vector along y arm in Earth-centered frame'))
    lines.append(macro_vec("ARM_Y_DIRECTION_Z", fmt_f(yz), f'{pfx} z-component of unit vector along y arm in Earth-centered frame'))

    lines.append("/** @} */")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(detectors, arm_len=None):
    lines = ["/* --- Sanity checks ---"]
    for d in detectors:
        xn = np.linalg.norm(d["x_dir"])
        yn = np.linalg.norm(d["y_dir"])
        if xn == 0 or yn == 0:
            lines.append(f"   {d['prefix']}: null stream (arm vectors zeroed)")
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(d["x_dir"], d["y_dir"]), -1, 1)))
        lines.append(
            f"   {d['prefix']}: |x_dir|={xn:.6f}  |y_dir|={yn:.6f}  opening_angle={angle:.4f} deg"
        )
        
    if len(detectors) > 1 and arm_len is not None:
        lines.append("   --- Arm lengths ---")
        for i in range(len(detectors)):
            if detectors[i]["prefix"].endswith("0"): # skip null stream
                continue
            d1 = detectors[i]
            d2 = detectors[(i + 1) % len(detectors)]
            if d2["prefix"].endswith("0"):
                d2 = detectors[0]
            if d1 == d2:
                continue
            dist = np.linalg.norm(d1["vertex"] - d2["vertex"])
            lines.append(f"   {d1['prefix']} -> {d2['prefix']}: {dist:.2f} m (specified: {arm_len:.2f} m)")
    lines.append("*/")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath        = sys.argv[1]
    prefix_override = sys.argv[2] if len(sys.argv) > 2 else None

    params    = parse_bilby_file(filepath)
    params    = validate_params(params)
    detectors = build_detectors(params, prefix_override)

    shape = params['shape']
    name  = params['name']
    lat   = params['latitude']
    lon   = params['longitude']
    elev  = params['elevation']

    print(f"/**")
    print(f" * \\name {name} detector constants")
    print(f" * Generated by bilby_to_lal_detector.py from '{filepath}'")
    print(f" * Shape: {shape}")
    print(f" * Vertex: lat={lat:.6f} deg, lon={lon:.6f} deg, elev={elev} m")
    if shape == 'Triangle':
        print(f" * Sub-detectors rotated by 0, 120, 240 deg; null stream has zeroed arms.")
    print(f" */")
    print()

    for d in detectors:
        print(detector_to_defines(d))
        print()

    arm_len = params['length'] * 1000.0 if params['length'] is not None else None
    print(sanity_check(detectors, arm_len))


if __name__ == "__main__":
    main()