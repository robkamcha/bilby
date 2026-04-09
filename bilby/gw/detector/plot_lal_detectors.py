#!/usr/bin/env python3
"""
plot_lal_detectors.py
---------------------
Convert one or more bilby detector definition files to an interactive map,
by calling bilby_to_lal_detector.py's conversion logic directly.

Both scripts must live in the same directory.

Usage:
    python plot_lal_detectors.py <detector1.cfg> [detector2.cfg ...] [-o map.html]
                                 [--arm-scale FLOAT]

    detector.cfg  -- bilby detector definition file (L-shaped or Triangle)
    -o            -- output HTML file (default: lal_detectors_map.html)
    --arm-scale   -- display arm length = midpoint * arm_scale (default: 3.0)
                     increase if arms are too short to see on the map

Output:
    A self-contained HTML file using Leaflet.js (loaded from CDN).  Open it
    in any browser — no server required.
"""

import sys
import math
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Import the converter — both scripts live in the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from generate_lal_detector import parse_bilby_file, validate_params, build_detectors


# ---------------------------------------------------------------------------
# Geometry: ECEF unit vector -> (dlat, dlon) offset at a given distance
# ---------------------------------------------------------------------------

def ecef_to_latlon_delta(lat_deg, lon_deg, ecef_unit, distance_m):
    """
    Project an ECEF unit vector onto the local ENU plane and return the
    (dlat_deg, dlon_deg) offset corresponding to `distance_m` metres.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    ux, uy, uz = ecef_unit

    east  = (-math.sin(lon),  math.cos(lon),  0.0)
    north = (-math.sin(lat)*math.cos(lon),
             -math.sin(lat)*math.sin(lon),
              math.cos(lat))

    e = ux*east[0]  + uy*east[1]  + uz*east[2]
    n = ux*north[0] + uy*north[1] + uz*north[2]

    # Use the same great-circle math and Earth radius as the generator for exact alignment
    brng_r = math.atan2(e, n)
    R_EARTH = 6378136.6
    
    phi1 = lat
    lam1 = lon
    d = distance_m
    
    phi2 = math.asin(
        math.sin(phi1) * math.cos(d / R_EARTH) +
        math.cos(phi1) * math.sin(d / R_EARTH) * math.cos(brng_r)
    )
    lam2 = lam1 + math.atan2(
        math.sin(brng_r) * math.sin(d / R_EARTH) * math.cos(phi1),
        math.cos(d / R_EARTH) - math.sin(phi1) * math.sin(phi2)
    )

    dlat = math.degrees(phi2) - lat_deg
    dlon = math.degrees(lam2) - lon_deg
    return dlat, dlon


# ---------------------------------------------------------------------------
# Convert build_detectors() output into map-ready dicts
# ---------------------------------------------------------------------------

COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
]


def detectors_to_map_data(all_groups, arm_scale):
    """
    all_groups: list of (shape, [detector_dict, ...]) tuples
                where shape is 'L' or 'Triangle' and each detector_dict
                is the format returned by build_detectors().

    Returns (det_list, link_list) for embedding in the Leaflet HTML.
    """
    det_list  = []
    link_list = []

    for color_idx, (shape, group) in enumerate(all_groups):
        color = COLORS[color_idx % len(COLORS)]

        # Collect real (non-null-stream) members for the linking polygon
        real_members = [d for d in group if d['midpoint_m'] > 0]

        for d in group:
            lat     = math.degrees(d['lat_rad'])
            lon     = math.degrees(d['lon_rad'])
            midpt   = d['midpoint_m']
            arm_len = midpt * arm_scale

            arms = []
            
            # For Triangle geometries, Bilby's spherical coordinate generation creates a small ~16m gap 
            # where the triangle doesn't technically close. To draw perfectly aligned lines in Leaflet,
            # we can snap the arm vectors directly along the geometric paths to adjacent vertices.
            is_triangle = (shape == 'Triangle' and len(real_members) >= 3 and midpt > 0)
            
            if is_triangle:
                real_idx = real_members.index(d)
                target_x = real_members[(real_idx + 1) % len(real_members)]
                target_y = real_members[(real_idx - 1) % len(real_members)]
                
                fraction = arm_scale / 2.0  # (midpt * 2.0 is full arm length)
                
                arms.append({
                    'dlat': (math.degrees(target_x['lat_rad']) - lat) * fraction,
                    'dlon': (math.degrees(target_x['lon_rad']) - lon) * fraction,
                    'label': 'x', 'weight': 2.5, 'dash': None
                })
                arms.append({
                    'dlat': (math.degrees(target_y['lat_rad']) - lat) * fraction,
                    'dlon': (math.degrees(target_y['lon_rad']) - lon) * fraction,
                    'label': 'y', 'weight': 1.5, 'dash': '5 4'
                })
            else:
                for key, label, weight, dash in [
                        ('x_dir', 'x', 2.5, None),
                        ('y_dir', 'y', 1.5, '5 4')]:
                    vec = d.get(key)
                    if vec is None:
                        arms.append(None)
                        continue
                    norm = math.sqrt(sum(v**2 for v in vec))
                    if norm < 1e-9 or arm_len == 0:
                        arms.append(None)
                        continue
                    uv = tuple(v / norm for v in vec)
                    dlat, dlon = ecef_to_latlon_delta(lat, lon, uv, arm_len)
                    arms.append({'dlat': dlat, 'dlon': dlon,
                                 'label': label, 'weight': weight,
                                 'dash': dash})

            det_list.append({
                'prefix': d['prefix'],
                'name':   d['name'],
                'lat':    lat,
                'lon':    lon,
                'elev':   d['elev_m'],
                'midpt':  midpt,
                'color':  color,
                'arms':   arms,
            })

        # Connecting polygon for Triangle detectors
        if shape == 'Triangle' and len(real_members) > 1:
            coords = [[math.degrees(d['lat_rad']), math.degrees(d['lon_rad'])]
                      for d in real_members]
            coords.append(coords[0])   # close the loop
            link_list.append({'coords': coords, 'color': color})

    return det_list, link_list


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Bilby Detector Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ display: flex; height: 100vh; font-family: 'Segoe UI', sans-serif; }}
    #sidebar {{
      width: 280px; min-width: 200px; background: #1a1a2e; color: #eee;
      overflow-y: auto; padding: 16px; flex-shrink: 0;
    }}
    #sidebar h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
                   color: #888; margin-bottom: 12px; }}
    .det-card {{
      background: #16213e; border-radius: 6px; padding: 10px 12px;
      margin-bottom: 8px; border-left: 4px solid var(--c);
    }}
    .det-card .pfx   {{ font-size: 15px; font-weight: 700; color: var(--c); }}
    .det-card .dname {{ font-size: 11px; color: #888; margin-top: 2px;
                        word-break: break-all; }}
    .det-card .info  {{ font-size: 11px; color: #ccc; margin-top: 5px;
                        line-height: 1.6; }}
    #map {{ flex: 1; }}
    .ifo-label {{
      background: transparent; border: none; box-shadow: none;
      font-weight: 700; font-size: 13px; color: #fff;
      text-shadow: 0 0 4px #000, 0 0 4px #000;
    }}
    .legend {{
      background: rgba(20,20,40,0.88); color: #ddd;
      padding: 8px 12px; border-radius: 6px;
      font-size: 12px; line-height: 2;
    }}
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>Detectors</h2>
    {sidebar_html}
  </div>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
  var map = L.map('map').setView({center}, {zoom});

  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19
  }}).addTo(map);

  {detector_json}.forEach(function(d) {{
    L.circleMarker([d.lat, d.lon], {{
      radius: 7, color: d.color, fillColor: d.color, fillOpacity: 0.9, weight: 2
    }}).addTo(map).bindPopup(
      '<b>' + d.prefix + '</b><br>'
      + '<span style="color:#aaa;font-size:11px">' + d.name + '</span><br>'
      + 'lat ' + d.lat.toFixed(5) + '°,  lon ' + d.lon.toFixed(5) + '°'
      + (d.elev  ? '<br>elev ' + d.elev.toFixed(1) + ' m' : '')
      + (d.midpt ? '<br>arm ½-len ' + (d.midpt/1000).toFixed(2) + ' km' : '')
    );

    L.marker([d.lat, d.lon], {{
      icon: L.divIcon({{ className:'ifo-label', html: d.prefix, iconAnchor:[-10,6] }}),
      interactive: false
    }}).addTo(map);

    d.arms.forEach(function(arm) {{
      if (!arm) return;
      L.polyline([[d.lat, d.lon], [d.lat+arm.dlat, d.lon+arm.dlon]], {{
        color: d.color, weight: arm.weight, opacity: 0.9,
        dashArray: arm.dash || null
      }}).addTo(map).bindTooltip(d.prefix + ' ' + arm.label + '-arm');
    }});
  }});

  {link_json}.forEach(function(lk) {{
    L.polyline(lk.coords, {{
      color: lk.color, weight: 1.2, opacity: 0.4, dashArray: '6 4'
    }}).addTo(map);
  }});

  var leg = L.control({{position:'bottomright'}});
  leg.onAdd = function() {{
    var d = L.DomUtil.create('div','legend');
    d.innerHTML = [
      '<span style="display:inline-block;width:24px;height:3px;background:#fff;vertical-align:middle;margin-right:6px"></span>x-arm',
      '<span style="display:inline-block;width:24px;border-top:2px dashed #fff;vertical-align:middle;margin-right:6px;opacity:.5"></span>y-arm',
      '<span style="display:inline-block;width:24px;border-top:2px dashed #aaa;vertical-align:middle;margin-right:6px;opacity:.5"></span>triangle link'
    ].join('<br>');
    return d;
  }};
  leg.addTo(map);
  </script>
</body>
</html>
"""


def build_html(all_groups, arm_scale):
    det_list, link_list = detectors_to_map_data(all_groups, arm_scale)

    if not det_list:
        raise ValueError("No detectors to display.")

    # Sidebar cards
    sidebar = []
    for d in det_list:
        info = f'lat {d["lat"]:+.5f}°  lon {d["lon"]:+.5f}°'
        if d['elev']:
            info += f'<br>elev {d["elev"]:.0f} m'
        if d['midpt']:
            info += f'<br>arm ½-len {d["midpt"]/1000:.1f} km'
        sidebar.append(
            f'<div class="det-card" style="--c:{d["color"]}">'
            f'<div class="pfx">{d["prefix"]}</div>'
            f'<div class="dname">{d["name"]}</div>'
            f'<div class="info">{info}</div>'
            f'</div>'
        )

    lats   = [d['lat'] for d in det_list]
    lons   = [d['lon'] for d in det_list]
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]
    span   = max(max(lats)-min(lats), max(lons)-min(lons))
    zoom   = 3 if span > 30 else 4 if span > 5 else 6

    return HTML_TEMPLATE.format(
        sidebar_html  = '\n    '.join(sidebar),
        center        = json.dumps(center),
        zoom          = zoom,
        detector_json = 'var _d=' + json.dumps(det_list) + '; _d',
        link_json     = 'var _l=' + json.dumps(link_list) + '; _l',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('configs', nargs='+',
                    help='Bilby detector definition file(s)')
    ap.add_argument('-o', '--output', default='lal_detectors_map.html',
                    help='Output HTML file (default: lal_detectors_map.html)')
    ap.add_argument('--prefix', default=None,
                    help='LAL prefix override (forwarded to build_detectors)')
    ap.add_argument('--arm-scale', type=float, default=2.0,
                    help='Display arm length = midpoint × arm_scale (default: 2.0)')
    args = ap.parse_args()

    all_groups = []   # list of (shape, [detector_dict, ...])

    for path in args.configs:
        try:
            params    = parse_bilby_file(path)
            params    = validate_params(params)
            detectors = build_detectors(params, args.prefix)
            all_groups.append((params['shape'], detectors))
            print(f"Loaded '{path}': shape={params['shape']}, "
                  f"{len(detectors)} sub-detector(s) "
                  f"({', '.join(d['prefix'] for d in detectors)})")
        except Exception as exc:
            print(f"Error loading '{path}': {exc}", file=sys.stderr)
            sys.exit(1)

    html = build_html(all_groups, args.arm_scale)
    out  = Path(args.output)
    out.write_text(html)
    print(f"Map written to: {out.resolve()}")


if __name__ == '__main__':
    main()