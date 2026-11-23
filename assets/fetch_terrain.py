"""Download and cache real Rwandan terrain for the 3D viewer.

Elevation comes from the free AWS Terrain Tiles API (terrarium encoding) and
imagery from the free Esri World Imagery API. Default location: farmland near
Musanze, Rwanda. Run once: `uv run python assets/fetch_terrain.py`.
"""

import io
import json
import math
import sys
from pathlib import Path

import requests
from PIL import Image

ELEVATION_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

MUSANZE = (-1.478, 29.564)
OUT_DIR = Path(__file__).resolve().parent / "viewer" / "terrain"


def tile_coords(lat, lon, zoom):
    # Convert lat/lon to slippy-map tile indices.
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def stitch(url_template, zoom, x0, y0, tiles, label):
    # Download a square block of map tiles and stitch them into one image.
    mosaic = Image.new("RGB", (256 * tiles, 256 * tiles))
    for dx in range(tiles):
        for dy in range(tiles):
            url = url_template.format(z=zoom, x=x0 + dx, y=y0 + dy)
            print(f"  {label} tile z{zoom} ({x0 + dx},{y0 + dy})")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            mosaic.paste(Image.open(io.BytesIO(resp.content)).convert("RGB"), (dx * 256, dy * 256))
    return mosaic


def main():
    # Fetch elevation + satellite mosaics around the target coordinates.
    lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else MUSANZE
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    elev_zoom, elev_tiles = 13, 2
    x, y = tile_coords(lat, lon, elev_zoom)
    x0, y0 = x - elev_tiles // 2, y - elev_tiles // 2
    print(f"Fetching elevation around ({lat}, {lon})...")
    stitch(ELEVATION_URL, elev_zoom, x0, y0, elev_tiles, "elevation").save(OUT_DIR / "terrain.png")

    sat_zoom = elev_zoom + 2
    scale = 2 ** (sat_zoom - elev_zoom)
    print("Fetching satellite imagery...")
    sat = stitch(IMAGERY_URL, sat_zoom, x0 * scale, y0 * scale, elev_tiles * scale, "imagery")
    sat.resize((2048, 2048)).save(OUT_DIR / "satellite.jpg", quality=88)

    meta = {"lat": lat, "lon": lon, "zoom": elev_zoom, "tiles": elev_tiles,
            "encoding": "terrarium", "place": "Musanze, Rwanda"}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved terrain to {OUT_DIR}")


if __name__ == "__main__":
    main()
