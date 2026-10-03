"""Fetch NASA POWER daily weather for the 2024 G2F test environments (gate-2 amendment 2).

Why: the released 4_Testing_Weather_Data_2024_full_year.csv is complete only to 2024-07-01. The G2F data
note states the weather was downloaded from power.larc.nasa.gov; same parameters, weather-station
coordinates from 2_Testing_Meta_Data_2024.csv. Saves raw JSON per env, a combined CSV and a manifest.
"""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "g2f2024"
OUT = RAW / "power_2024"
OUT.mkdir(exist_ok=True)
P = "RH2M,T2M_MAX,ALLSKY_SFC_SW_DWN,T2MWET,GWETTOP,QV2M,GWETPROF,T2M_MIN,T2MDEW,PS,T2M,GWETROOT,ALLSKY_SFC_PAR_TOT,WS2M,ALLSKY_SFC_SW_DNI,PRECTOTCORR"
m = pd.read_csv(RAW / "2_Testing_Meta_Data_2024.csv")
lat_c = "Weather_Station_Latitude (in decimal numbers NOT DMS)"
lon_c = "Weather_Station_Longitude (in decimal numbers NOT DMS)"
rows, manifest = [], {"fetched": time.strftime("%Y-%m-%d %H:%M:%S %z"), "source": "https://power.larc.nasa.gov/api/temporal/daily/point", "envs": {}}
for _, r in m.iterrows():
    e, lat, lon, coord = r["Env"], r[lat_c], r[lon_c], "weather_station"
    if pd.isna(lat) or pd.isna(lon):  # no station coordinates: centre of the four field corners
        lat = pd.Series([r[f"Latitude_of_Field_Corner_#{k} ({s})"] for k, s in ((1, "lower left"), (2, "lower right"), (3, "upper right"), (4, "upper left"))]).mean()
        lon = pd.Series([r[f"Longitude_of_Field_Corner_#{k} ({s})"] for k, s in ((1, "lower left"), (2, "lower right"), (3, "upper right"), (4, "upper left"))]).mean()
        coord = "field_corner_centre"
    lat, lon = round(float(lat), 6), round(float(lon), 6)
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={P}&community=AG"
           f"&longitude={lon}&latitude={lat}&start=20240101&end=20241231&format=JSON")
    f = OUT / f"{e}.json"
    for attempt in range(5):
        try:
            if not f.exists():
                f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
            j = json.loads(f.read_text())
            break
        except Exception as ex:  # retry transient network errors
            f.unlink(missing_ok=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"failed {e}")
    d = pd.DataFrame(j["properties"]["parameter"])
    d.index.name = "Date"
    d = d.reset_index()
    d.insert(0, "Env", e)
    rows.append(d)
    manifest["envs"][e] = {"lat": lat, "lon": lon, "coord_source": coord, "url": url, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    print(e, len(d), flush=True)
w = pd.concat(rows, ignore_index=True)
w = w.replace(-999.0, float("nan"))
w.to_csv(OUT / "power_2024_daily.csv", index=False)
manifest["combined_sha256"] = hashlib.sha256((OUT / "power_2024_daily.csv").read_bytes()).hexdigest()
manifest["nan_share"] = float(w.drop(columns=["Env", "Date"]).isna().mean().mean())
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("envs", w.Env.nunique(), "rows", len(w), "NaN share", manifest["nan_share"])
