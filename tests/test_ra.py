"""Checks for the RA input builder (docs/prereg_reanalysis_published_2026-09-30.md §5.2): no phenotype of the target
year or later is written; training tables stop before Y; the template lists the benchmark scored cells only."""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd


def make_raw(tmp: Path):
    raw = tmp / "raw"
    raw.mkdir()
    rows = []
    for y in (2014, 2015, 2016, 2017):
        for e in ("IAH1", "NEH1"):
            for h in ("A/X", "B/X", "C/X"):
                rows.append({"Env": f"{e}_{y}", "Year": y, "Hybrid": h, "Yield_Mg_ha": 1000.0 + y, "Range": 1, "Pass": 1})
    pd.DataFrame(rows).to_csv(raw / "1_Training_Trait_Data_2014_2023.csv", index=False)
    envs = sorted({r["Env"] for r in rows})
    meta = pd.DataFrame({"Year": [int(e[-4:]) for e in envs], "Env": envs, "Treatment": ["Irrigated", "Standard"] * 4,
                         "Previous_Crop": "soybean"})
    meta.to_csv(raw / "2_Training_Meta_Data_2014_2023.csv", index=False)
    meta.iloc[:0].to_csv(raw / "2_Testing_Meta_Data_2024.csv", index=False)
    pd.DataFrame({"Year": meta["Year"], "Env": envs, "pH": 6.0}).to_csv(raw / "3_Training_Soil_Data_2015_2023.csv", index=False)
    pd.DataFrame({"Env": envs, "Date": 20150101, "T2M": 20.0}).to_csv(raw / "4_Training_Weather_Data_2014_2023_seasons_only.csv", index=False)
    pd.DataFrame({"Env": envs, "EC1": 1.0}).to_csv(raw / "6_Training_EC_Data_2014_2023.csv", index=False)
    (raw / "5_Genotype_Data_All_2014_2025_Hybrids.vcf").write_text("#CHROM\n")
    bench = tmp / "bench"
    bench.mkdir()
    cells = pd.DataFrame(rows).rename(columns={"Env": "env", "Year": "year", "Hybrid": "genotype", "Yield_Mg_ha": "y"})
    cells[["env", "year", "genotype", "y"]].to_parquet(bench / "cells_G2F.parquet")
    json.dump({"G2F": {"scorable_environments": {"2016": ["IAH1_2016"]}}}, open(bench / "splits.json", "w"))
    return raw, bench


def test_builder_writes_no_target_or_later_phenotype(tmp_path, monkeypatch):
    raw, bench = make_raw(tmp_path)
    out = tmp_path / "ra"
    monkeypatch.setenv("RAW", str(raw))
    monkeypatch.setenv("BENCH", str(bench))
    monkeypatch.setenv("OUT", str(out))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    mod = importlib.import_module("ra_build_inputs")
    importlib.reload(mod)
    mod.main(["2016"])
    d = out / "Y2016"
    tr = pd.read_csv(d / "Training_Data" / mod.N["trait"])
    assert tr["Year"].max() == 2015 and not (tr["Yield_Mg_ha"] >= 2016 + 1000).any()
    for f in d.rglob("*.csv"):
        text = f.read_text()
        assert "3016.0" not in text and "3017.0" not in text, f     # yields of 2016/2017 (1000 + year) never written
    tpl = pd.read_csv(d / "Testing_Data" / mod.N["tpl"])
    assert set(tpl["Env"]) == {"IAH1_2016"} and tpl["Yield_Mg_ha"].isna().all() and len(tpl) == 3
    tm = pd.read_csv(d / "Testing_Data" / mod.N["tmeta"])
    assert list(tm["Irrigated"]) == ["yes"]
    assert set(pd.read_csv(d / "Training_Data" / mod.N["ec"])["Env"].str[-4:].astype(int)) == {2014, 2015}
