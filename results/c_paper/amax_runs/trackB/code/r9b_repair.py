"""Repair the downloaded 437k VCF without touching the original file.
Diagnosis: each byte-range part was built by appending curl bodies; failed requests appended an HTML error page and
the next request resumed at (part start + bytes held), so every HTML block of length L sits exactly at the file
offsets whose ORIGINAL bytes were skipped. Repair = copy the file, and for every HTML block [a, b) fetch original bytes
[a, b) with an HTTP range request and write them in place. Integrity is then verified line by line.
Usage (from runs/trackB): python code/r9b_repair.py
Writes cache/inbreds_437k_repaired.vcf and out/vcf_repair_report.json.
"""
import os, sys, json, time, mmap, shutil, hashlib, urllib.request
BASE = os.environ.get("R9B_BASE", "<server>/egp_gxe"); RUN = f"{BASE}/runs/trackB"
SRC = f"{BASE}/data/g2f_inbred_437k/inbreds_G2F_2014-2023_437k.vcf"; DST = f"{RUN}/cache/inbreds_437k_repaired.vcf"
URL = "https://data.cyverse.org/dav-anon/iplant/projects/commons_repo/curated/GenomesToFields_G2F_genotypic_data_2014_to_2023/inbreds_G2F_2014-2023_437k.vcf"
SIZE = 3852860306
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
assert os.path.getsize(SRC) == SIZE
t0 = time.time()
with open(SRC, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    blocks = []; i = 0
    while True:
        a = mm.find(b"<!DOCTYPE html>", i)
        if a < 0: break
        b = mm.find(b"</html>", a); assert b > a; b += len(b"</html>"); blocks.append((a, b)); i = b
    stray = 0; j = 0
    while True:                                   # '<html' not inside a located block
        k = mm.find(b"<html", j)
        if k < 0: break
        if not any(a <= k < b for a, b in blocks): stray += 1
        j = k + 1
    mm.close()
log(f"HTML blocks {len(blocks)}, bytes {sum(b - a for a, b in blocks)}, stray '<html' {stray}")
assert stray == 0
shutil.copyfile(SRC, DST + ".tmp")
fetched = 0
with open(DST + ".tmp", "r+b") as out:
    for n, (a, b) in enumerate(blocks):
        for attempt in range(30):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206: raise IOError(f"status {r.status}")
                    data = r.read()
                if len(data) != b - a or b"<html" in data or b"DOCTYPE" in data: raise IOError(f"bad body len {len(data)}")
                out.seek(a); out.write(data); fetched += len(data); break
            except Exception as e:
                time.sleep(min(60, 3 * (attempt + 1)))
        else:
            raise SystemExit(f"range {a}-{b} failed after retries")
        if n % 50 == 0: log(f"repaired {n + 1}/{len(blocks)}")
os.replace(DST + ".tmp", DST)
# ---- verification
n_hdr = n_var = bad = 0; nsamp = None; last = {}; unsorted = 0; h = hashlib.sha256()
with open(DST, "rb") as f:
    for raw in f:
        h.update(raw); line = raw.decode("ascii", errors="replace")
        if line.startswith("##"): n_hdr += 1; continue
        if line.startswith("#CHROM"): nsamp = len(line.rstrip("\n").split("\t")) - 9; continue
        p = line.rstrip("\n").split("\t")
        ok = len(p) == 9 + nsamp and p[1].isdigit() and "html" not in line
        if not ok: bad += 1; continue
        n_var += 1; c, pos = p[0], int(p[1])
        if c in last and pos < last[c]: unsorted += 1
        last[c] = pos
rep = dict(source_size=SIZE, html_blocks=len(blocks), html_bytes=sum(b - a for a, b in blocks), bytes_refetched=fetched,
           header_lines=n_hdr, samples=nsamp, variant_lines=n_var, malformed_lines=bad, unsorted_positions=unsorted,
           expected_variant_lines=437214, intact=bool(bad == 0 and n_var == 437214 and unsorted == 0),
           sha256_repaired=h.hexdigest(), seconds=round(time.time() - t0))
json.dump(rep, open(f"{RUN}/out/vcf_repair_report.json", "w"), indent=1)
log(json.dumps(rep))
