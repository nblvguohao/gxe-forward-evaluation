"""Wave 2e input preparation (run once where openpyxl is installed): ESWYT Table S3 (sheet 'Table S3' of the
Frontiers supplementary workbook, header on the second row) -> TableS3_ESWYT_BLUEs.csv next to it. The 4090
environment has no openpyxl; the CSV and both SHA256 sums are recorded in NEWDATA/SHA256SUMS_eswyt.txt."""
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
d = pd.read_excel(src, sheet_name="Table S3", header=1)
d.to_csv(src.parent / "TableS3_ESWYT_BLUEs.csv", index=False)
print(d.shape)
