English translation of the Chinese strings in the C-line pre-registration JSON files results/c_paper/workspace/prereg/*.json (original strings in Chinese). The Chinese original is authoritative; this translation was prepared on 2026-10-02 for the Supplementary Information. Internal workflow references (local directory names, working-session names, the name of the internal rules document and journal-planning remarks) have been neutralised in this translation; the Chinese originals are unchanged and their hashes are listed in MANIFEST.md.

Scope: all 10 JSON files in `results/c_paper/workspace/prereg/` were scanned for CJK characters (keys and values). Eight files contain no Chinese text: `prereg_v10_final_confirmation.json`, `prereg_v3_briwecs_oracle.json`, `prereg_v4_nonlinear_ceiling.json`, `prereg_v5_C_L4_pilot.json`, `prereg_v6_level1_engineering.json`, `prereg_v7_confirmatory_C_H3.json`, `prereg_v8_P3_default.json`, `prereg_v9_mainline_external.json`. Two files contain Chinese text: `prereg_AB_gonogo.json` and `prereg_v2_remediation.json`. No JSON key contains Chinese text.

In every case the only Chinese text is the local directory name `<lab-data>` inside an absolute input-file path. `<lab-data>` means "PhD group meeting". In the English rendering below, only that directory name is translated; the rest of each path is kept exactly as in the original. The JSON files were not modified.

## prereg_AB_gonogo.json

| Key path | English rendering |
|---|---|
| `$.inputs_g2f.1_Training_Trait_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/1_Training_Trait_Data_2014_2021.csv` |
| `$.inputs_g2f.2_Training_Meta_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/2_Training_Meta_Data_2014_2021.csv` |
| `$.inputs_g2f.3_Training_Soil_Data_2015_2021.csv.path` | `<local>/gp_project/data/raw/g2f/3_Training_Soil_Data_2015_2021.csv` |
| `$.inputs_g2f.4_Training_Weather_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/4_Training_Weather_Data_2014_2021.csv` |
| `$.inputs_g2f.5_Genotype_Data_All_2014_2025_Hybrids.vcf.path` | `<local>/gp_project/data/raw/g2f/5_Genotype_Data_All_2014_2025_Hybrids.vcf` |
| `$.inputs_g2f.6_Training_EC_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/6_Training_EC_Data_2014_2021.csv` |

## prereg_v2_remediation.json

| Key path | English rendering |
|---|---|
| `$.depends_on.g2f_inputs.1_Training_Trait_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/1_Training_Trait_Data_2014_2021.csv` |
| `$.depends_on.g2f_inputs.2_Training_Meta_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/2_Training_Meta_Data_2014_2021.csv` |
| `$.depends_on.g2f_inputs.3_Training_Soil_Data_2015_2021.csv.path` | `<local>/gp_project/data/raw/g2f/3_Training_Soil_Data_2015_2021.csv` |
| `$.depends_on.g2f_inputs.4_Training_Weather_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/4_Training_Weather_Data_2014_2021.csv` |
| `$.depends_on.g2f_inputs.5_Genotype_Data_All_2014_2025_Hybrids.vcf.path` | `<local>/gp_project/data/raw/g2f/5_Genotype_Data_All_2014_2025_Hybrids.vcf` |
| `$.depends_on.g2f_inputs.6_Training_EC_Data_2014_2021.csv.path` | `<local>/gp_project/data/raw/g2f/6_Training_EC_Data_2014_2021.csv` |
