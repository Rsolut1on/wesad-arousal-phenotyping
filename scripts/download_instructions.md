# WESAD Dataset Download Instructions

This repository **does not include raw WESAD participant data**. You must obtain the dataset separately and place it locally before running the pipeline.

## Official Sources

- UCI Machine Learning Repository: [WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
- Original publication: Schmidt et al., 2018 — *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*

## Expected Local Layout

After downloading and extracting WESAD, place subject folders under:

```text
data/raw/WESAD/
  S2/
    S2.pkl
    S2_readme.txt
    ...
  S3/
  ...
```

Each subject directory must contain a `{subject_id}.pkl` file with synchronized chest (RespiBAN) and wrist (Empatica E4) signals and protocol labels.

## Quick Setup

```bash
mkdir -p data/raw/WESAD
# Copy or symlink your downloaded WESAD subject folders into data/raw/WESAD/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data\raw\WESAD
# Copy extracted WESAD folders into data\raw\WESAD\
```

If you already have WESAD elsewhere on disk, you can pass `--raw-dir` to preprocessing:

```bash
python scripts/run_preprocess.py --config configs/default.yaml --raw-dir /path/to/WESAD
```

## Governance Notes

- Do **not** commit raw `.pkl` files or derived feature tables to version control.
- Use subject-independent evaluation (leave-one-subject-out) when reporting performance.
- Cite the original WESAD paper when publishing results based on this pipeline.
