# Portfolio Repo Spec for Cursor: WESAD Stress and Arousal Phenotyping Pipeline

## Recommended Repository Title

`wesad-arousal-phenotyping`

## One-Line Positioning

A reproducible multimodal wearable physiology pipeline for daytime stress and autonomic arousal phenotyping, built on WESAD as a technical demonstrator for sleep, stress and arousal research workflows.

## Why This Repo Fits the ETH PhD Position

The position emphasizes human sleep, stress, arousal, multimodal biosignals, mobile health, decentralized phenotyping, preprocessing, synchronization, quality control, feature extraction, statistical modeling, machine learning, and physiological interpretation.

WESAD does not contain sleep recordings, so the repo should not claim to study sleep directly. Instead, frame it as a daytime stress/arousal pipeline that could be extended to sleep and home-based phenotyping studies. This is stronger and more scientifically honest.

Core message:

> This project demonstrates my ability to build robust, reproducible computational workflows for multimodal physiological data, including synchronization, artifact handling, quality control, feature extraction, visualization, subject-independent modeling, and interpretable physiological reporting.

## Target README Abstract

This repository implements a reproducible pipeline for multimodal wearable stress and arousal phenotyping using the WESAD dataset. It processes wrist- and chest-worn physiological signals including ECG, respiration, EDA, BVP, temperature, EMG, and accelerometry; performs signal quality control and windowed feature extraction; evaluates subject-independent stress and affect classification; and generates interpretable visual reports linking autonomic and behavioral physiology to stress and arousal states.

The project is designed as a research-facing prototype for mobile and decentralized phenotyping workflows relevant to sleep, stress, arousal, and recovery studies. It intentionally does not include raw WESAD data and requires users to obtain the dataset from the original source.

## Scientific Question

Primary question:

> Can multimodal wearable physiology provide subject-independent signatures of daytime stress and arousal states, and which sensor modalities contribute most robustly?

Secondary questions:

1. How do wrist-only and chest-only sensor pipelines compare?
2. Which physiological feature families are most informative: HRV, EDA, respiration, temperature, motion, or BVP?
3. How stable are stress predictions under leave-one-subject-out evaluation?
4. Can we build transparent QC reports that would be useful in real-world and home-based studies?

## Dataset

Use WESAD:

- 15 subjects
- Wrist and chest devices
- Conditions: baseline, stress, amusement
- Signals:
  - ECG
  - EDA
  - EMG
  - respiration
  - BVP
  - body temperature
  - three-axis acceleration
  - self-report questionnaires

Important:

- Do not commit raw WESAD data.
- Add `data/raw/` to `.gitignore`.
- Provide `scripts/download_instructions.md` explaining that users must obtain WESAD from the official dataset source.
- Include tiny synthetic toy data only for tests.

## Strong Repo Outcome

The repo should look like a small but serious research software project, not just a notebook.

It should include:

- Clean README with motivation, dataset, pipeline diagram, results table, and example figures
- `requirements.txt` or `environment.yml`
- Config-driven preprocessing and modeling
- CLI scripts for reproducibility
- Notebooks for exploration and visualization
- Unit tests for core feature functions
- Subject-independent evaluation
- QC reports per subject
- Model interpretation
- Clear statement that raw data are excluded

## Proposed Repository Structure

```text
wesad-arousal-phenotyping/
  README.md
  LICENSE
  .gitignore
  requirements.txt
  environment.yml
  pyproject.toml
  configs/
    default.yaml
    sensors_chest.yaml
    sensors_wrist.yaml
  data/
    raw/                  # ignored; user places WESAD here
    interim/              # ignored
    processed/            # ignored
  notebooks/
    01_dataset_overview.ipynb
    02_signal_qc_examples.ipynb
    03_feature_visualization.ipynb
    04_model_interpretation.ipynb
  reports/
    figures/
    tables/
    example_qc_report.md
  scripts/
    run_preprocess.py
    run_extract_features.py
    run_train.py
    run_evaluate.py
    make_report.py
    download_instructions.md
  src/
    wesad_arousal/
      __init__.py
      config.py
      data.py
      labels.py
      preprocessing.py
      quality.py
      features/
        __init__.py
        hrv.py
        eda.py
        respiration.py
        motion.py
        temperature.py
        windowing.py
      modeling/
        __init__.py
        baselines.py
        evaluation.py
        interpretation.py
      visualization.py
      reporting.py
  tests/
    test_windowing.py
    test_hrv_features.py
    test_eda_features.py
    test_quality.py
  outputs/
    .gitkeep
```

## Pipeline Design

### 1. Data Ingestion

Implement a loader for WESAD subject `.pkl` files.

Expected command:

```bash
python scripts/run_preprocess.py --config configs/default.yaml --raw-dir data/raw/WESAD --out-dir data/interim
```

Functions:

- Load each subject
- Extract wrist and chest signals
- Extract labels
- Align signals by timestamps or sample indices according to WESAD metadata
- Convert into standardized long-format or window-ready arrays

Suggested internal representation:

```text
subject_id | device | signal | timestamp | value_x | value_y | value_z | label
```

or separate parquet files by subject and signal.

### 2. Label Handling

Use three main states:

- baseline
- stress
- amusement

Also create binary labels:

- stress
- non-stress

Exclude undefined/transient labels unless explicitly analyzing transitions.

Critical point:

Use subject-wise splits only. Do not randomly split windows across subjects, because that causes identity leakage and inflated performance.

### 3. Signal Quality Control

Create automated QC metrics for each subject and sensor:

- Missingness
- Flatline segments
- Out-of-range values
- Abrupt jumps
- Signal coverage per condition
- Motion intensity summary
- ECG peak detection success rate
- EDA valid range percentage
- Respiration signal plausibility
- Window exclusion counts

Output:

```bash
python scripts/make_report.py --qc outputs/qc_metrics.csv --out reports/example_qc_report.md
```

README should show one small QC table and one example figure.

### 4. Preprocessing

Keep preprocessing transparent and conservative.

Suggested methods:

- ECG:
  - bandpass filtering
  - R-peak detection with NeuroKit2
  - interbeat interval extraction
  - HRV time-domain and frequency-domain features
- EDA:
  - smoothing
  - tonic/phasic decomposition with NeuroKit2
  - skin conductance response count and amplitude
- BVP:
  - pulse rate and pulse variability features
- Respiration:
  - breathing rate
  - variability
  - amplitude/rhythm features
- Accelerometry:
  - vector magnitude
  - mean, standard deviation, energy
  - posture/motion proxy features
- Temperature:
  - mean
  - slope
  - variability
- EMG:
  - RMS
  - median absolute deviation
  - spectral energy if feasible

### 5. Windowing

Use sliding windows:

- default window length: 60 seconds
- default step: 30 seconds
- sensitivity analysis: 30s, 60s, 120s

Each window receives a label by majority vote. Drop windows with mixed labels above a threshold, for example if the majority label covers less than 80%.

### 6. Feature Extraction

Output one row per subject-window:

```text
subject_id | window_start | window_end | label | binary_label | features...
```

Feature groups:

- HRV features
- EDA features
- respiration features
- motion features
- temperature features
- BVP features
- EMG features
- signal quality features

Save:

```text
data/processed/features.parquet
data/processed/features.csv
```

Do not commit these files if derived data redistribution is not clearly allowed.

### 7. Modeling

Implement simple, interpretable baselines first:

- Logistic Regression
- Random Forest
- Gradient Boosting or XGBoost/LightGBM if available
- DummyClassifier baseline

Optional advanced model:

- time-series classifier on minimally processed windows, such as InceptionTime or a simple 1D CNN

But for the application portfolio, classical feature-based models are preferable because they are easier to interpret physiologically.

### 8. Evaluation

Use leave-one-subject-out cross-validation as the headline result.

Metrics:

- balanced accuracy
- macro F1
- ROC-AUC for binary stress detection
- confusion matrix
- per-subject performance distribution
- calibration curve for binary model

Compare:

1. Chest only
2. Wrist only
3. Chest + wrist
4. Modality ablation:
   - ECG/HRV
   - EDA
   - respiration
   - motion
   - temperature

README table template:

```text
| Model | Sensors | Task | CV | Balanced Accuracy | Macro F1 |
|---|---|---|---|---:|---:|
| Dummy baseline | all | 3-class | LOSO | TBD | TBD |
| Logistic Regression | chest | 3-class | LOSO | TBD | TBD |
| Random Forest | wrist | binary stress | LOSO | TBD | TBD |
| Gradient Boosting | chest+wrist | binary stress | LOSO | TBD | TBD |
```

### 9. Interpretation

Add a model interpretation section:

- Permutation importance
- SHAP summary if dependency is acceptable
- Feature group ablation
- Physiological interpretation:
  - stress expected to increase heart rate and sympathetic activation
  - EDA phasic activity may increase during stress
  - respiration rate and irregularity may shift
  - movement may confound wearable signals

Important:

Avoid overclaiming mechanisms. Say these are markers or signatures, not causal proof.

### 10. Visualizations

Figures for README:

1. Pipeline diagram
2. Example synchronized multimodal subject segment
3. QC heatmap by subject and modality
4. Feature distributions by condition
5. Confusion matrix
6. Sensor/modality ablation result
7. Feature importance plot

## Suggested Technology Stack

Python:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- neurokit2
- PyYAML
- pyarrow
- joblib
- tqdm
- pytest

Optional:

- shap
- xgboost or lightgbm
- plotly
- mne only if needed, but WESAD does not require EEG processing

## Minimal `requirements.txt`

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
neurokit2
PyYAML
pyarrow
joblib
tqdm
pytest
```

## Reproducible Commands

README should provide:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then:

```bash
python scripts/run_preprocess.py --config configs/default.yaml
python scripts/run_extract_features.py --config configs/default.yaml
python scripts/run_train.py --config configs/default.yaml --task binary
python scripts/run_evaluate.py --config configs/default.yaml
python scripts/make_report.py --config configs/default.yaml
```

## README Sections

Use this exact README outline:

```markdown
# WESAD Arousal Phenotyping

## Overview
## Why This Matters for Sleep, Stress and Arousal Research
## Dataset
## Pipeline
## Installation
## Reproducing the Analysis
## Quality Control
## Feature Extraction
## Modeling and Evaluation
## Results
## Physiological Interpretation
## Limitations
## Repository Structure
## Citation
```

## Limitations Section

Include this because it makes the repo look scientifically mature:

- WESAD is a small laboratory dataset with 15 subjects.
- The dataset contains daytime stress and affect states, not overnight sleep.
- Results may not transfer directly to home-based sleep studies.
- Subject-independent evaluation is essential because random window splits can overestimate performance.
- Wearable signals are affected by motion artifacts, device placement, and individual physiology.

## Good Project Name Alternatives

1. `wesad-arousal-phenotyping`
2. `wearable-stress-arousal-pipeline`
3. `multimodal-stress-physiology`
4. `wesad-mobile-phenotyping`

Best choice: `wesad-arousal-phenotyping`.

## Development Milestones for Cursor

### Milestone 1: Repo Scaffold

- Create repo structure
- Add README skeleton
- Add `.gitignore`
- Add requirements
- Add default YAML config
- Add placeholder reports and figures directories

### Milestone 2: WESAD Loader

- Implement subject `.pkl` loading
- Standardize labels
- Save interim per-subject files
- Add tests with synthetic toy data

### Milestone 3: QC and Windowing

- Implement window generation
- Implement label majority logic
- Implement basic signal QC metrics
- Generate QC CSV

### Milestone 4: Feature Extraction

- Implement ECG/HRV features
- Implement EDA features
- Implement respiration, motion, temperature, and BVP features
- Save feature matrix

### Milestone 5: Modeling

- Implement LOSO cross-validation
- Train dummy, logistic regression, random forest, gradient boosting
- Save metrics and confusion matrices

### Milestone 6: Interpretation and Reporting

- Add feature importance
- Add modality ablation
- Generate Markdown report
- Update README with example plots and result tables

## Cursor Prompt

Copy this into Cursor:

```text
You are helping me build a polished public GitHub portfolio repository for a PhD application in sleep, stress, arousal, human physiology, mobile health, and computational phenotyping.

Repository name: wesad-arousal-phenotyping

Goal:
Build a reproducible Python research pipeline using the WESAD dataset. The repo should demonstrate robust multimodal physiological data handling, preprocessing, synchronization, quality control, windowed feature extraction, visualization, subject-independent modeling, and physiological interpretation. It should not include raw WESAD data.

Scientific framing:
WESAD contains daytime stress, baseline, and amusement states, not sleep. Frame this honestly as a wearable daytime stress/arousal phenotyping pipeline that is relevant to future sleep, recovery, and decentralized home-monitoring studies.

Required structure:
- README.md
- requirements.txt
- environment.yml
- .gitignore
- configs/default.yaml
- scripts/run_preprocess.py
- scripts/run_extract_features.py
- scripts/run_train.py
- scripts/run_evaluate.py
- scripts/make_report.py
- scripts/download_instructions.md
- src/wesad_arousal/
- tests/
- notebooks/
- reports/figures/
- reports/tables/

Implementation requirements:
1. Do not commit raw data or derived WESAD files.
2. Add data/raw, data/interim, data/processed, outputs, and model artifacts to .gitignore.
3. Implement a WESAD .pkl loader.
4. Standardize labels for baseline, stress, amusement, and binary stress vs non-stress.
5. Use subject-wise evaluation only, especially leave-one-subject-out cross-validation.
6. Implement sliding windows with configurable window length and step size.
7. Implement quality control metrics: missingness, flatlines, out-of-range values, signal coverage per condition, ECG peak detection success rate where feasible, and window exclusion counts.
8. Extract interpretable physiological features:
   - ECG/HRV
   - EDA tonic/phasic features
   - respiration features
   - BVP/pulse features
   - accelerometry/motion features
   - temperature features
   - optional EMG features
9. Train baseline models:
   - DummyClassifier
   - LogisticRegression
   - RandomForestClassifier
   - HistGradientBoostingClassifier or GradientBoostingClassifier
10. Report metrics:
   - balanced accuracy
   - macro F1
   - ROC-AUC for binary stress detection
   - confusion matrix
   - per-subject metrics
11. Add modality comparison:
   - chest only
   - wrist only
   - chest + wrist
   - feature-group ablations
12. Generate plots:
   - pipeline diagram or schematic
   - example synchronized signal segment
   - QC heatmap
   - feature distributions by condition
   - confusion matrix
   - ablation bar chart
   - feature importance plot
13. Add pytest tests using synthetic toy data.
14. Write a professional README suitable for a PhD application portfolio.

Use clean, modular Python. Prefer pandas, numpy, scipy, scikit-learn, neurokit2, matplotlib, seaborn, PyYAML, pyarrow, joblib, tqdm, and pytest.

Please implement the repo step by step. Start with the scaffold, config, .gitignore, README skeleton, and core package layout. Then implement loader, labels, windowing, QC, features, modeling, evaluation, reporting, and tests.
```

## What to Put in the GitHub Description

```text
Reproducible multimodal wearable physiology pipeline for stress and autonomic arousal phenotyping using WESAD, with QC, feature extraction, subject-independent modeling, and interpretable reports.
```

## What to Mention in the PhD Application

Short version:

> To demonstrate fit with the project, I built a reproducible WESAD-based wearable physiology pipeline for daytime stress and arousal phenotyping. The repository emphasizes the research workflow described in the PhD call: multimodal biosignal preprocessing, synchronization, artifact/QC handling, feature extraction, visualization, subject-independent modeling, and interpretable physiological reporting. I intentionally used leave-one-subject-out evaluation and excluded raw data from the repository to avoid leakage and respect dataset governance.

Longer version:

> As a technical portfolio piece, I developed a public GitHub repository using the WESAD dataset to demonstrate reproducible analysis of multimodal wearable physiology for stress and arousal phenotyping. Although WESAD does not contain sleep recordings, the project is designed as a transferable pipeline for the type of mobile and decentralized phenotyping workflows described in the PhD position. It includes data loading, signal quality control, synchronization, sliding-window feature extraction from ECG, EDA, respiration, BVP, temperature and accelerometry, subject-independent machine learning evaluation, modality ablations and interpretable reports. The repository does not include raw participant data and documents the limitations of translating lab-based daytime stress findings to real-world sleep and recovery research.

## Final Quality Checklist Before Publishing

- README has screenshots/figures.
- README has a results table, even if marked as preliminary.
- Commands run from a fresh clone.
- Tests pass with synthetic data.
- No raw WESAD files are committed.
- No participant identifiers beyond WESAD subject IDs are exposed unnecessarily.
- Evaluation uses subject-wise splits.
- Limitations are explicit.
- The repo connects clearly to sleep, stress, arousal, mobile health, and physiological interpretation.

