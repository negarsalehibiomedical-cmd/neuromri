# NeuroMRI Research — Bilingual MRI Research Prototype

A bilingual (Persian/English) research web app for uploading brain MRI files and running an **experimental image-classification model** once a researcher trains and configures one.

## Important scope and safety
- This is a research prototype, **not a medical device** and not suitable for diagnosis, screening, treatment decisions, or patient-facing clinical use.
- The included model architecture and training pipeline are a starting point, not a validated Alzheimer's detector.
- The app deliberately refuses to produce a probability when no trained checkpoint is installed. It never invents a result.
- MRI appearance alone is not enough to diagnose Alzheimer's disease. Clinical diagnosis requires history, cognitive assessment, imaging interpretation, and clinician judgment.
- Use only data you are authorized to process. Remove direct identifiers and obtain required ethics/IRB approval and consent/waiver before using human research data.

## Features
- Persian / English interface with language switch.
- Upload PNG, JPG, JPEG, NIfTI (`.nii`, `.nii.gz`) and DICOM (`.dcm`) files.
- Upload validation, file-size limit, temporary processing, and delete-after-analysis behavior.
- FastAPI backend and PyTorch model inference hook.
- Training script for a researcher-prepared image dataset.
- Clear unavailable-model state instead of fake predictions.

## Quick start (Python 3.10+ recommended)

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

The frontend is served by FastAPI. API docs: http://127.0.0.1:8000/docs

## Configure a trained model
1. Prepare a dataset with **patient-level separation** (never put slices from the same patient in multiple splits):
   ```
   data/
     train/
       control/
       alzheimer_pattern/
     val/
       control/
       alzheimer_pattern/
     test/
       control/
       alzheimer_pattern/
   ```
   Each folder contains 2D image slices (`.png`, `.jpg`, `.jpeg`). Use a consistent preprocessing pipeline and document MRI sequence, scanner/site, age distribution, and inclusion criteria.
2. Install training dependencies:
   ```bash
   pip install -r backend/requirements-train.txt
   ```
3. Run from the project root:
   ```bash
   python backend/train.py --data data --epochs 20 --output backend/models/research_model.pt
   ```
4. Set environment variable:
   ```bash
   # macOS/Linux
   export MODEL_PATH=backend/models/research_model.pt
   # Windows PowerShell
   $env:MODEL_PATH="backend/models/research_model.pt"
   ```
5. Restart the API.

The two classes are **dataset labels**, not a clinical diagnosis. The example training script is intentionally basic. For serious research, prefer 3D volumes, patient-level cross-validation, external-site testing, calibration, subgroup analysis, and comparison against an appropriate clinical baseline.

## Supported input
- PNG/JPG: accepted as image slices.
- NIfTI: `.nii` / `.nii.gz`, requires nibabel; central slice is used in this prototype.
- DICOM: `.dcm`, requires pydicom; one file/slice is used.
- This prototype does not reconstruct a full 3D study or automatically select the best sequence.

## Research checklist before any publication
- Ethics approval / data-use agreement and consent or waiver.
- De-identification, secure storage, access control, retention policy.
- Patient-level splits; no slice leakage.
- External validation across institutions/scanners.
- Sensitivity, specificity, AUROC, calibration, confidence intervals.
- Age, sex, education, comorbidity and site subgroup evaluation.
- Prospective workflow and independent clinical validation.
- Regulatory review before any clinical deployment.

## Project layout
```
backend/
  app/main.py
  app/model.py
  static/index.html
  requirements.txt
  requirements-train.txt
  train.py
  models/.gitkeep
```
