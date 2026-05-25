# AI-Powered Appliance Energy Disaggregation (NILM)

This project is a working end-to-end Python machine learning pipeline for Non-Intrusive Load Monitoring (NILM). It generates or loads aggregate household energy data, performs EDA, trains baseline machine learning models, trains sequence models, evaluates all models, and exposes a Streamlit dashboard.

## Project Structure

```text
C:\project
|-- 01_dataset_loader.py
|-- 02_exploratory_analysis.py
|-- 03_baseline_models.py
|-- 04_deep_learning_models.py
|-- 05_model_evaluation.py
|-- run_complete_pipeline.py
|-- streamlit_app.py
|-- nilm_pipeline.py
|-- requirements.txt
|-- package.json
|-- scripts\
    |-- dataset_loader.py
    |-- exploratory_analysis.py
    |-- baseline_models.py
    |-- deep_learning_models.py
    |-- model_evaluation.py
```

`nilm_pipeline.py` contains the shared implementation. The numbered files are runnable entry points for each stage. The files under `scripts\` are compatibility imports for older code that imports modules such as `scripts.dataset_loader`.

## Setup

Run these commands from `C:\project`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is optional. If TensorFlow is not installed, the sequence-model stage uses a fast sklearn MLP fallback so the pipeline still runs. To use real LSTM, GRU, and CNN-LSTM models:

```powershell
pip install "tensorflow>=2.16"
```

## Run Each Stage

```powershell
python 01_dataset_loader.py
python 02_exploratory_analysis.py
python 03_baseline_models.py
python 04_deep_learning_models.py
python 05_model_evaluation.py
```

## Run The Complete Pipeline

```powershell
python run_complete_pipeline.py
```

Generated outputs are written to:

```text
C:\project\artifacts\models
C:\project\artifacts\plots
C:\project\artifacts\reports
```

The main evaluation report is:

```text
C:\project\artifacts\reports\nilm_comprehensive_report.txt
```

## Run The Streamlit App

```powershell
streamlit run streamlit_app.py
```

In the sidebar you can generate synthetic data, upload a CSV, train selected models, and inspect evaluation results.

## CSV Dataset Format

For uploaded data, the required columns are:

```text
timestamp,aggregate
```

Optional appliance columns can also be included:

```text
fridge,washing_machine,microwave,ac,tv
```

If appliance-level columns are not present, the app estimates appliance splits from aggregate data for demonstration purposes. This is useful for testing the workflow, but real NILM model quality requires measured appliance-level training labels.

## About `package.json`

This is not a Node.js or Next.js project, so `package.json` is not required for running the ML pipeline or Streamlit app. It is kept only as a harmless metadata file with an informational script because the original file contained unrelated Next.js dependencies.
