# AgriSmart AI

AI-powered crop disease detection and intelligent agriculture assistant for the SIH 2026 AgriSmart AI challenge.

## Project scope

The mandatory core task is leaf-image disease classification. The project uses **PlantVillage** as the main labeled training dataset and **PlantDoc** as an external field-image robustness check. Optional modules will be added after the core classifier works:

- Weather intelligence using Open-Meteo or NASA POWER
- Smart irrigation using weather and sensor inputs
- Crop recommendation using soil and crop-production data
- Sustainability scoring
- Farmer guidance and disease precautions
- Simulated IoT sensor dashboard

The college is not providing a training dataset. Therefore, all training and development data will come from documented public sources. The organizer's held-out field test, if provided, will never be used for training.

## Data sources

| Role | Source | Use |
|---|---|---|
| Main training data | [PlantVillage](https://github.com/spMohanty/PlantVillage) | Labeled leaf images for model training and validation |
| Field robustness check | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) | Real-world images with natural backgrounds and lighting |
| Weather | [Open-Meteo](https://open-meteo.com/) or [NASA POWER](https://power.larc.nasa.gov/) | Forecast and historical weather signals |
| Soil | [SoilGrids](https://www.isric.org/explore/soilgrids) | Soil properties by location |
| India crop data | [data.gov.in](https://data.gov.in/) | Crop-production and agriculture context |
| Global crop data | [FAOSTAT](https://www.fao.org/faostat/) | Crop and yield context |
| IoT demo | `data/sample_sensor_readings.csv` | Clearly documented simulated sensor feed |

## Initial supported classes

The first model will use only classes that are explicitly mapped and present in the selected datasets. The initial target list is:

- Apple: Apple Scab, Healthy
- Corn: Common Rust, Healthy
- Grape: Black Rot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy

The exact class directory names must be checked before training. We will not merge similarly named diseases across datasets without an explicit mapping.

## Repository structure

```text
agrismart-ai/
├── app/                 # Streamlit user interface
├── data/
│   ├── raw/             # Local datasets; ignored by Git
│   ├── processed/       # Generated splits; ignored by Git
│   └── sample_sensor_readings.csv
├── model/               # Training and inference code
├── reports/             # Metrics and model report
├── scripts/             # Dataset preparation utilities
├── src/                 # Reusable project modules
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the downloaded PlantVillage images under `data/raw/plantvillage/`. The expected layout is one directory per class:

```text
data/raw/plantvillage/
├── Apple___Apple_scab/
├── Apple___healthy/
└── ...
```

### Kaggle workflow

Kaggle can be used either as the download source or as the training environment. Raw data is deliberately ignored by GitHub. Never commit Kaggle API tokens, downloaded datasets, or large model checkpoints to this repository.

For local download, install the Kaggle CLI and authenticate using Kaggle's account settings. Store the downloaded `kaggle.json` in the location recommended by Kaggle, or configure the equivalent environment variables. Then run:

```powershell
python scripts/download_kaggle_dataset.py --slug OWNER/DATASET-SLUG --output-dir data/raw/plantvillage
python scripts/inspect_dataset.py --input-dir data/raw/plantvillage
```

The exact Kaggle dataset slug must be recorded in the final README after we verify that it contains the intended PlantVillage classes. If the Kaggle dataset has an extra nested directory, move or point the preparation command at the directory containing the class folders.

For Kaggle Notebook training, attach the dataset to the notebook, run `scripts/inspect_dataset.py`, then use the same `scripts/prepare_dataset.py` and `model.train` commands. Kaggle secrets should be used for credentials; they must not be pasted into this repository or chat.

Then prepare a reproducible split:

```powershell
python scripts/prepare_dataset.py --input-dir data/raw/plantvillage --output-dir data/processed/plantvillage
```

Train the first transfer-learning model:

```powershell
python -m model.train --data-dir data/processed/plantvillage --output-dir artifacts/efficientnet_b0 --epochs 10
```

Run inference on one image:

```powershell
python -m model.predict --checkpoint artifacts/efficientnet_b0/best.pt --image path\to\leaf.jpg
```

## Evaluation policy

The core report will include macro-F1, accuracy, per-class precision/recall, and a confusion matrix. PlantVillage validation/test results and PlantDoc robustness results will be reported separately. No claim will be made about the official SIH held-out score until the organizers evaluate the prediction interface.

## Originality and reproducibility

Public datasets, pretrained backbones, and open-source libraries are used with attribution. The implementation, data preparation, class mapping, evaluation, and application integration are developed for this project and will be documented in the repository.
