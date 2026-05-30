# Titanic Data Cleaning & Preprocessing

Script-based Titanic preprocessing project for an AI/ML internship submission. The pipeline cleans missing values, engineers categorical features, detects and removes outliers, standardizes numerical features, and saves a final ML-ready dataset.

## Project Summary

This project transforms the raw Titanic passenger dataset into a clean, model-ready dataset using a reproducible Python script workflow. It demonstrates practical data cleaning decisions, feature engineering, IQR-based outlier handling, and feature scaling with professional GitHub presentation.

## Objective

Prepare the Titanic dataset for machine learning by applying a structured preprocessing pipeline that preserves useful signal, removes noise, and produces a final cleaned dataset for downstream modeling.

## Dataset Information

- Source: Titanic passenger dataset
- Original size: 891 rows, 12 columns
- Final size after preprocessing: 721 rows, 14 columns
- Target variable: Survived
- Raw data location: data/raw/Titanic-Dataset.csv
- Final output location: data/processed/cleaned_titanic.csv

## Technologies Used

- Python 3.x
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Project Structure

```text
Titanic-Data-Cleaning-Preprocessing/
├── data/
│   ├── raw/
│   │   └── Titanic-Dataset.csv
│   └── processed/
│       └── cleaned_titanic.csv
├── images/
│   └── plots/
├── outputs/
│   ├── processed_data/
│   └── reports/
├── scripts/
│   └── preprocessing.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Purpose Of Each Folder And File

| Path | Purpose |
|---|---|
| data/raw/ | Stores the untouched source dataset. |
| data/processed/ | Stores the final cleaned dataset used for modeling. |
| images/plots/ | Stores saved visualizations for GitHub and documentation. |
| outputs/processed_data/ | Stores intermediate processed checkpoints from each pipeline stage. |
| outputs/reports/ | Stores Markdown reports generated during each preprocessing step. |
| scripts/preprocessing.py | Main script that runs the full preprocessing workflow end to end. |
| README.md | Explains the project for reviewers, recruiters, and GitHub visitors. |
| requirements.txt | Lists the Python dependencies required to run the project. |
| .gitignore | Excludes local, temporary, and machine-specific files from version control. |

## Data Preprocessing Workflow

1. Dataset Loading and Exploration
2. Missing Value Handling
3. Categorical Feature Encoding
4. Outlier Detection
5. Outlier Removal
6. Feature Scaling
7. Save Final Dataset

### Workflow Notes

- Dataset Loading and Exploration: inspect shape, columns, dtypes, summary statistics, and missing values.
- Missing Value Handling: fill Age with median, fill Embarked with mode, and drop Cabin.
- Categorical Feature Encoding: engineer Title from Name, label encode Sex, one-hot encode Embarked and Title, and drop Name and Ticket.
- Outlier Detection: visualize numeric features with boxplots and detect outliers using the IQR method.
- Outlier Removal: remove rows outside IQR bounds for Age and Fare only.
- Feature Scaling: standardize Pclass, Age, SibSp, Parch, and Fare using StandardScaler.
- Save Final Dataset: export the final dataset to data/processed/cleaned_titanic.csv.

## Results

- Missing values were resolved in Age and Embarked.
- Cabin was removed due to extremely high missingness.
- Title was engineered from Name to preserve useful social information.
- Outliers were detected and removed using the IQR method on Age and Fare.
- Numeric features were standardized for modeling.
- The final dataset is ready for ML training and experimentation.

## How To Run The Project

```bash
python scripts/preprocessing.py
```

Recommended setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/preprocessing.py
```

## Output Files

### Final Dataset
- data/processed/cleaned_titanic.csv

### Intermediate Processed Data
- outputs/processed_data/titanic_missing_handled.csv
- outputs/processed_data/titanic_categorical_encoded.csv
- outputs/processed_data/titanic_outliers_removed.csv

### Reports
- outputs/reports/step1_dataset_exploration_report.md
- outputs/reports/step2_missing_value_handling_report.md
- outputs/reports/step3_categorical_encoding_report.md
- outputs/reports/step4_outlier_detection_report.md
- outputs/reports/step5_outlier_removal_report.md
- outputs/reports/step6_feature_scaling_report.md

### Visualizations
- images/plots/step4_boxplots_numerical_features.png
- images/plots/step5_boxplots_before_outlier_removal.png
- images/plots/step5_boxplots_after_outlier_removal.png

## Author Information

- Author: Aryan
- Project Type: AI/ML Internship Submission
- Repository Focus: Data Cleaning, Preprocessing, and GitHub Presentation

## Repository Description

Titanic data preprocessing pipeline in Python that converts raw passenger data into a clean, scaled dataset for machine learning.
