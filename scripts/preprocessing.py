"""Titanic data preprocessing pipeline.

Current scope:
- Step 1: Dataset loading and exploration
- Step 2: Missing value handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Titanic-Dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = OUTPUT_DIR / "step1_dataset_exploration_report.md"
STEP2_REPORT_PATH = OUTPUT_DIR / "step2_missing_value_handling_report.md"
STEP2_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "processed_data" / "titanic_missing_handled.csv"


def ensure_output_directory() -> None:
	"""Create the report directory if it does not already exist."""

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
	"""Load the Titanic dataset from the raw data folder."""

	if not file_path.exists():
		raise FileNotFoundError(f"Dataset not found: {file_path}")

	return pd.read_csv(file_path)


def classify_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	"""Classify columns into numerical and categorical features."""

	numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
	categorical_features = df.select_dtypes(exclude=["number"]).columns.tolist()
	return numerical_features, categorical_features


def explore_dataset(df: pd.DataFrame) -> Dict[str, object]:
	"""Collect the core Step 1 exploration outputs."""

	missing_counts = df.isnull().sum()
	missing_percentage = ((missing_counts / len(df)) * 100).round(2)
	numerical_features, categorical_features = classify_features(df)

	profile = {
		"shape": df.shape,
		"columns": df.columns.tolist(),
		"dtypes": df.dtypes.astype(str).to_dict(),
		"summary": df.describe(include="all").transpose(),
		"missing_counts": missing_counts,
		"missing_percentage": missing_percentage,
		"numerical_features": numerical_features,
		"categorical_features": categorical_features,
		"target_variable": "Survived",
	}
	return profile


def build_report(df: pd.DataFrame, profile: Dict[str, object]) -> str:
	"""Build a readable Markdown report for Step 1 findings."""

	shape = profile["shape"]
	columns = profile["columns"]
	dtypes = profile["dtypes"]
	summary = profile["summary"]
	missing_counts = profile["missing_counts"]
	missing_percentage = profile["missing_percentage"]
	numerical_features = profile["numerical_features"]
	categorical_features = profile["categorical_features"]
	target_variable = profile["target_variable"]

	missing_counts_table = missing_counts.to_frame(name="missing_count").to_string()
	missing_percentage_table = missing_percentage.to_frame(name="missing_percentage").to_string()
	summary_table = summary.to_string()

	report_lines = [
		"# Titanic Dataset Exploration Report - Step 1",
		"",
		"## Objective",
		"Understand the raw Titanic dataset before any cleaning or transformation.",
		"",
		"## Dataset Overview",
		f"- Shape: {shape[0]} rows and {shape[1]} columns",
		f"- Target variable: {target_variable}",
		"",
		"## Columns",
		", ".join(columns),
		"",
		"## Data Types",
	]

	for column_name, dtype_name in dtypes.items():
		report_lines.append(f"- {column_name}: {dtype_name}")

	report_lines.extend([
		"",
		"## Missing Values",
		"```text",
		missing_counts_table,
		"```",
		"",
		"## Missing Value Percentage",
		"```text",
		missing_percentage_table,
		"```",
		"",
		"## Numerical Features",
		", ".join(numerical_features),
		"",
		"## Categorical Features",
		", ".join(categorical_features),
		"",
		"## Statistical Summary",
		"```text",
		summary_table,
		"```",
		"",
		"## Notes",
		"- PassengerId is numeric in format but acts like an identifier and is usually not used as a predictive feature.",
		"- Survived is the target variable for future modeling tasks.",
		"- Age, Cabin, and Embarked contain missing values and will need preprocessing in later steps.",
	])

	return "\n".join(report_lines)


def analyze_missing_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
	"""Decide the treatment strategy for each missing column."""

	column_strategies = {
		"Age": {
			"missing_type": "Numeric, moderate missingness",
			"strategy": "Fill with median",
			"reason": "Age is skewed and has missing values; median is robust to outliers.",
		},
		"Embarked": {
			"missing_type": "Categorical, very low missingness",
			"strategy": "Fill with mode",
			"reason": "Embarked has only two missing values, so the most frequent category is appropriate.",
		},
		"Cabin": {
			"missing_type": "Categorical, extremely high missingness",
			"strategy": "Drop column",
			"reason": "Cabin is missing for most rows, so imputation would add noise and unreliable assumptions.",
		},
	}
	return column_strategies


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
	"""Apply the chosen missing-value handling strategy."""

	working_df = df.copy()
	strategy_map = analyze_missing_columns(working_df)
	before_missing = working_df.isnull().sum()

	age_median = working_df["Age"].median()
	embarked_mode = working_df["Embarked"].mode(dropna=True)[0]

	working_df["Age"] = working_df["Age"].fillna(age_median)
	working_df["Embarked"] = working_df["Embarked"].fillna(embarked_mode)
	working_df = working_df.drop(columns=["Cabin"])

	after_missing = working_df.isnull().sum()

	handling_summary = {
		"before_missing": before_missing,
		"after_missing": after_missing,
		"age_median": age_median,
		"embarked_mode": embarked_mode,
		"dropped_columns": ["Cabin"],
		"strategies": strategy_map,
	}

	return working_df, handling_summary


def build_missing_value_report(summary: Dict[str, object], cleaned_df: pd.DataFrame) -> str:
	"""Create a markdown report for Step 2 missing value handling."""

	before_missing = summary["before_missing"]
	after_missing = summary["after_missing"]
	strategies = summary["strategies"]
	age_median = summary["age_median"]
	embarked_mode = summary["embarked_mode"]
	dropped_columns = summary["dropped_columns"]

	before_table = before_missing.to_frame(name="missing_count").to_string()
	after_table = after_missing.to_frame(name="missing_count").to_string()

	lines = [
		"# Titanic Missing Value Handling Report - Step 2",
		"",
		"## Objective",
		"Handle missing values using a data-driven strategy before moving to encoding.",
		"",
		"## Column-Wise Strategy",
	]

	for column_name, details in strategies.items():
		lines.extend([
			f"### {column_name}",
			f"- Missing type: {details['missing_type']}",
			f"- Strategy: {details['strategy']}",
			f"- Reason: {details['reason']}",
			"",
		])

	lines.extend([
		"## Before Missing Value Counts",
		"```text",
		before_table,
		"```",
		"",
		"## After Missing Value Counts",
		"```text",
		after_table,
		"```",
		"",
		"## Applied Values",
		f"- Age median used: {age_median}",
		f"- Embarked mode used: {embarked_mode}",
		f"- Dropped columns: {', '.join(dropped_columns)}",
		"",
		"## Output Snapshot",
		f"- Shape after handling missing values: {cleaned_df.shape[0]} rows and {cleaned_df.shape[1]} columns",
		f"- Remaining missing values: {int(cleaned_df.isnull().sum().sum())}",
	])

	return "\n".join(lines)


def main() -> None:
	"""Run Step 1 and Step 2 of the Titanic preprocessing pipeline."""

	ensure_output_directory()

	df = load_dataset(RAW_DATA_PATH)
	profile = explore_dataset(df)

	print("Titanic Dataset Loaded Successfully")
	print(f"Dataset Path: {RAW_DATA_PATH}")
	print(f"Shape: {profile['shape'][0]} rows, {profile['shape'][1]} columns")
	print("\nColumns:")
	print(profile["columns"])
	print("\nData Types:")
	print(pd.Series(profile["dtypes"]))
	print("\nMissing Values:")
	print(profile["missing_counts"])
	print("\nMissing Value Percentage:")
	print(profile["missing_percentage"])
	print("\nNumerical Features:")
	print(profile["numerical_features"])
	print("\nCategorical Features:")
	print(profile["categorical_features"])
	print(f"\nTarget Variable: {profile['target_variable']}")

	report_text = build_report(df, profile)
	REPORT_PATH.write_text(report_text, encoding="utf-8")
	print(f"\nStep 1 report saved to: {REPORT_PATH}")

	cleaned_df, missing_summary = handle_missing_values(df)
	STEP2_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
	cleaned_df.to_csv(STEP2_CHECKPOINT_PATH, index=False)

	step2_report = build_missing_value_report(missing_summary, cleaned_df)
	STEP2_REPORT_PATH.write_text(step2_report, encoding="utf-8")

	print("\nStep 2: Missing Value Handling")
	print("Before Missing Values:")
	print(missing_summary["before_missing"])
	print("\nAfter Missing Values:")
	print(missing_summary["after_missing"])
	print(f"\nStep 2 checkpoint saved to: {STEP2_CHECKPOINT_PATH}")
	print(f"Step 2 report saved to: {STEP2_REPORT_PATH}")


if __name__ == "__main__":
	main()