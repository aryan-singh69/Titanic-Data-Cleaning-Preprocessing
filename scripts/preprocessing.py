"""Titanic data preprocessing pipeline.

Current scope:
- Step 1: Dataset loading and exploration
- Step 2: Missing value handling
- Step 3: Categorical feature handling and encoding
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Titanic-Dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = OUTPUT_DIR / "step1_dataset_exploration_report.md"
STEP2_REPORT_PATH = OUTPUT_DIR / "step2_missing_value_handling_report.md"
STEP2_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "processed_data" / "titanic_missing_handled.csv"
STEP3_REPORT_PATH = OUTPUT_DIR / "step3_categorical_encoding_report.md"
STEP3_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "processed_data" / "titanic_categorical_encoded.csv"
STEP4_REPORT_PATH = OUTPUT_DIR / "step4_outlier_detection_report.md"
IMAGES_DIR = PROJECT_ROOT / "images" / "plots"
STEP5_REPORT_PATH = OUTPUT_DIR / "step5_outlier_removal_report.md"
STEP5_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "processed_data" / "titanic_outliers_removed.csv"
STEP5_BEFORE_PLOT_PATH = IMAGES_DIR / "step5_boxplots_before_outlier_removal.png"
STEP5_AFTER_PLOT_PATH = IMAGES_DIR / "step5_boxplots_after_outlier_removal.png"


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


def analyze_categorical_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
	"""Decide how each categorical column should be handled."""

	return {
		"Name": {
			"information": "Passenger name, often containing honorifics such as Mr, Mrs, Miss, and Master.",
			"decision": "Feature engineer Title and drop Name",
			"reason": "Raw names are high-cardinality text, but titles can capture useful social information.",
		},
		"Sex": {
			"information": "Binary gender category.",
			"decision": "Label encode to 0/1",
			"reason": "Sex has only two values and works well as a binary numeric feature.",
		},
		"Ticket": {
			"information": "Ticket identifier with mixed alphanumeric patterns.",
			"decision": "Drop column",
			"reason": "Ticket has very high cardinality and behaves more like an identifier than a stable predictor.",
		},
		"Embarked": {
			"information": "Port of embarkation, a nominal categorical feature.",
			"decision": "One-hot encode",
			"reason": "Embarked has only a few categories and no natural order.",
		},
	}


def handle_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
	"""Apply categorical feature selection, engineering, and encoding."""

	working_df = df.copy()
	strategy_map = analyze_categorical_columns(working_df)
	before_shape = working_df.shape
	before_columns = working_df.columns.tolist()
	before_cardinality = {
		column_name: working_df[column_name].nunique(dropna=False)
		for column_name in ["Name", "Sex", "Ticket", "Embarked"]
		if column_name in working_df.columns
	}

	working_df["Title"] = working_df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
	working_df["Title"] = working_df["Title"].fillna("Rare").str.strip()
	working_df["Title"] = working_df["Title"].replace(
		{
			"Mlle": "Miss",
			"Ms": "Miss",
			"Mme": "Mrs",
			"Lady": "Rare",
			"Countess": "Rare",
			"the Countess": "Rare",
			"Capt": "Rare",
			"Col": "Rare",
			"Don": "Rare",
			"Dr": "Rare",
			"Major": "Rare",
			"Rev": "Rare",
			"Sir": "Rare",
			"Jonkheer": "Rare",
			"Dona": "Rare",
		}
	)

	working_df["Sex"] = working_df["Sex"].map({"male": 0, "female": 1}).astype("int64")
	working_df = working_df.drop(columns=["Name", "Ticket"])
	working_df = pd.get_dummies(working_df, columns=["Embarked", "Title"], drop_first=True, dtype=int)

	after_shape = working_df.shape
	after_columns = working_df.columns.tolist()
	new_columns = [column_name for column_name in after_columns if column_name not in before_columns]
	removed_columns = [column_name for column_name in before_columns if column_name not in after_columns]
	encoded_columns = [column_name for column_name in after_columns if column_name.startswith("Embarked_") or column_name.startswith("Title_")]

	summary = {
		"before_shape": before_shape,
		"after_shape": after_shape,
		"before_columns": before_columns,
		"after_columns": after_columns,
		"before_cardinality": before_cardinality,
		"strategies": strategy_map,
		"new_columns": new_columns,
		"removed_columns": removed_columns,
		"encoded_columns": encoded_columns,
	}

	return working_df, summary


def get_outlier_analysis_columns(df: pd.DataFrame) -> List[str]:
	"""Select numeric columns suitable for outlier analysis."""

	excluded_columns = {"PassengerId", "Survived"}
	analysis_columns = []

	for column_name in df.select_dtypes(include=["number"]).columns:
		if column_name in excluded_columns:
			continue
		if df[column_name].nunique(dropna=False) <= 2:
			continue
		analysis_columns.append(column_name)

	return analysis_columns


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
	"""Detect outliers in the selected columns using the IQR method."""

	outlier_summary: Dict[str, Dict[str, float]] = {}

	for column_name in columns:
		series = df[column_name]
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		lower_bound = q1 - 1.5 * iqr
		upper_bound = q3 + 1.5 * iqr
		outlier_mask = (series < lower_bound) | (series > upper_bound)
		outlier_count = int(outlier_mask.sum())
		outlier_percentage = round((outlier_count / len(series)) * 100, 2)

		outlier_summary[column_name] = {
			"q1": float(q1),
			"q3": float(q3),
			"iqr": float(iqr),
			"lower_bound": float(lower_bound),
			"upper_bound": float(upper_bound),
			"outlier_count": outlier_count,
			"outlier_percentage": outlier_percentage,
		}

	return outlier_summary


def create_boxplots(df: pd.DataFrame, columns: List[str]) -> List[Path]:
	"""Create and save boxplots for the selected numeric columns."""

	IMAGES_DIR.mkdir(parents=True, exist_ok=True)
	saved_plots: List[Path] = []

	if not columns:
		return saved_plots

	fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
	if len(columns) == 1:
		axes = [axes]

	for axis, column_name in zip(axes, columns):
		sns.boxplot(x=df[column_name], ax=axis, color="#4c72b0")
		axis.set_title(f"Boxplot of {column_name}")
		axis.set_xlabel(column_name)

	combined_plot_path = IMAGES_DIR / "step4_boxplots_numerical_features.png"
	fig.tight_layout()
	fig.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	saved_plots.append(combined_plot_path)

	return saved_plots


def build_outlier_report(df: pd.DataFrame, analysis_columns: List[str], outlier_summary: Dict[str, Dict[str, float]], plot_paths: List[Path]) -> str:
	"""Create a markdown report for Step 4 outlier detection."""

	selected_summary = df[analysis_columns].describe().transpose() if analysis_columns else pd.DataFrame()
	selected_summary_table = selected_summary.to_string() if not selected_summary.empty else "No numeric columns selected for outlier analysis."

	lines = [
		"# Titanic Outlier Detection Report - Step 4",
		"",
		"## Objective",
		"Identify potential outliers in the Titanic dataset using the IQR method and visualize them with boxplots, without removing any rows yet.",
		"",
		"## What Is An Outlier?",
		"An outlier is a value that lies far away from the typical range of the data. In machine learning, outliers can distort averages, affect model training, and mislead interpretation.",
		"",
		"## Why Outliers Matter",
		"- They can skew summary statistics.",
		"- They can influence distance-based and regression-based models.",
		"- They may represent data errors or genuine rare cases.",
		"",
		"## When To Remove Or Keep Outliers",
		"- Remove outliers when they are clearly data-entry errors or impossible values.",
		"- Keep outliers when they are valid rare observations that carry business or survival meaning.",
		"- For Titanic, detection should come before removal because the dataset contains real rare values such as high Fare.",
		"",
		"## Numerical Features Considered For Outlier Analysis",
		", ".join(analysis_columns) if analysis_columns else "None",
		"",
		"## Statistical Summary Before Removal",
		"```text",
		selected_summary_table,
		"```",
		"",
		"## IQR Rule",
		"Lower Bound = Q1 - 1.5 × IQR",
		"Upper Bound = Q3 + 1.5 × IQR",
		"Values outside these bounds are flagged as outliers.",
		"",
		"## Outlier Detection Summary",
	]

	for column_name, details in outlier_summary.items():
		lines.extend([
			f"### {column_name}",
			f"- Q1: {details['q1']}",
			f"- Q3: {details['q3']}",
			f"- IQR: {details['iqr']}",
			f"- Lower bound: {details['lower_bound']}",
			f"- Upper bound: {details['upper_bound']}",
			f"- Outlier count: {details['outlier_count']}",
			f"- Outlier percentage: {details['outlier_percentage']}%",
			"",
		])

	lines.extend([
		"## Saved Plots",
	])

	for plot_path in plot_paths:
		lines.append(f"- {plot_path}")

	lines.extend([
		"",
		"## Interpretation",
		"The Titanic dataset is better suited to IQR-based detection because some numerical features are skewed and contain real extreme values. IQR is robust against skewness, while Z-score assumes a more normal distribution.",
		"",
		"## Output Snapshot",
		f"- Rows analyzed: {df.shape[0]}",
		f"- Columns analyzed: {len(analysis_columns)}",
	])

	return "\n".join(lines)


def select_outlier_removal_columns(df: pd.DataFrame) -> List[str]:
	"""Select the continuous numerical features to use for IQR-based row removal."""

	# Age and Fare are continuous and informative; count-like columns are kept for later judgment.
	return [column_name for column_name in ["Age", "Fare"] if column_name in df.columns]


def compute_iqr_bounds(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
	"""Compute Q1, Q3, IQR, and the lower and upper bounds for each selected column."""

	bounds: Dict[str, Dict[str, float]] = {}

	for column_name in columns:
		q1 = df[column_name].quantile(0.25)
		q3 = df[column_name].quantile(0.75)
		iqr = q3 - q1
		lower_bound = q1 - 1.5 * iqr
		upper_bound = q3 + 1.5 * iqr
		bounds[column_name] = {
			"q1": float(q1),
			"q3": float(q3),
			"iqr": float(iqr),
			"lower_bound": float(lower_bound),
			"upper_bound": float(upper_bound),
		}

	return bounds


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
	"""Remove rows that fall outside IQR bounds for the selected columns."""

	working_df = df.copy()
	before_shape = working_df.shape
	bounds = compute_iqr_bounds(working_df, columns)
	feature_outlier_counts: Dict[str, int] = {}
	combined_mask = pd.Series(True, index=working_df.index)

	for column_name, bound_details in bounds.items():
		column_mask = working_df[column_name].between(bound_details["lower_bound"], bound_details["upper_bound"])
		feature_outlier_counts[column_name] = int((~column_mask).sum())
		combined_mask &= column_mask

	removed_df = working_df.loc[combined_mask].copy()
	after_shape = removed_df.shape
	rows_removed = before_shape[0] - after_shape[0]
	percentage_removed = round((rows_removed / before_shape[0]) * 100, 2)

	summary = {
		"before_shape": before_shape,
		"after_shape": after_shape,
		"rows_removed": rows_removed,
		"percentage_removed": percentage_removed,
		"feature_outlier_counts": feature_outlier_counts,
		"bounds": bounds,
		"columns_used_for_removal": columns,
	}

	return removed_df, summary


def create_outlier_removal_plots(before_df: pd.DataFrame, after_df: pd.DataFrame, columns: List[str]) -> List[Path]:
	"""Save boxplots before and after outlier removal."""

	IMAGES_DIR.mkdir(parents=True, exist_ok=True)
	saved_paths: List[Path] = []

	if not columns:
		return saved_paths

	fig_before, axes_before = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
	if len(columns) == 1:
		axes_before = [axes_before]

	for axis, column_name in zip(axes_before, columns):
		sns.boxplot(x=before_df[column_name], ax=axis, color="#c44e52")
		axis.set_title(f"Before Outlier Removal - {column_name}")
		axis.set_xlabel(column_name)

	fig_before.tight_layout()
	fig_before.savefig(STEP5_BEFORE_PLOT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig_before)
	saved_paths.append(STEP5_BEFORE_PLOT_PATH)

	fig_after, axes_after = plt.subplots(len(columns), 1, figsize=(10, 4 * len(columns)))
	if len(columns) == 1:
		axes_after = [axes_after]

	for axis, column_name in zip(axes_after, columns):
		sns.boxplot(x=after_df[column_name], ax=axis, color="#4c72b0")
		axis.set_title(f"After Outlier Removal - {column_name}")
		axis.set_xlabel(column_name)

	fig_after.tight_layout()
	fig_after.savefig(STEP5_AFTER_PLOT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig_after)
	saved_paths.append(STEP5_AFTER_PLOT_PATH)

	return saved_paths


def build_outlier_removal_report(summary: Dict[str, object], cleaned_df: pd.DataFrame, plot_paths: List[Path]) -> str:
	"""Create a markdown report for Step 5 outlier removal."""

	before_shape = summary["before_shape"]
	after_shape = summary["after_shape"]
	rows_removed = summary["rows_removed"]
	percentage_removed = summary["percentage_removed"]
	feature_outlier_counts = summary["feature_outlier_counts"]
	bounds = summary["bounds"]
	columns_used = summary["columns_used_for_removal"]

	lines = [
		"# Titanic Outlier Removal Report - Step 5",
		"",
		"## Objective",
		"Remove rows outside the IQR bounds for the selected continuous features while keeping the dataset ready for scaling later.",
		"",
		"## Why Outlier Removal Matters",
		"- Extreme values can distort averages and spread.",
		"- They can influence models sensitive to distance and variance.",
		"- Removing them can improve stability when they are not meaningful rare cases.",
		"",
		"## When Not To Remove Outliers",
		"- Keep valid rare observations if they carry domain meaning.",
		"- Do not remove too many rows because the dataset can lose signal.",
		"- In Titanic, count-like features such as Parch were not used for removal because they are highly zero-inflated and would cause excessive data loss.",
		"",
		"## IQR Formula",
		"- Q1 = 25th percentile",
		"- Q3 = 75th percentile",
		"- IQR = Q3 - Q1",
		"- Lower Bound = Q1 - 1.5 × IQR",
		"- Upper Bound = Q3 + 1.5 × IQR",
		"",
		"## Columns Used For Removal",
		", ".join(columns_used) if columns_used else "None",
		"",
		"## IQR Bounds And Feature-Wise Outlier Counts",
	]

	for column_name, bound_details in bounds.items():
		lines.extend([
			f"### {column_name}",
			f"- Q1: {bound_details['q1']}",
			f"- Q3: {bound_details['q3']}",
			f"- IQR: {bound_details['iqr']}",
			f"- Lower bound: {bound_details['lower_bound']}",
			f"- Upper bound: {bound_details['upper_bound']}",
			f"- Outliers detected in this feature: {feature_outlier_counts[column_name]}",
			"",
		])

	lines.extend([
		"## Dataset Size Comparison",
		f"- Dataset size before removal: {before_shape[0]} rows and {before_shape[1]} columns",
		f"- Dataset size after removal: {after_shape[0]} rows and {after_shape[1]} columns",
		f"- Number of rows removed: {rows_removed}",
		f"- Percentage removed: {percentage_removed}%",
		"",
		"## Summary Before Removal",
		"The pre-removal boxplots still show wider whiskers and stronger upper tails. Age and Fare were the best candidates for filtering because they are continuous and informative.",
		"",
		"## Summary After Removal",
		"The filtered dataset is tighter, with fewer extreme tails in the selected features. This is a cleaner input for the next scaling step.",
		"",
		"## Saved Plots",
	])

	for plot_path in plot_paths:
		lines.append(f"- {plot_path}")

	lines.extend([
		"",
		"## Output Snapshot",
		f"- Rows remaining: {cleaned_df.shape[0]}",
		f"- Remaining missing values: {int(cleaned_df.isnull().sum().sum())}",
	])

	return "\n".join(lines)


def build_categorical_encoding_report(summary: Dict[str, object], encoded_df: pd.DataFrame) -> str:
	"""Create a markdown report for Step 3 categorical handling and encoding."""

	strategies = summary["strategies"]
	before_shape = summary["before_shape"]
	after_shape = summary["after_shape"]
	before_columns = summary["before_columns"]
	after_columns = summary["after_columns"]
	before_cardinality = summary["before_cardinality"]
	new_columns = summary["new_columns"]
	removed_columns = summary["removed_columns"]
	encoded_columns = summary["encoded_columns"]

	lines = [
		"# Titanic Categorical Feature Handling & Encoding Report - Step 3",
		"",
		"## Objective",
		"Convert categorical information into model-ready numeric features while removing noisy identifier-like columns.",
		"",
		"## Theory Summary",
		"- Categorical data stores labels or groups instead of continuous numeric measurements.",
		"- Machine learning models require numeric input because they perform mathematical operations on features.",
		"- Label encoding converts categories into integers and is useful for binary variables.",
		"- One-hot encoding creates separate binary columns for each category and is preferred for nominal variables without order.",
		"",
		"## Column-Wise Decisions",
	]

	for column_name, details in strategies.items():
		lines.extend([
			f"### {column_name}",
			f"- Information: {details['information']}",
			f"- Decision: {details['decision']}",
			f"- Reason: {details['reason']}",
			f"- Unique values before encoding: {before_cardinality.get(column_name, 'N/A')}",
			"",
		])

	lines.extend([
		"## Before Encoding",
		f"- Shape: {before_shape[0]} rows and {before_shape[1]} columns",
		"- Columns:",
		"```text",
		", ".join(before_columns),
		"```",
		"",
		"## After Encoding",
		f"- Shape: {after_shape[0]} rows and {after_shape[1]} columns",
		"- Columns:",
		"```text",
		", ".join(after_columns),
		"```",
		"",
		"## Added Columns",
		", ".join(new_columns) if new_columns else "None",
		"",
		"## Removed Columns",
		", ".join(removed_columns) if removed_columns else "None",
		"",
		"## Encoded Columns",
		", ".join(encoded_columns) if encoded_columns else "None",
		"",
		"## Output Snapshot",
		f"- Final missing values: {int(encoded_df.isnull().sum().sum())}",
		f"- Dataframe ready for later steps: {encoded_df.shape[0]} rows and {encoded_df.shape[1]} columns",
	])

	return "\n".join(lines)


def main() -> None:
	"""Run Step 1 through Step 4 of the Titanic preprocessing pipeline."""

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

	encoded_df, categorical_summary = handle_categorical_features(cleaned_df)
	STEP3_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
	encoded_df.to_csv(STEP3_CHECKPOINT_PATH, index=False)

	step3_report = build_categorical_encoding_report(categorical_summary, encoded_df)
	STEP3_REPORT_PATH.write_text(step3_report, encoding="utf-8")

	print("\nStep 3: Categorical Feature Handling & Encoding")
	print(f"Before Encoding Shape: {categorical_summary['before_shape']}")
	print(f"After Encoding Shape: {categorical_summary['after_shape']}")
	print("\nRemoved Columns:")
	print(categorical_summary["removed_columns"])
	print("\nAdded Columns:")
	print(categorical_summary["new_columns"])
	print(f"\nStep 3 checkpoint saved to: {STEP3_CHECKPOINT_PATH}")
	print(f"Step 3 report saved to: {STEP3_REPORT_PATH}")

	analysis_columns = get_outlier_analysis_columns(encoded_df)
	plot_paths = create_boxplots(encoded_df, analysis_columns)
	outlier_summary = detect_outliers_iqr(encoded_df, analysis_columns)
	step4_report = build_outlier_report(encoded_df, analysis_columns, outlier_summary, plot_paths)
	STEP4_REPORT_PATH.write_text(step4_report, encoding="utf-8")

	print("\nStep 4: Outlier Detection & Visualization")
	print("Outlier Analysis Columns:")
	print(analysis_columns)
	print("\nOutlier Summary:")
	for column_name, details in outlier_summary.items():
		print(
			f"{column_name}: outliers={details['outlier_count']}, "
			f"percentage={details['outlier_percentage']}%, "
			f"bounds=({details['lower_bound']}, {details['upper_bound']})"
		)
	print("\nSaved Plots:")
	for plot_path in plot_paths:
		print(plot_path)
	print(f"\nStep 4 report saved to: {STEP4_REPORT_PATH}")

	removal_columns = select_outlier_removal_columns(encoded_df)
	filtered_df, removal_summary = remove_outliers_iqr(encoded_df, removal_columns)
	removal_plot_paths = create_outlier_removal_plots(encoded_df, filtered_df, removal_columns)
	STEP5_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
	filtered_df.to_csv(STEP5_CHECKPOINT_PATH, index=False)

	step5_report = build_outlier_removal_report(removal_summary, filtered_df, removal_plot_paths)
	STEP5_REPORT_PATH.write_text(step5_report, encoding="utf-8")

	print("\nStep 5: Outlier Removal")
	print("Removal Columns:")
	print(removal_columns)
	print(f"Dataset size before removal: {removal_summary['before_shape']}")
	print(f"Dataset size after removal: {removal_summary['after_shape']}")
	print(f"Rows removed: {removal_summary['rows_removed']}")
	print(f"Percentage removed: {removal_summary['percentage_removed']}%")
	print("\nFeature-wise outlier counts used for removal:")
	print(removal_summary["feature_outlier_counts"])
	print("\nSaved removal plots:")
	for plot_path in removal_plot_paths:
		print(plot_path)
	print(f"\nStep 5 checkpoint saved to: {STEP5_CHECKPOINT_PATH}")
	print(f"Step 5 report saved to: {STEP5_REPORT_PATH}")


if __name__ == "__main__":
	main()