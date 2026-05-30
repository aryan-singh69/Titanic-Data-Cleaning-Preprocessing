# Titanic Outlier Removal Report - Step 5

## Objective
Remove rows outside the IQR bounds for the selected continuous features while keeping the dataset ready for scaling later.

## Why Outlier Removal Matters
- Extreme values can distort averages and spread.
- They can influence models sensitive to distance and variance.
- Removing them can improve stability when they are not meaningful rare cases.

## When Not To Remove Outliers
- Keep valid rare observations if they carry domain meaning.
- Do not remove too many rows because the dataset can lose signal.
- In Titanic, count-like features such as Parch were not used for removal because they are highly zero-inflated and would cause excessive data loss.

## IQR Formula
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

## Columns Used For Removal
Age, Fare

## IQR Bounds And Feature-Wise Outlier Counts
### Age
- Q1: 22.0
- Q3: 35.0
- IQR: 13.0
- Lower bound: 2.5
- Upper bound: 54.5
- Outliers detected in this feature: 66

### Fare
- Q1: 7.9104
- Q3: 31.0
- IQR: 23.0896
- Lower bound: -26.724
- Upper bound: 65.6344
- Outliers detected in this feature: 116

## Dataset Size Comparison
- Dataset size before removal: 891 rows and 14 columns
- Dataset size after removal: 721 rows and 14 columns
- Number of rows removed: 170
- Percentage removed: 19.08%

## Summary Before Removal
The pre-removal boxplots still show wider whiskers and stronger upper tails. Age and Fare were the best candidates for filtering because they are continuous and informative.

## Summary After Removal
The filtered dataset is tighter, with fewer extreme tails in the selected features. This is a cleaner input for the next scaling step.

## Saved Plots
- C:\Users\Aryan\OneDrive\Desktop\Titanic\images\plots\step5_boxplots_before_outlier_removal.png
- C:\Users\Aryan\OneDrive\Desktop\Titanic\images\plots\step5_boxplots_after_outlier_removal.png

## Output Snapshot
- Rows remaining: 721
- Remaining missing values: 0