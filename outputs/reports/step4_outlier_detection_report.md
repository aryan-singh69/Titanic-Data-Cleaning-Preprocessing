# Titanic Outlier Detection Report - Step 4

## Objective
Identify potential outliers in the Titanic dataset using the IQR method and visualize them with boxplots, without removing any rows yet.

## What Is An Outlier?
An outlier is a value that lies far away from the typical range of the data. In machine learning, outliers can distort averages, affect model training, and mislead interpretation.

## Why Outliers Matter
- They can skew summary statistics.
- They can influence distance-based and regression-based models.
- They may represent data errors or genuine rare cases.

## When To Remove Or Keep Outliers
- Remove outliers when they are clearly data-entry errors or impossible values.
- Keep outliers when they are valid rare observations that carry business or survival meaning.
- For Titanic, detection should come before removal because the dataset contains real rare values such as high Fare.

## Numerical Features Considered For Outlier Analysis
Pclass, Age, SibSp, Parch, Fare

## Statistical Summary Before Removal
```text
        count       mean        std   min      25%      50%   75%       max
Pclass  891.0   2.308642   0.836071  1.00   2.0000   3.0000   3.0    3.0000
Age     891.0  29.361582  13.019697  0.42  22.0000  28.0000  35.0   80.0000
SibSp   891.0   0.523008   1.102743  0.00   0.0000   0.0000   1.0    8.0000
Parch   891.0   0.381594   0.806057  0.00   0.0000   0.0000   0.0    6.0000
Fare    891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292
```

## IQR Rule
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Values outside these bounds are flagged as outliers.

## Outlier Detection Summary
### Pclass
- Q1: 2.0
- Q3: 3.0
- IQR: 1.0
- Lower bound: 0.5
- Upper bound: 4.5
- Outlier count: 0
- Outlier percentage: 0.0%

### Age
- Q1: 22.0
- Q3: 35.0
- IQR: 13.0
- Lower bound: 2.5
- Upper bound: 54.5
- Outlier count: 66
- Outlier percentage: 7.41%

### SibSp
- Q1: 0.0
- Q3: 1.0
- IQR: 1.0
- Lower bound: -1.5
- Upper bound: 2.5
- Outlier count: 46
- Outlier percentage: 5.16%

### Parch
- Q1: 0.0
- Q3: 0.0
- IQR: 0.0
- Lower bound: 0.0
- Upper bound: 0.0
- Outlier count: 213
- Outlier percentage: 23.91%

### Fare
- Q1: 7.9104
- Q3: 31.0
- IQR: 23.0896
- Lower bound: -26.724
- Upper bound: 65.6344
- Outlier count: 116
- Outlier percentage: 13.02%

## Saved Plots
- C:\Users\Aryan\OneDrive\Desktop\Titanic\images\plots\step4_boxplots_numerical_features.png

## Interpretation
The Titanic dataset is better suited to IQR-based detection because some numerical features are skewed and contain real extreme values. IQR is robust against skewness, while Z-score assumes a more normal distribution.

## Output Snapshot
- Rows analyzed: 891
- Columns analyzed: 5