# Titanic Feature Scaling Report - Step 6

## Objective
Standardize the final numeric features so the dataset is ready for model training.

## What Is Feature Scaling?
Feature scaling is the process of bringing numeric variables onto a comparable scale so that large-valued features do not dominate model learning.

## Why It Matters
- Distance-based models become biased when features have different ranges.
- Gradient-based models converge more smoothly with scaled inputs.
- Regularized models are easier to interpret when features are on the same scale.

## Normalization vs Standardization
- Normalization rescales values to a fixed range such as 0 to 1.
- Standardization converts values into z-scores with mean 0 and standard deviation 1.
- For Titanic, standardization is preferred because the numeric features have different spreads and include count-like and continuous variables.

## Algorithms Affected By Feature Scale
- k-NN
- k-Means
- SVM
- PCA
- Logistic Regression and other gradient-based models

## Columns Scaled
Pclass, Age, SibSp, Parch, Fare

## Before Scaling Summary
```text
             mean        std  min   max
Pclass   2.509015   0.717046  1.0   3.0
Age     28.094313  10.021961  3.0  54.0
SibSp    0.414702   0.853916  0.0   5.0
Parch    0.323162   0.788549  0.0   6.0
Fare    17.389845  13.563036  0.0  65.0
```

## After Scaling Summary
```text
                mean       std       min       max
Pclass -2.710114e-16  1.000694 -2.105950  0.685208
Age    -1.872443e-16  1.000694 -2.505671  2.586686
SibSp  -9.854962e-18  1.000694 -0.485984  5.373458
Parch  -6.282538e-17  1.000694 -0.410103  7.204088
Fare    5.420229e-17  1.000694 -1.283040  3.512724
```

## Dataset Size
- Shape before scaling: 721 rows and 14 columns
- Shape after scaling: 721 rows and 14 columns

## Output Snapshot
- Remaining missing values: 0
- Final processed dataset saved to: C:\Users\Aryan\OneDrive\Desktop\Titanic\data\processed\cleaned_titanic.csv