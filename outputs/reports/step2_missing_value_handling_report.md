# Titanic Missing Value Handling Report - Step 2

## Objective
Handle missing values using a data-driven strategy before moving to encoding.

## Column-Wise Strategy
### Age
- Missing type: Numeric, moderate missingness
- Strategy: Fill with median
- Reason: Age is skewed and has missing values; median is robust to outliers.

### Embarked
- Missing type: Categorical, very low missingness
- Strategy: Fill with mode
- Reason: Embarked has only two missing values, so the most frequent category is appropriate.

### Cabin
- Missing type: Categorical, extremely high missingness
- Strategy: Drop column
- Reason: Cabin is missing for most rows, so imputation would add noise and unreliable assumptions.

## Before Missing Value Counts
```text
             missing_count
PassengerId              0
Survived                 0
Pclass                   0
Name                     0
Sex                      0
Age                    177
SibSp                    0
Parch                    0
Ticket                   0
Fare                     0
Cabin                  687
Embarked                 2
```

## After Missing Value Counts
```text
             missing_count
PassengerId              0
Survived                 0
Pclass                   0
Name                     0
Sex                      0
Age                      0
SibSp                    0
Parch                    0
Ticket                   0
Fare                     0
Embarked                 0
```

## Applied Values
- Age median used: 28.0
- Embarked mode used: S
- Dropped columns: Cabin

## Output Snapshot
- Shape after handling missing values: 891 rows and 11 columns
- Remaining missing values: 0