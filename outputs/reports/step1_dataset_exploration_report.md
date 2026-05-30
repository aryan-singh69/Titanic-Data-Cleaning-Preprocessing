# Titanic Dataset Exploration Report - Step 1

## Objective
Understand the raw Titanic dataset before any cleaning or transformation.

## Dataset Overview
- Shape: 891 rows and 12 columns
- Target variable: Survived

## Columns
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

## Data Types
- PassengerId: int64
- Survived: int64
- Pclass: int64
- Name: str
- Sex: str
- Age: float64
- SibSp: int64
- Parch: int64
- Ticket: str
- Fare: float64
- Cabin: str
- Embarked: str

## Missing Values
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

## Missing Value Percentage
```text
             missing_percentage
PassengerId                0.00
Survived                   0.00
Pclass                     0.00
Name                       0.00
Sex                        0.00
Age                       19.87
SibSp                      0.00
Parch                      0.00
Ticket                     0.00
Fare                       0.00
Cabin                     77.10
Embarked                   0.22
```

## Numerical Features
PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare

## Categorical Features
Name, Sex, Ticket, Cabin, Embarked

## Statistical Summary
```text
             count unique                      top freq       mean         std   min     25%      50%    75%       max
PassengerId  891.0    NaN                      NaN  NaN      446.0  257.353842   1.0   223.5    446.0  668.5     891.0
Survived     891.0    NaN                      NaN  NaN   0.383838    0.486592   0.0     0.0      0.0    1.0       1.0
Pclass       891.0    NaN                      NaN  NaN   2.308642    0.836071   1.0     2.0      3.0    3.0       3.0
Name           891    891  Braund, Mr. Owen Harris    1        NaN         NaN   NaN     NaN      NaN    NaN       NaN
Sex            891      2                     male  577        NaN         NaN   NaN     NaN      NaN    NaN       NaN
Age          714.0    NaN                      NaN  NaN  29.699118   14.526497  0.42  20.125     28.0   38.0      80.0
SibSp        891.0    NaN                      NaN  NaN   0.523008    1.102743   0.0     0.0      0.0    1.0       8.0
Parch        891.0    NaN                      NaN  NaN   0.381594    0.806057   0.0     0.0      0.0    0.0       6.0
Ticket         891    681                   347082    7        NaN         NaN   NaN     NaN      NaN    NaN       NaN
Fare         891.0    NaN                      NaN  NaN  32.204208   49.693429   0.0  7.9104  14.4542   31.0  512.3292
Cabin          204    147                       G6    4        NaN         NaN   NaN     NaN      NaN    NaN       NaN
Embarked       889      3                        S  644        NaN         NaN   NaN     NaN      NaN    NaN       NaN
```

## Notes
- PassengerId is numeric in format but acts like an identifier and is usually not used as a predictive feature.
- Survived is the target variable for future modeling tasks.
- Age, Cabin, and Embarked contain missing values and will need preprocessing in later steps.