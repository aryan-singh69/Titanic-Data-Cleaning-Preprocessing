# Titanic Categorical Feature Handling & Encoding Report - Step 3

## Objective
Convert categorical information into model-ready numeric features while removing noisy identifier-like columns.

## Theory Summary
- Categorical data stores labels or groups instead of continuous numeric measurements.
- Machine learning models require numeric input because they perform mathematical operations on features.
- Label encoding converts categories into integers and is useful for binary variables.
- One-hot encoding creates separate binary columns for each category and is preferred for nominal variables without order.

## Column-Wise Decisions
### Name
- Information: Passenger name, often containing honorifics such as Mr, Mrs, Miss, and Master.
- Decision: Feature engineer Title and drop Name
- Reason: Raw names are high-cardinality text, but titles can capture useful social information.
- Unique values before encoding: 891

### Sex
- Information: Binary gender category.
- Decision: Label encode to 0/1
- Reason: Sex has only two values and works well as a binary numeric feature.
- Unique values before encoding: 2

### Ticket
- Information: Ticket identifier with mixed alphanumeric patterns.
- Decision: Drop column
- Reason: Ticket has very high cardinality and behaves more like an identifier than a stable predictor.
- Unique values before encoding: 681

### Embarked
- Information: Port of embarkation, a nominal categorical feature.
- Decision: One-hot encode
- Reason: Embarked has only a few categories and no natural order.
- Unique values before encoding: 3

## Before Encoding
- Shape: 891 rows and 11 columns
- Columns:
```text
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Embarked
```

## After Encoding
- Shape: 891 rows and 14 columns
- Columns:
```text
PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S, Title_Miss, Title_Mr, Title_Mrs, Title_Rare
```

## Added Columns
Embarked_Q, Embarked_S, Title_Miss, Title_Mr, Title_Mrs, Title_Rare

## Removed Columns
Name, Ticket, Embarked

## Encoded Columns
Embarked_Q, Embarked_S, Title_Miss, Title_Mr, Title_Mrs, Title_Rare

## Output Snapshot
- Final missing values: 0
- Dataframe ready for later steps: 891 rows and 14 columns