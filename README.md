# Fashion Forward Forecasting Pipeline

This project builds an end-to-end machine learning pipeline to predict whether a customer recommends a clothing product (`Recommended IND`) using:
- Numeric features (`Age`, `Positive Feedback Count`)
- Categorical features (`Clothing ID`, `Division Name`, `Department Name`, `Class Name`)
- Text features (`Title`, `Review Text`)

The final solution is implemented in the starter notebook with preprocessing, NLP feature engineering, model training, hyperparameter tuning, and held-out test evaluation.

## Repository Structure

- `starter/starter.ipynb`: Completed notebook with full pipeline workflow and results.
- `starter/data/reviews.csv`: Input dataset used by the notebook.
- `requirements.txt`: Python dependencies.

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
cd starter
jupyter notebook starter.ipynb
```

Or execute it from the command line:

```bash
./.venv/bin/jupyter nbconvert --to notebook --execute --inplace starter/starter.ipynb
```

## Modeling Approach

The notebook uses:
- `ColumnTransformer` to process each data type correctly.
- `Pipeline` to combine preprocessing and model training in one reusable object.
- `TfidfVectorizer` on both text columns with normalization, tokenization, stop-word filtering, and n-gram features.
- `RandomForestClassifier` with class balancing.
- `GridSearchCV` with stratified cross-validation for hyperparameter tuning.

## Evaluation

Evaluation is performed on a held-out test split and includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The tuned model is selected based on cross-validated F1 and evaluated once on test data.

## License

Licensed under the terms in [LICENSE.txt](LICENSE.txt).
