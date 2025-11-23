# Loan Payback Prediction

This project aims to predict the probability that a borrower will pay back their loan using various machine learning models and ensemble techniques.

## Project Structure

*   `data/`: Contains the dataset (`train.csv`, `test.csv`, `sample_submission.csv`).
*   `eda.ipynb`: Exploratory Data Analysis notebook.
*   `model.ipynb`: Initial XGBoost model implementation.
*   `ensemble_modeling.ipynb`: Ensemble of XGBoost and Logistic Regression using Voting Classifier.
*   `advanced_ensemble.ipynb`: Advanced Stacking Ensemble with XGBoost, Logistic Regression, and MLP (Neural Network), tuned with Optuna.
*   `requirements.txt`: List of Python dependencies.

## Models & Techniques

### 1. Feature Engineering
*   **Log Transformation**: Applied to skewed features like `annual_income`, `loan_amount`, and `debt_to_income_ratio`.
*   **Interaction Features**: Created `loan_to_income` and `disposable_income`.
*   **Encoding**: Ordinal encoding for `grade_subgrade` and One-Hot encoding for other categorical variables.
*   **Scaling**: Standard scaling for Logistic Regression and MLP.

### 2. Models
*   **XGBoost**: Gradient Boosting Decision Trees.
*   **Logistic Regression**: Linear model with L1/L2 regularization.
*   **MLPClassifier**: Multi-Layer Perceptron (Neural Network).

### 3. Hyperparameter Tuning
*   **RandomizedSearchCV**: Used in `ensemble_modeling.ipynb`.
*   **Optuna**: Bayesian Optimization used in `advanced_ensemble.ipynb` for efficient hyperparameter search.

### 4. Ensembling
*   **VotingClassifier**: Soft voting (averaging probabilities).
*   **StackingClassifier**: Uses a meta-learner (Logistic Regression) to combine predictions from base models (XGBoost, LR, MLP).

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the notebooks in Jupyter Lab or VS Code.
    *   Start with `eda.ipynb` to understand the data.
    *   Run `advanced_ensemble.ipynb` for the best performing model.

3.  The submission file `submission_stacking.csv` will be generated in the root directory.