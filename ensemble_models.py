"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# Признаки German Credit Dataset
NOMINAL_FEATURES = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Sex_Marital_Status", "Guarantors", "Most_valuable_available_asset",
    "Concurrent_Credits", "Type_of_apartment", "Occupation", "Telephone", "Foreign_Worker"
]

NUMERIC_FEATURES = [
    "Duration_of_Credit_monthly", "Credit_Amount", "Value_Savings_Stocks",
    "Length_of_current_employment", "Instalment_per_cent", "Duration_in_Current_address",
    "Age_years", "No_of_Credits_at_this_Bank", "No_of_dependents"
]

ALL_FEATURES = NOMINAL_FEATURES + NUMERIC_FEATURES


def _to_df(X, default_cols):
    """Безопасное преобразование входных данных в DataFrame."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        if list(df.columns) != default_cols and len(df.columns) == len(default_cols):
            df.columns = default_cols
        return df
    return pd.DataFrame(X, columns=default_cols)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test."""

    X_train = _to_df(data.train_features, ALL_FEATURES)
    y_train = np.asarray(data.train_target).ravel()
    X_test = _to_df(data.test_features, ALL_FEATURES)
    y_test = np.asarray(data.test_target).ravel()
    seed = getattr(data, "random_seed", 42)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, NOMINAL_FEATURES),
            ('num', numeric_transformer, NUMERIC_FEATURES)
        ],
        remainder='drop'
    )

   
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_depth=5, l2_regularization=1.0,
        random_state=seed
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=seed, n_jobs=-1
    )
    lr = LogisticRegression(
        C=0.1, max_iter=1000, random_state=seed
    )


    ensemble = VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf), ('lr', lr)],
        voting='soft',
        weights=[2, 1, 1]  
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    param_grid = {
        'classifier__hgb__max_depth': [4, 5, 6],
        'classifier__hgb__learning_rate': [0.05, 0.1],
        'classifier__rf__max_depth': [6, 8, 10]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_model_name = f"VotingSoft(HGB+RF+LR) | Best: {search.best_params_}"
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return EnsembleResult(
        model_name=best_model_name,
        probabilities=y_pred_proba.tolist(), # Сериализуемо для JSON
        predictions=y_pred.tolist(),
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )