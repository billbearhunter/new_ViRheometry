import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# =========================================================================
# 1. CUSTOM TRANSFORMERS
# =========================================================================

class TransformationTransformer(BaseEstimator, TransformerMixin):
    """Applies a mathematical transformation to the data."""
    def __init__(self, method='log'):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.method:
            return X
        print(f"Applying transformation: {self.method}")
        X_transformed = np.copy(X)
        if self.method == 'log':
            X_transformed = np.log(X_transformed + 1e-6)
        return X_transformed

class SmoothingTransformer(BaseEstimator, TransformerMixin):
    """Applies a moving average to each feature column."""
    def __init__(self, window_size=5):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.window_size <= 1:
            return X
        print(f"Applying smoothing with window size: {self.window_size}")
        return pd.DataFrame(X).rolling(window=self.window_size, min_periods=1, center=True).mean().values

class DetrendingTransformer(BaseEstimator, TransformerMixin):
    """Applies polynomial detrending to each feature column."""
    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.degree < 1:
            return X
        print(f"Applying detrending with degree: {self.degree}")
        X_detrended = np.copy(X)
        for i in range(X.shape[1]):
            x_axis = np.arange(len(X_detrended[:, i]))
            coeffs = np.polyfit(x_axis, X_detrended[:, i], self.degree)
            trend = np.polyval(coeffs, x_axis)
            X_detrended[:, i] = X_detrended[:, i] - trend
        return X_detrended

# =========================================================================
# 2. DATA CLEANING FUNCTION
# =========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic, rule-based data cleaning (non-statistical).
    """
    print("--- Running Data Cleaning ---")
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    if 'eta' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['eta'] >= 0]
    print(f"Data cleaning complete. Final shape: {df_cleaned.shape}")
    return df_cleaned

# =========================================================================
# 3. DYNAMIC FEATURE PIPELINE CREATION
# =========================================================================

def create_feature_pipeline(numerical_features: list, categorical_features: list, args) -> ColumnTransformer:
    """
    Creates a dynamic scikit-learn pipeline for feature preprocessing
    based on command-line arguments.
    """
    print("--- Creating Dynamic Feature Pipeline ---")

    # Start building the pipeline for numerical features
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]

    # Conditionally add the transformation step
    if args.transformation:
        numeric_steps.append(('transform', TransformationTransformer(method=args.transformation)))
    
    # Conditionally add the smoothing step
    if args.smooth_window > 1:
        numeric_steps.append(('smooth', SmoothingTransformer(window_size=args.smooth_window)))

    # Conditionally add the detrending step
    if args.detrend_degree > 0:
        numeric_steps.append(('detrend', DetrendingTransformer(degree=args.detrend_degree)))

    numeric_transformer = Pipeline(steps=numeric_steps)

    # Define the pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    print("Dynamic feature pipeline created successfully.")
    return preprocessor