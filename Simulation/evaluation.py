"""Model evaluation metrics and calculations"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


def evaluate_and_print_metrics(y_true, y_pred, target_names):
    """
    Calculates and prints regression metrics for each target variable.
    This function handles the core evaluation logic.
    """
    
    # Ensure y_true is a numpy array if it's a pandas DataFrame
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    all_metrics = []
    print("\n--- Model Performance on Test Set ---")
    for i, target_name in enumerate(target_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        
        metrics = {
            'Target': target_name,
            'R²': r2,
            'MSE': mse,
            'MAE': mae
        }
        all_metrics.append(metrics)

    # Print metrics in a clean table format for the log file
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    return metrics_df


def cross_validate_model(pipeline, X, y, cv=5):
    """
    Perform cross-validation and return mean and std of MSE.
    This function remains for optional, more rigorous validation checks.
    
    Args:
        pipeline: Full preprocessing and modeling pipeline
        X (array): Feature matrix
        y (array): Target values
        cv (int): Number of cross-validation folds
    
    Returns:
        tuple: (mean MSE, std MSE)
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Convert to positive MSE
    return np.mean(mse_scores), np.std(mse_scores)