import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import r2_score
from itertools import product

def tune_hyperparameters(X_train, y_train, X_val, y_val, base_pipeline, param_grid):
    """
    Perform hyperparameter tuning using validation set
    
    Args:
        X_train (array): Training features
        y_train (array): Training targets
        X_val (array): Validation features
        y_val (array): Validation targets
        base_pipeline (Pipeline): Pipeline instance to tune
        param_grid (dict): Parameter grid for search
    
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    best_score = -np.inf
    best_params = None
    best_model = None
    
    # Create mapping from simple parameter names to pipeline parameter names
    param_map = {
        'degree': 'poly__degree',
        'alpha': 'model__alpha'
    }
    
    # Create valid parameter grid with correct pipeline parameter names
    valid_param_grid = {}
    for param, values in param_grid.items():
        if param in param_map:
            valid_param_grid[param_map[param]] = values
        else:
            valid_param_grid[param] = values
    
    # Calculate number of combinations
    combinations = list(get_param_combinations(valid_param_grid))
    print(f"Starting hyperparameter tuning with {len(combinations)} combinations")
    
    # Iterate over parameter combinations
    for params in combinations:
        # Clone the pipeline to ensure a fresh copy
        current_pipeline = clone(base_pipeline)
        
        try:
            # Set parameters dynamically using the param_grid keys
            current_pipeline.set_params(**params)
        except ValueError as e:
            print(f"Warning: Could not set parameters {params}: {str(e)}")
            continue
        
        # Train model
        current_pipeline.fit(X_train, y_train)
        
        # Generate predictions for evaluation
        y_pred = current_pipeline.predict(X_val)
        
        # Calculate R² score explicitly
        score = r2_score(y_val, y_pred)
        
        # Update best model if improved
        if score > best_score:
            best_score = score
            best_params = params
            best_model = current_pipeline
            print(f"New best: R²={best_score:.4f} with {params}")
    
    if best_model is None:
        raise RuntimeError("Hyperparameter tuning failed - no valid parameter combination found")
    
    print(f"Tuning complete. Best R²: {best_score:.4f}")
    return best_model, best_params, best_score

def get_param_combinations(param_grid):
    """
    Generate parameter combinations from grid
    
    Args:
        param_grid (dict): Parameter grid with values as lists
        
    Yields:
        dict: Dictionary of parameter combinations
    """
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

def grid_search_tuning(X_train, y_train, base_pipeline, param_grid, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV with cross-validation
    
    Args:
        X_train (array): Training features
        y_train (array): Training targets
        base_pipeline (Pipeline): Base pipeline to tune
        param_grid (dict): Parameter grid for search
        cv (int): Number of cross-validation folds
    
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    # Create mapping from simple parameter names to pipeline parameter names
    param_map = {
        'degree': 'poly__degree',
        'alpha': 'model__alpha'
    }
    
    # Create valid parameter grid with correct pipeline parameter names
    valid_param_grid = {}
    for param, values in param_grid.items():
        if param in param_map:
            valid_param_grid[param_map[param]] = values
        else:
            valid_param_grid[param] = values
    
    # Clone the base pipeline to avoid modifying the original
    pipeline_clone = clone(base_pipeline)
    
    # Configure GridSearchCV
    gs = GridSearchCV(
        estimator=pipeline_clone,
        param_grid=valid_param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        error_score='raise'
    )
    
    print(f"Starting GridSearchCV with {cv}-fold cross-validation")
    print(f"Parameter grid: {valid_param_grid}")
    
    # Execute grid search
    gs.fit(X_train, y_train)
    
    print(f"Best parameters: {gs.best_params_}")
    print(f"Best cross-validation R²: {gs.best_score_:.4f}")
    
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def select_tuning_method(X_train, y_train, X_val, y_val, base_pipeline, param_grid, 
                         use_cv=False, cv_folds=5):
    """
    Select appropriate tuning method based on dataset size and resources
    
    Args:
        use_cv (bool): Whether to use cross-validation
        cv_folds (int): Number of folds if using cross-validation
        
    Returns:
        tuple: (best_model, best_params, best_score)
    """
    if use_cv:
        return grid_search_tuning(X_train, y_train, base_pipeline, param_grid, cv=cv_folds)
    else:
        return tune_hyperparameters(X_train, y_train, X_val, y_val, base_pipeline, param_grid)