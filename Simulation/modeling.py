from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

def create_pipeline(model_type='ridge', degree=2, alpha=0.0, multi_output=False):
    """
    Creates a modeling pipeline containing feature expansion, scaling, and a regressor.
    """
    # Define the sequence of steps
    steps = [
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler())
    ]
    
    # Select the regression model
    if model_type == 'ridge':
        model = Ridge(alpha=alpha, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha, random_state=42)
    elif model_type == 'svr':
        # Corrected: SVR does not accept a 'random_state' parameter
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    else:
        model = Ridge(alpha=alpha, random_state=42)

    # Wrap the model for multi-output regression if needed
    if multi_output:
        model = MultiOutputRegressor(model)
    
    steps.append(('model', model))
    
    return Pipeline(steps)