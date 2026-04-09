#!/usr/bin/env python3
"""
Test script for loading and using the full prediction pipeline
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.exceptions import NotFittedError
from sklearn.base import check_is_fitted
import traceback

def main():
    """Main function for testing the prediction pipeline"""
    parser = argparse.ArgumentParser(description='Test the full prediction pipeline')
    parser.add_argument('model_path', type=str, help='Path to the saved pipeline model')
    parser.add_argument('--test-data', type=str, help='Path to CSV file with test data')
    parser.add_argument('--single-sample', action='store_true', help='Test with a single random sample')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of random samples to test (default: 10)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of predictions')
    args = parser.parse_args()

    # Load the full prediction pipeline
    print(f"Loading prediction pipeline from: {args.model_path}")
    try:
        pipeline = joblib.load(args.model_path)
        print("Pipeline loaded successfully")
        
        # Check if the pipeline is fitted
        try:
            check_is_fitted(pipeline)
        except NotFittedError:
            print("Warning: Pipeline is not fitted! It may have been saved incorrectly.")

            # Try to check if it's a TransformedTargetRegressor with an unfitted regressor
            if hasattr(pipeline, 'regressor_'):
                print(f"Regressor type: {type(pipeline.regressor_).__name__}")
                try:
                    check_is_fitted(pipeline.regressor_)
                except NotFittedError:
                    print("Regressor inside TransformedTargetRegressor is not fitted")
            
            raise NotFittedError("The loaded pipeline has not been fitted. Please ensure you're loading a trained model.")
        
        print(f"Pipeline is fitted and ready for predictions")
        
        # Get feature names from the pipeline if available
        feature_names = get_feature_names(pipeline)
        target_names = ['x_01', 'x_02', 'x_03', 'x_04', 'x_05', 'x_06', 'x_07', 'x_08']  # Adjust as needed
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        return

    # Prepare test data
    if args.test_data:
        print(f"Loading test data from: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        print(f"Loaded {len(test_df)} samples")
        
        # Ensure required columns are present
        if feature_names:
            missing_cols = [col for col in feature_names if col not in test_df.columns]
            if missing_cols:
                print(f"Error: Missing columns in test data: {missing_cols}")
                return
            X_test = test_df[feature_names]
        else:
            print("Warning: Feature names not available in pipeline, using all columns")
            X_test = test_df
    elif args.single_sample:
        X_test = generate_sample(feature_names)
    else:
        X_test = generate_samples(args.batch_size, feature_names)

    try:
        # Make predictions
        print("\nMaking predictions...")
        predictions = pipeline.predict(X_test)
        
        # Convert to DataFrame for better display
        pred_df = pd.DataFrame(predictions, columns=target_names)
        
        # Display results
        print("\nPrediction Results:")
        print(pred_df.head(min(10, len(pred_df))))  # Show first 10 or all if less
        
        # Save predictions to CSV
        output_path = Path(args.model_path).parent / "test_predictions.csv"
        pred_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Generate visualization if requested
        if args.visualize:
            visualize_predictions(X_test, predictions, feature_names, target_names, output_path.parent)
    
    except NotFittedError as e:
        print(f"Prediction failed: {str(e)}")
        print("The model appears not to have been trained. Please check the model file.")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")

def get_feature_names(pipeline):
    """Extract feature names from the pipeline if available"""
    try:
        # For regular Pipeline
        if hasattr(pipeline, 'steps'):
            # If it's a TransformedTargetRegressor
            if hasattr(pipeline, 'regressor'):
                feature_pipeline = pipeline.regressor.named_steps.get('feature_preprocessing')
            else:
                feature_pipeline = pipeline.named_steps.get('feature_preprocessing')
            
            # If we can get feature names from the pipeline
            if hasattr(feature_pipeline, 'feature_names_in_'):
                return list(feature_pipeline.feature_names_in_)
        
        # For models trained with DataFrame input
        if hasattr(pipeline, 'feature_names_in_'):
            return list(pipeline.feature_names_in_)
    except Exception:
        pass
    
    # Default feature names (adjust based on your training data)
    return ["n", "eta", "sigma_y", "width", "height"]

def generate_sample(feature_names):
    """Generate a single random sample for testing"""
    print("Generating a single random sample")
    sample = {
        "n": np.random.uniform(0.4, 0.6),
        "eta": np.random.uniform(0.5, 1.5),
        "sigma_y": np.random.uniform(0.005, 0.02),
        "width": np.random.randint(5, 15),
        "height": np.random.randint(3, 8)
    }
    
    # Create DataFrame ensuring correct column order
    return pd.DataFrame([sample])[feature_names]

def generate_samples(n_samples, feature_names):
    """Generate multiple random samples for testing"""
    print(f"Generating {n_samples} random samples")
    samples = []
    for _ in range(n_samples):
        samples.append({
            "n": np.random.uniform(0.4, 0.6),
            "eta": np.random.uniform(0.5, 1.5),
            "sigma_y": np.random.uniform(0.005, 0.02),
            "width": np.random.randint(5, 15),
            "height": np.random.randint(3, 8)
        })
    
    return pd.DataFrame(samples)[feature_names]

def visualize_predictions(X_test, predictions, feature_names, target_names, output_dir):
    """Generate visualizations of predictions"""
    print("\nGenerating prediction visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    pred_df = pd.DataFrame(predictions, columns=target_names)
    results_df = pd.concat([X_test.reset_index(drop=True), pred_df], axis=1)
    
    # 1. Feature vs Prediction plots
    for feature in feature_names:
        for target in target_names:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=results_df, x=feature, y=target)
            plt.title(f"{target} vs {feature}")
            plt.tight_layout()
            plt.savefig(output_dir / f"{target}_vs_{feature}.png", dpi=150)
            plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = results_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature-Target Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
    
    # 3. Prediction distributions
    plt.figure(figsize=(12, 8))
    for i, target in enumerate(target_names):
        plt.subplot(3, 3, i+1)
        sns.histplot(predictions[:, i], kde=True)
        plt.title(f"{target} Prediction Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_distributions.png", dpi=150)
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()