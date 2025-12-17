"""
Model Training Script for Car Price Prediction
Trains multiple models and selects the best one based on performance metrics.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cleaned_data(file_path):
    """
    Load cleaned data from CSV file.
    
    Args:
        file_path (str): Path to the cleaned data CSV file
        
    Returns:
        tuple: (features_df, target_series)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other loading errors
    """
    try:
        logger.info(f"Loading cleaned data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Separate features and target
        if 'price' not in df.columns:
            raise ValueError("Target column 'price' not found in the data")
        
        X = df.drop(columns=['price'])
        y = df['price']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        return X, y
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target series
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data into train/test sets ({1-test_size:.0%}/{test_size:.0%})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train, cv_folds=5):
    """
    Train Linear Regression model with cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        tuple: (model, cv_r2_scores, cv_rmse_scores)
    """
    logger.info("Training Linear Regression model...")
    
    model = LinearRegression()
    
    # Cross-validation for R² score
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    # Cross-validation for RMSE (negative MSE, so we take negative)
    cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    logger.info(f"  CV R² Score: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    logger.info(f"  CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")
    
    return model, cv_r2_scores, cv_rmse_scores


def train_random_forest(X_train, y_train, cv_folds=5, n_estimators=100, random_state=42):
    """
    Train Random Forest model with cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed
        
    Returns:
        tuple: (model, cv_r2_scores, cv_rmse_scores)
    """
    logger.info("Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Cross-validation for R² score
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    # Cross-validation for RMSE
    cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    logger.info(f"  CV R² Score: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    logger.info(f"  CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")
    
    return model, cv_r2_scores, cv_rmse_scores


def train_gradient_boosting(X_train, y_train, cv_folds=5, n_estimators=100, random_state=42):
    """
    Train Gradient Boosting model with cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        n_estimators (int): Number of boosting stages
        random_state (int): Random seed
        
    Returns:
        tuple: (model, cv_r2_scores, cv_rmse_scores)
    """
    logger.info("Training Gradient Boosting model...")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Cross-validation for R² score
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    # Cross-validation for RMSE
    cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(cv_mse_scores)
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    logger.info(f"  CV R² Score: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
    logger.info(f"  CV RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")
    
    return model, cv_r2_scores, cv_rmse_scores


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model on test set and return metrics.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    }
    
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    
    return metrics, y_pred


def select_best_model(models_results):
    """
    Select the best model based on R² score and RMSE.
    
    Args:
        models_results (list): List of dictionaries containing model info and metrics
        
    Returns:
        dict: Best model information
    """
    logger.info("Selecting best model...")
    
    # Score each model (higher R² and lower RMSE is better)
    # We'll use a composite score: R² - normalized RMSE
    best_model = None
    best_score = -np.inf
    
    for result in models_results:
        # Normalize RMSE by dividing by mean of target (for comparison)
        # Use CV scores for selection
        avg_r2 = result['cv_r2_mean']
        avg_rmse = result['cv_rmse_mean']
        
        # Composite score: prioritize R², penalize high RMSE
        # Normalize RMSE by dividing by a reference value (e.g., mean of training target)
        score = avg_r2 - (avg_rmse / result.get('target_mean', 1))
        
        logger.info(f"  {result['name']}: Composite Score = {score:.4f} (R²={avg_r2:.4f}, RMSE={avg_rmse:.4f})")
        
        if score > best_score:
            best_score = score
            best_model = result
    
    logger.info(f"Best model selected: {best_model['name']}")
    logger.info(f"  R² Score: {best_model['cv_r2_mean']:.4f}")
    logger.info(f"  RMSE: {best_model['cv_rmse_mean']:.4f}")
    
    return best_model


def save_model(model, file_path):
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained model to save
        file_path (str): Path to save the model
    """
    try:
        logger.info(f"Saving model to {file_path}...")
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("Model saved successfully!")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def print_evaluation_summary(models_results, best_model):
    """
    Print comprehensive evaluation summary.
    
    Args:
        models_results (list): List of all model results
        best_model (dict): Best model information
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\nCross-Validation Results:")
    print("-" * 80)
    print(f"{'Model':<25} {'CV R² Score':<15} {'CV RMSE':<15} {'Test R²':<15} {'Test RMSE':<15}")
    print("-" * 80)
    
    for result in models_results:
        print(f"{result['name']:<25} "
              f"{result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}  "
              f"{result['cv_rmse_mean']:.4f}±{result['cv_rmse_std']:.4f}  "
              f"{result['test_r2']:.4f}      "
              f"{result['test_rmse']:.4f}")
    
    print("\n" + "=" * 80)
    print("BEST MODEL DETAILED METRICS")
    print("=" * 80)
    print(f"Model: {best_model['name']}")
    print(f"\nCross-Validation Metrics:")
    print(f"  R² Score: {best_model['cv_r2_mean']:.4f} (±{best_model['cv_r2_std']:.4f})")
    print(f"  RMSE: {best_model['cv_rmse_mean']:.4f} (±{best_model['cv_rmse_std']:.4f})")
    
    print(f"\nTest Set Metrics:")
    print(f"  R² Score: {best_model['test_r2']:.4f}")
    print(f"  RMSE: {best_model['test_rmse']:.4f}")
    print(f"  MAE: {best_model['test_mae']:.4f}")
    print(f"  MAPE: {best_model['test_mape']:.2f}%")
    
    print("\n" + "=" * 80)


def main():
    """
    Main model training pipeline.
    """
    try:
        # Define paths
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
        processed_data_path = data_dir / 'processed' / 'cleaned_data.csv'
        model_path = script_dir.parent / 'models' / 'car_price_model.pkl'
        
        logger.info("=" * 80)
        logger.info("Starting Car Price Model Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Load cleaned data
        X, y = load_cleaned_data(processed_data_path)
        
        # Step 2: Split data into train/test sets (80/20)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Step 3: Train multiple models with cross-validation
        models_results = []
        
        # Linear Regression
        lr_model, lr_cv_r2, lr_cv_rmse = train_linear_regression(X_train, y_train, cv_folds=5)
        lr_test_metrics, _ = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
        models_results.append({
            'name': 'Linear Regression',
            'model': lr_model,
            'cv_r2_mean': lr_cv_r2.mean(),
            'cv_r2_std': lr_cv_r2.std(),
            'cv_rmse_mean': lr_cv_rmse.mean(),
            'cv_rmse_std': lr_cv_rmse.std(),
            'test_r2': lr_test_metrics['R² Score'],
            'test_rmse': lr_test_metrics['RMSE'],
            'test_mae': lr_test_metrics['MAE'],
            'test_mape': lr_test_metrics['MAPE (%)'],
            'target_mean': y_train.mean()
        })
        
        # Random Forest
        rf_model, rf_cv_r2, rf_cv_rmse = train_random_forest(X_train, y_train, cv_folds=5, n_estimators=100)
        rf_test_metrics, _ = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        models_results.append({
            'name': 'Random Forest',
            'model': rf_model,
            'cv_r2_mean': rf_cv_r2.mean(),
            'cv_r2_std': rf_cv_r2.std(),
            'cv_rmse_mean': rf_cv_rmse.mean(),
            'cv_rmse_std': rf_cv_rmse.std(),
            'test_r2': rf_test_metrics['R² Score'],
            'test_rmse': rf_test_metrics['RMSE'],
            'test_mae': rf_test_metrics['MAE'],
            'test_mape': rf_test_metrics['MAPE (%)'],
            'target_mean': y_train.mean()
        })
        
        # Gradient Boosting
        gb_model, gb_cv_r2, gb_cv_rmse = train_gradient_boosting(X_train, y_train, cv_folds=5, n_estimators=100)
        gb_test_metrics, _ = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        models_results.append({
            'name': 'Gradient Boosting',
            'model': gb_model,
            'cv_r2_mean': gb_cv_r2.mean(),
            'cv_r2_std': gb_cv_r2.std(),
            'cv_rmse_mean': gb_cv_rmse.mean(),
            'cv_rmse_std': gb_cv_rmse.std(),
            'test_r2': gb_test_metrics['R² Score'],
            'test_rmse': gb_test_metrics['RMSE'],
            'test_mae': gb_test_metrics['MAE'],
            'test_mape': gb_test_metrics['MAPE (%)'],
            'target_mean': y_train.mean()
        })
        
        # Step 4: Select best model
        best_model = select_best_model(models_results)
        
        # Step 5: Save best model
        save_model(best_model['model'], model_path)
        
        # Step 6: Print evaluation summary
        print_evaluation_summary(models_results, best_model)
        
        logger.info("=" * 80)
        logger.info("Model training pipeline completed successfully!")
        logger.info("=" * 80)
        
        return best_model['model'], models_results
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

