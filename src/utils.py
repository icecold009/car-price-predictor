"""
Utility Functions for Car Price Prediction
Common helper functions used across the project.
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Dict, Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_feature_importance(model, feature_names=None, top_n=20, figsize=(10, 8), save_path=None):
    """
    Plot feature importance for tree-based models or coefficients for linear models.
    
    Args:
        model: Trained model (RandomForest, GradientBoosting, or LinearRegression)
        feature_names (list, optional): List of feature names. If None, uses generic names.
        top_n (int): Number of top features to display (default: 20)
        figsize (tuple): Figure size (width, height)
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        # Check if model has feature_importances_ (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            title = 'Feature Importance'
        # Check if model has coefficients (linear models)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            title = 'Feature Coefficients (Absolute Values)'
        else:
            raise ValueError("Model does not support feature importance visualization. "
                           "Model must have 'feature_importances_' or 'coef_' attribute.")
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create dataframe for easier manipulation
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance and take top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'{title} (Top {top_n})', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        return plt.gcf()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
    
    Returns:
        dict: Dictionary containing various metrics:
            - R² Score
            - RMSE (Root Mean Squared Error)
            - MAE (Mean Absolute Error)
            - MAPE (Mean Absolute Percentage Error)
            - Max Error
            - Explained Variance Score
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check for matching shapes
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Explained Variance Score
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        metrics = {
            'R² Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'Max Error': max_error,
            'Explained Variance': explained_variance
        }
        
        logger.info("Metrics calculated successfully:")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise


def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath (str or Path): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other loading errors
    """
    try:
        filepath = Path(filepath)
        logger.info(f"Loading data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def save_model(model, filepath):
    """
    Save trained model to pickle file.
    
    Args:
        model: Trained model to save
        filepath (str or Path): Path to save the model
    
    Raises:
        Exception: For saving errors
    """
    try:
        filepath = Path(filepath)
        logger.info(f"Saving model to {filepath}")
        
        # Create directory if it doesn't exist
        output_dir = filepath.parent
        if output_dir and not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("Model saved successfully!")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(filepath):
    """
    Load trained model from pickle file.
    
    Args:
        filepath (str or Path): Path to the model file
    
    Returns:
        Trained model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: For other loading errors
    """
    try:
        filepath = Path(filepath)
        logger.info(f"Loading model from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info("Model loaded successfully!")
        
        return model
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

