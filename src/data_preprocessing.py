"""
Data Preprocessing Script for Car Price Prediction
Handles data loading, cleaning, encoding, and feature engineering.
"""
# data_preprocessing.py

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other loading errors
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def handle_missing_values(df):
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    logger.info("Handling missing values...")
    
    initial_missing = df.isnull().sum().sum()
    logger.info(f"Total missing values: {initial_missing}")
    
    if initial_missing > 0:
        # Display missing values per column
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            logger.info(f"  {col}: {missing_count} missing values")
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"  Filled {col} missing values with mode: {mode_val}")
    
    final_missing = df.isnull().sum().sum()
    logger.info(f"Missing values after handling: {final_missing}")
    
    return df


def extract_brand(car_name):
    """
    Extract brand name from CarName.
    
    Args:
        car_name (str): Full car name
        
    Returns:
        str: Brand name
    """
    if pd.isna(car_name):
        return 'unknown'
    
    # Split by space and take the first part (brand name)
    brand = str(car_name).split()[0].lower()
    return brand


def feature_engineering(df):
    """
    Perform feature engineering on the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    logger.info("Performing feature engineering...")
    
    # Extract brand from CarName
    if 'CarName' in df.columns:
        df['brand'] = df['CarName'].apply(extract_brand)
        logger.info(f"Extracted brand feature. Unique brands: {df['brand'].nunique()}")
        logger.info(f"Brand distribution:\n{df['brand'].value_counts().head(10)}")
        
        # Optionally drop CarName if not needed (keeping it for now)
        # df.drop('CarName', axis=1, inplace=True)
    else:
        logger.warning("CarName column not found. Skipping brand extraction.")
    
    logger.info("Feature engineering completed.")
    return df


def encode_categorical_variables(df, target_col='price'):
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column to exclude from encoding
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    logger.info("Encoding categorical variables...")
    
    # Identify categorical columns (exclude target if it exists)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    logger.info(f"Categorical columns to encode: {categorical_cols}")
    
    # Perform one-hot encoding
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        logger.info(f"Shape after encoding: {df_encoded.shape}")
        return df_encoded
    else:
        logger.info("No categorical columns to encode.")
        return df


def split_features_target(df, target_col='price'):
    """
    Split dataframe into features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        tuple: (features_df, target_series)
    """
    logger.info(f"Splitting features and target (target: {target_col})...")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Drop car_ID if it exists (it's just an identifier)
    columns_to_drop = [target_col]
    if 'car_ID' in df.columns:
        columns_to_drop.append('car_ID')
        logger.info("Dropping car_ID column (identifier)")
    
    X = df.drop(columns=columns_to_drop)
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, y


def save_cleaned_data(X, y, output_path):
    """
    Save cleaned features and target to CSV.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target series
        output_path (str): Path to save the cleaned data
    """
    try:
        logger.info(f"Saving cleaned data to {output_path}...")
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory: {output_dir}")
        
        # Combine X and y for saving
        cleaned_df = X.copy()
        cleaned_df['price'] = y
        
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved successfully. Shape: {cleaned_df.shape}")
        
    except Exception as e:
        logger.error(f"Error saving cleaned data: {str(e)}")
        raise


def main():
    """
    Main preprocessing pipeline.
    """
    try:
        # Define paths
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
        raw_data_path = data_dir / 'raw' / 'CarPrice.csv'
        processed_data_path = data_dir / 'processed' / 'cleaned_data.csv'
        
        logger.info("=" * 60)
        logger.info("Starting Car Price Data Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df = load_data(raw_data_path)
        
        # Step 2: Handle missing values
        df = handle_missing_values(df)
        
        # Step 3: Feature engineering
        df = feature_engineering(df)
        
        # Step 4: Encode categorical variables
        df = encode_categorical_variables(df, target_col='price')
        
        # Step 5: Split features and target
        X, y = split_features_target(df, target_col='price')
        
        # Step 6: Save cleaned data
        save_cleaned_data(X, y, processed_data_path)
        
        logger.info("=" * 60)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 60)
        
        return X, y
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

