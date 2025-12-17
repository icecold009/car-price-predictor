"""
Main Pipeline Script for Car Price Prediction
Orchestrates the entire machine learning pipeline with command-line interface.
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules from src/
try:
    from src import data_preprocessing
    from src import model_training
    from src import prediction
    from src import utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    """
    Run data preprocessing pipeline.
    
    Returns:
        tuple: (X, y) features and target
    """
    logger.info("=" * 80)
    logger.info("RUNNING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    try:
        X, y = data_preprocessing.main()
        logger.info("Preprocessing completed successfully!")
        return X, y
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)


def run_training():
    """
    Run model training pipeline.
    
    Returns:
        tuple: (best_model, models_results)
    """
    logger.info("=" * 80)
    logger.info("RUNNING MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    
    try:
        best_model, models_results = model_training.main()
        
        # Plot feature importance for the best model
        try:
            logger.info("\nGenerating feature importance plot...")
            # Load cleaned data to get feature names
            script_dir = Path(__file__).parent
            processed_data_path = script_dir / 'data' / 'processed' / 'cleaned_data.csv'
            
            if processed_data_path.exists():
                df = utils.load_data(processed_data_path)
                feature_names = [col for col in df.columns if col != 'price']
                
                # Save feature importance plot
                plot_path = script_dir / 'models' / 'feature_importance.png'
                utils.plot_feature_importance(
                    best_model,
                    feature_names=feature_names,
                    top_n=20,
                    save_path=plot_path
                )
                logger.info(f"Feature importance plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {str(e)}")
        
        logger.info("Training completed successfully!")
        return best_model, models_results
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


def run_prediction(input_file=None, single_input=None):
    """
    Run prediction pipeline.
    
    Args:
        input_file (str, optional): Path to CSV file with features for batch prediction
        single_input (dict, optional): Dictionary with features for single prediction
    """
    logger.info("=" * 80)
    logger.info("RUNNING PREDICTION PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Load model
        model = prediction.load_model()
        logger.info("Model loaded successfully!")
        
        if input_file:
            # Batch prediction from file
            logger.info(f"Loading input data from {input_file}")
            input_df = utils.load_data(input_file)
            
            # Remove price column if it exists (for evaluation)
            has_price = 'price' in input_df.columns
            if has_price:
                y_true = input_df['price']
                input_features = input_df.drop(columns=['price'])
            else:
                input_features = input_df
            
            # Make predictions
            predictions = prediction.predict_batch(model, input_features)
            
            # Display results
            print("\n" + "=" * 80)
            print("BATCH PREDICTION RESULTS")
            print("=" * 80)
            
            results_df = input_features.copy()
            results_df['predicted_price'] = predictions
            
            if has_price:
                results_df['actual_price'] = y_true
                results_df['error'] = results_df['actual_price'] - results_df['predicted_price']
                results_df['error_pct'] = (results_df['error'] / results_df['actual_price']) * 100
                
                # Calculate metrics
                metrics = utils.calculate_metrics(y_true, predictions)
                
                print("\nPrediction Summary:")
                print(f"  Total predictions: {len(predictions)}")
                print(f"  Mean predicted price: ${predictions.mean():.2f}")
                print(f"  Min predicted price: ${predictions.min():.2f}")
                print(f"  Max predicted price: ${predictions.max():.2f}")
                print("\nEvaluation Metrics:")
                for metric_name, metric_value in metrics.items():
                    if not np.isnan(metric_value):
                        print(f"  {metric_name}: {metric_value:.4f}")
            
            # Save results
            output_file = Path(input_file).parent / f"predictions_{Path(input_file).stem}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"\nPredictions saved to {output_file}")
            
            # Display first few predictions
            print("\nFirst 5 Predictions:")
            print(results_df.head().to_string())
            
        elif single_input:
            # Single prediction
            price = prediction.predict_price(model, single_input)
            print("\n" + "=" * 80)
            print("PREDICTION RESULT")
            print("=" * 80)
            print(f"\nPredicted Car Price: ${price:,.2f}")
            print("=" * 80)
            
        else:
            logger.error("Please provide either --input_file or --single_input for prediction")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


def run_full_pipeline():
    """
    Run the complete pipeline: preprocessing -> training -> results.
    """
    logger.info("=" * 80)
    logger.info("RUNNING FULL PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Preprocessing
    X, y = run_preprocessing()
    
    # Step 2: Training
    best_model, models_results = run_training()
    
    # Step 3: Display final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Use the trained model for predictions:")
    print("     python main.py --predict --input_file <path_to_features.csv>")
    print("  2. Or make a single prediction:")
    print("     python main.py --predict --single_input '{\"feature1\": value1, ...}'")
    print("=" * 80)


def parse_single_input(input_str):
    """
    Parse single input string (JSON-like format) to dictionary.
    
    Args:
        input_str (str): String representation of input dictionary
        
    Returns:
        dict: Parsed input dictionary
    """
    try:
        import json
        # Try JSON parsing first
        return json.loads(input_str)
    except json.JSONDecodeError:
        try:
            # Try eval as fallback (less safe but more flexible)
            import ast
            return ast.literal_eval(input_str)
        except:
            raise ValueError(f"Could not parse input string: {input_str}")


def main():
    """
    Main entry point with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Car Price Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py
  
  # Run only preprocessing
  python main.py --preprocess
  
  # Run only training
  python main.py --train
  
  # Make batch predictions from CSV file
  python main.py --predict --input_file data/test_features.csv
  
  # Make single prediction
  python main.py --predict --single_input '{"symboling": 3, "enginesize": 130, "horsepower": 111}'
        """
    )
    
    # Add mutually exclusive group for modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--preprocess',
        action='store_true',
        help='Run only data preprocessing'
    )
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Run only model training'
    )
    mode_group.add_argument(
        '--predict',
        action='store_true',
        help='Run prediction mode'
    )
    
    # Prediction-specific arguments
    parser.add_argument(
        '--input_file',
        type=str,
        help='Path to CSV file with features for batch prediction'
    )
    parser.add_argument(
        '--single_input',
        type=str,
        help='Dictionary string with features for single prediction (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Determine which mode to run
    if args.preprocess:
        run_preprocessing()
    elif args.train:
        run_training()
    elif args.predict:
        single_input_dict = None
        if args.single_input:
            single_input_dict = parse_single_input(args.single_input)
        run_prediction(input_file=args.input_file, single_input=single_input_dict)
    else:
        # Default: run full pipeline
        run_full_pipeline()


if __name__ == "__main__":
    main()

