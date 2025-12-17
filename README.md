# Car Price Predictor 🚗💰

A machine learning project that predicts car prices based on various vehicle features using multiple regression models. The project includes a complete end-to-end pipeline from data preprocessing to model training and prediction.

## 📋 Table of Contents

- [Project Description](#project-description)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Description

Car Price Predictor is a comprehensive machine learning project designed to predict car prices based on various vehicle characteristics such as engine specifications, fuel type, body type, and other technical features. The project implements multiple regression models (Linear Regression, Random Forest, and Gradient Boosting) and automatically selects the best-performing model based on cross-validation metrics.

### Key Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and performs feature engineering
- **Multiple Models**: Compares Linear Regression, Random Forest, and Gradient Boosting models
- **Cross-Validation**: Uses 5-fold cross-validation for robust model evaluation
- **Automated Selection**: Automatically selects the best model based on R² score and RMSE
- **Batch & Single Predictions**: Supports both batch predictions from CSV files and single instance predictions
- **Feature Importance**: Visualizes feature importance for tree-based models
- **Comprehensive Metrics**: Calculates R², RMSE, MAE, MAPE, and other evaluation metrics

## 📊 Dataset Information

The project uses a car price dataset containing information about various car models with their specifications and prices.

### Dataset Features

- **26 features** including:
  - Car specifications (symboling, wheelbase, car length, width, height, curb weight)
  - Engine details (engine type, cylinder number, engine size, fuel system, boreratio, stroke, compression ratio, horsepower, peak RPM)
  - Fuel and performance metrics (fuel type, aspiration, city MPG, highway MPG)
  - Car body and design (door number, car body, drive wheel, engine location)
  - Brand information (extracted from car name)

- **Target Variable**: `price` (car price in USD)

### Dataset Source

The dataset (`CarPrice.csv`) is located in `data/raw/` directory. A data dictionary (`Data Dictionary - carprices.xlsx`) is also provided for reference.

**Note**: If you need to use a different dataset, ensure it contains similar features or modify the preprocessing pipeline accordingly.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/car-price-predictor.git
   cd car-price-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn; print('All packages installed successfully!')"
   ```

## 💻 Usage

### Running the Full Pipeline

To run the complete pipeline (preprocessing → training → results):

```bash
python main.py
```

### Running Individual Steps

#### 1. Data Preprocessing Only

```bash
python main.py --preprocess
```

This will:
- Load data from `data/raw/CarPrice.csv`
- Handle missing values
- Encode categorical variables
- Perform feature engineering (extract brand from car name)
- Save cleaned data to `data/processed/cleaned_data.csv`

#### 2. Model Training Only

```bash
python main.py --train
```

This will:
- Load cleaned data from `data/processed/cleaned_data.csv`
- Split data into train/test sets (80/20)
- Train Linear Regression, Random Forest, and Gradient Boosting models
- Perform 5-fold cross-validation
- Select the best model based on R² and RMSE
- Save the best model to `models/car_price_model.pkl`
- Generate feature importance plot

#### 3. Making Predictions

**Batch Prediction from CSV File:**

```bash
python main.py --predict --input_file data/test_features.csv
```

The input CSV should contain feature columns matching the trained model. If the file includes a `price` column, the script will calculate evaluation metrics.

**Single Prediction:**

```bash
python main.py --predict --single_input '{"symboling": 3, "enginesize": 130, "horsepower": 111, "citympg": 21, "highwaympg": 27}'
```

**Note**: For single predictions, you need to provide all features that were used during training. Check `data/processed/cleaned_data.csv` for the complete list of feature columns.

### Command-Line Help

```bash
python main.py --help
```

## 📁 Project Structure

```
car-price-predictor/
│
├── data/
│   ├── raw/
│   │   └── CarPrice.csv              # Original dataset
│   └── processed/
│       └── cleaned_data.csv           # Preprocessed data
│
├── models/
│   ├── car_price_model.pkl           # Trained model (saved after training)
│   └── feature_importance.png        # Feature importance visualization
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Jupyter notebook for EDA
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data cleaning and preprocessing
│   ├── model_training.py             # Model training and evaluation
│   ├── prediction.py                 # Prediction functions
│   └── utils.py                      # Utility functions
│
├── main.py                           # Main pipeline orchestrator
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── Data Dictionary - carprices.xlsx  # Dataset documentation
```

### Module Descriptions

- **`data_preprocessing.py`**: Handles data loading, missing value imputation, categorical encoding, and feature engineering
- **`model_training.py`**: Trains multiple models, performs cross-validation, selects best model, and evaluates performance
- **`prediction.py`**: Provides functions for loading models and making predictions (single and batch)
- **`utils.py`**: Contains utility functions for metrics calculation, feature importance plotting, and model I/O
- **`main.py`**: Orchestrates the entire pipeline with command-line interface

## 📈 Model Performance

The project trains and compares three regression models:

1. **Linear Regression**: Baseline model for comparison
2. **Random Forest Regressor**: Ensemble method with 100 trees
3. **Gradient Boosting Regressor**: Boosting ensemble with 100 estimators

### Evaluation Metrics

The models are evaluated using:
- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Expected Performance

After training, the script displays:
- Cross-validation scores for each model
- Test set performance metrics
- Model comparison table
- Best model selection based on composite score

**Note**: Actual performance metrics will vary based on the dataset and model hyperparameters. Run the training pipeline to see current performance metrics.

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive notebook environment for EDA
- **pickle**: Model serialization (built-in)

## 🔮 Future Improvements

- [ ] **Hyperparameter Tuning**: Implement grid search or random search for optimal hyperparameters
- [ ] **Feature Selection**: Add automated feature selection techniques
- [ ] **Advanced Models**: Include XGBoost, LightGBM, and Neural Networks
- [ ] **Model Interpretability**: Add SHAP values and LIME explanations
- [ ] **API Development**: Create REST API using Flask/FastAPI for web integration
- [ ] **Docker Support**: Containerize the application for easy deployment
- [ ] **CI/CD Pipeline**: Set up automated testing and deployment
- [ ] **Data Validation**: Add data validation schemas using Pydantic
- [ ] **Experiment Tracking**: Integrate MLflow or Weights & Biases
- [ ] **Automated Retraining**: Schedule periodic model retraining
- [ ] **Web Interface**: Build a user-friendly web interface for predictions
- [ ] **Database Integration**: Store predictions and model versions in a database
- [ ] **Unit Tests**: Add comprehensive test coverage
- [ ] **Documentation**: Expand code documentation and add API docs

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Include comments for complex logic
4. **Test your changes**
   - Ensure existing functionality still works
   - Add tests for new features if applicable
5. **Commit your changes**
   ```bash
   git commit -m "Add: Description of your changes"
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Contribution Guidelines

- Write clear, descriptive commit messages
- Keep code changes focused and atomic
- Update documentation for new features
- Ensure backward compatibility when possible
- Add logging for debugging purposes
- Follow the existing code style

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset providers and contributors
- scikit-learn community for excellent ML tools
- Open source community for inspiration and support

## 📞 Support

If you encounter any issues or have questions:

1. Check existing [Issues](https://github.com/yourusername/car-price-predictor/issues)
2. Create a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS information

---

**Happy Predicting! 🚀**

