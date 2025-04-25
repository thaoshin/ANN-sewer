# Sewer Pipe Condition Prediction using Artificial Neural Networks

This project implements a deep learning solution for predicting sewer pipe failures, start with (INL) using various pipe characteristics and soil data. The implementation includes data preprocessing, model training with advanced features, and comprehensive evaluation metrics.

## Project Structure

```
.
├── data_preparation.py    # Data preprocessing and preparation
├── model_architecture.py  # Neural network architecture and training
├── model_evaluation.py    # Model evaluation and visualization
├── main.py               # Main script to run the pipeline
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Features

1. **Data Preparation**
   - Merging pipe and soil data
   - Handling class imbalance using class weights (instead of SMOTE/SMOTEENN)
   - Feature preprocessing:
     - Log transformation (np.log1p) for:
       * AGE
       * INND (pipe diameter)
       * LLANG (pipe length)
       * soil_change_dist
     - Standard scaling for most numerical features
     - MinMax scaling for log-transformed features
     - One-hot encoding for categorical variables with 'drop first' strategy
   - Monte Carlo correlation analysis for feature relationships
   - Train/test split with k-fold cross-validation (5 folds)

2. **Model Architecture**
   - Two-layer neural network (64, 32 neurons)
   - Batch normalization after each layer
   - Dropout layers for regularization (0.3 rate)
   - L2 regularization on dense layers
   - Configurable activation functions (ReLU or LeakyReLU)
   - Sigmoid activation for binary classification output

3. **Advanced Features**
   - Bayesian optimization for hyperparameter tuning
   - K-fold cross-validation (5 folds)
   - Early stopping to prevent overfitting
   - Class weight balancing for imbalanced data
   - Comparative analysis of activation functions

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - ROC curve and AUC
   - Confusion matrix
   - Performance comparison across different activation functions
   - Training and validation curves by activation function

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place your pipe data CSV file in the project directory
   - Place your soil data CSV file in the project directory
   - Update the file paths in `main.py`

2. Run the analysis:
   ```bash
   python main.py
   ```

3. Check the results in the `results` directory:
   - `training_history.png`: Training and validation metrics over time
   - `confusion_matrix.png`: Confusion matrix visualization
   - `roc_curve.png`: ROC curve with AUC score
   - `shap_values.png`: Feature importance using SHAP values
   - `feature_importance.png`: Feature importance comparison
   - `baseline_comparison.csv`: Comparison with baseline models

## Model Architecture Details

The neural network architecture has been enhanced with the following features:

1. **Input Layer**: Matches the dimensionality of the preprocessed features

2. **Hidden Layers**:
   - Two-layer architecture (64, 32 neurons)
   - Batch normalization after each layer
   - Configurable activation functions (ReLU or LeakyReLU)
   - Dropout for regularization (rate: 0.3)
   - L2 regularization on dense layers

3. **Output Layer**:
   - Single neuron with sigmoid activation for binary classification

4. **Hyperparameter Optimization**:
   - Learning rate: Optimized through Bayesian optimization
   - Batch size: Optimized through Bayesian optimization
   - Dropout rates: 0.2-0.5
   - Activation functions: ReLU or LeakyReLU

## Advanced Features

1. **Model Training**:
   - K-fold cross-validation
   - Comparative analysis of activation functions
   - Learning rate optimization
   - Early stopping to prevent overfitting

2. **Performance Analysis**:
   - Separate performance metrics for different activation functions
   - Training and validation curves by activation function
   - Comprehensive metrics including accuracy, precision, recall, F1, and AUC
   - Visualization of learning curves for different model configurations

3. **Model Evaluation**:
   - Detailed performance metrics per fold
   - Activation function comparison
   - Feature importance analysis
   - Model interpretability through visualization


## Data Requirements

Your input data should contain the following features:

1. **Pipe Data**:
   - AGE: Age of the pipe
   - INND: Pipe diameter
   - LLANG: Pipe length
   - MATERIAL: Pipe material (categorical)
   - FTYP: Pipe type (categorical)
   - RENOV_AR: Renovation status (binary)
   - RENOV_METOD: Renovation method (categorical)

2. **Soil Data**:
   - Soil type 
   - soil_change
   - soil_change_dist
   - Must share a common unique ID with pipe data for merging

## Contributing

Feel free to submit issues and enhancement requests! 