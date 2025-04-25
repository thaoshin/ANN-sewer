import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # This import is required
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
import os
import random
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

class DataPreparation:   
    def __init__(self):
        """Initialize the DataPreparation class with feature lists."""
        self.scaler: Optional[StandardScaler] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numerical_features: List[str] = ['AGE', 'INND', 'LLANG', 'soil_change_dist']
        self.categorical_features: List[str] = ['MATERIAL', 'FTYP', 'RENOV_METOD', 'soil', 'soil_change']
        self.binary_features: List[str] = ['RENOV_AR']
        self.condition_features: List[str] = ['INL', 'RBR', 'SPR', 'YTS', 'DEF']
        self.imbalance_handler: Optional[Union[SMOTE, SMOTEENN, ADASYN]] = None
        self.imputer: Optional[IterativeImputer] = None
        
    def monte_carlo_correlation(self, X: pd.DataFrame, y: pd.Series, n_iterations: int = 1000) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Perform Monte Carlo correlation analysis with variance checks."""
        correlations = []
        feature_names = X.columns
        n_features = len(feature_names)

        for iter_num in range(n_iterations):
            sample_idx = random.sample(range(len(X)), int(0.8 * len(X)))
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Initialize correlations for this iteration
            iter_corr = pd.Series(np.nan, index=feature_names)

            # Check variance of y_sample first
            if y_sample.nunique() <= 1:
                # If target has no variance, all correlations are 0
                iter_corr.fillna(0.0, inplace=True)
            else:
                # Calculate correlation only for columns with variance in X_sample
                for col in feature_names:
                    if X_sample[col].nunique() > 1:
                        # Calculate correlation for this specific column
                        # Use try-except for safety during calculation
                        try:
                             correlation_value = np.corrcoef(X_sample[col], y_sample)[0, 1]
                             iter_corr[col] = correlation_value if not np.isnan(correlation_value) else 0.0
                        except Exception:
                             iter_corr[col] = 0.0 # Set to 0 if calculation fails
                    else:
                        # If X_sample[col] has no variance, correlation is 0
                        iter_corr[col] = 0.0
                        
                # Fill any remaining NaNs (shouldn't happen with the loop logic, but for safety)
                iter_corr.fillna(0.0, inplace=True)

            correlations.append(iter_corr)
        
        # Aggregate results
        if not correlations:
             print("Error: No correlation results were successfully calculated.")
             # Return Series with correct index and 0 values
             zero_series = pd.Series(0.0, index=feature_names)
             return zero_series, zero_series.copy(), zero_series.copy()
             
        corr_df = pd.DataFrame(correlations)
        mean_corr = corr_df.mean()
        ci_lower = corr_df.quantile(0.025)
        ci_upper = corr_df.quantile(0.975)
        
        # Final check for NaNs in aggregated results
        mean_corr.fillna(0.0, inplace=True)
        ci_lower.fillna(0.0, inplace=True)
        ci_upper.fillna(0.0, inplace=True)

        return mean_corr, ci_lower, ci_upper
             
        
    def load_and_merge_data(self, pipe_data_path: str, soil_data_path: str) -> pd.DataFrame:
        """Load data and merge pipe and soil data."""
        
        # --- Load Data ---
        if not os.path.exists(pipe_data_path):
            raise FileNotFoundError(f"Pipe data file not found: {pipe_data_path}")
        if not os.path.exists(soil_data_path):
            raise FileNotFoundError(f"Soil data file not found: {soil_data_path}")
            
        try:
            pipe_data = pd.read_excel(pipe_data_path)
            soil_data = pd.read_excel(soil_data_path)
        except Exception as e:
            raise ValueError(f"Error loading data files: {str(e)}")
        
        print("\nInitial number of pipes:", len(pipe_data))
        
        # --- Pipe Data Cleaning ---
        pipe_data['AGE'] = 2025 - pipe_data['ANLAR']
        
        pipe_data = pipe_data[pipe_data['INL'].notna()].copy()
        print("\nNumber of inspected pipes:", len(pipe_data))
        pipe_data['RENOV_AR'] = (pipe_data['RENOV_AR'] > 0).astype(int)

        # --- Soil Data Feature Engineering ---
        print("Processing soil data...")
        soil_eng_cols = ['id', 'soil', 'soil_change', 'soil_change_dist']
        temp_soil_df = pd.DataFrame({'id': soil_data['id']})
        
        # Process soil type (keep as categorical but use numeric codes)
        temp_soil_df['soil'] = soil_data['soil'].astype(str)
        
        # Process soil_change (keep as categorical with numeric codes)
        if 'soil_change' in soil_data.columns:
            temp_soil_df['soil_change'] = soil_data['soil_change'].fillna('missing').astype(str)
        else:
            temp_soil_df['soil_change'] = 'missing'
            
        # Process soil_change_dist with log transform
        if 'soil_change_dist' in soil_data.columns:
            # Add small constant before log transform to handle zeros
            temp_soil_df['soil_change_dist'] = np.log1p(pd.to_numeric(soil_data['soil_change_dist'], errors='coerce').fillna(0))
        else:
            temp_soil_df['soil_change_dist'] = np.nan

        # Select columns for merging
        soil_data_to_merge = temp_soil_df[soil_eng_cols]

        # --- Merge Data ---
        merged_data = pd.merge(pipe_data, soil_data_to_merge, on='id', how='left')
        print(f"\nNumber of pipes after merging: {len(merged_data)}")
        
        # --- Post-Merge Processing ---
        print("\nMissing values after merge:")
        print(merged_data.isnull().sum())
        # Imputation will handle these NaNs later in preprocess_data

        # Convert target INL to binary
        merged_data['INL'] = (merged_data['INL'] > 0).astype(int)
        print("\nINL binary distribution:")
        print(merged_data['INL'].value_counts())
        
        # Plot distribution for all numerical features after merge
        print("\nGenerating numerical feature distribution plots...")
        for feature in self.numerical_features:
            self.plot_numerical_distribution(merged_data, feature_name=feature)
        print("Numerical feature distribution plots generated.")
            
        return merged_data
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data with improved missing value handling."""
        # Extract target variable
        y = data['INL'].astype(int)  # Ensure target is integer
        
        # Select features
        feature_cols = (
            self.numerical_features +
            self.categorical_features +
            self.binary_features +
            [col for col in self.condition_features if col != 'INL']
        )
        X = data[feature_cols].copy()
        
        # Apply log transform to skewed numerical features before scaling
        if 'AGE' in X.columns:
            X['AGE'] = np.log1p(X['AGE'])
        if 'INND' in X.columns:
            X['INND'] = np.log1p(X['INND'])
        if 'LLANG' in X.columns:
            X['LLANG'] = np.log1p(X['LLANG'])
        
        # Convert categorical columns to string type
        for col in self.categorical_features:
            X[col] = X[col].astype(str)
        
        # Convert binary features to int
        for col in self.binary_features:
            X[col] = X[col].astype(int)
        
        # Define features for different scaling strategies
        standard_scale_features = [f for f in self.numerical_features if f not in ['AGE', 'INND', 'LLANG', 'soil_change_dist']]
        minmax_scale_features = ['AGE', 'INND', 'LLANG', 'soil_change_dist']
        
        # Create preprocessing pipelines for numerical features
        standard_numerical_transformer = Pipeline([
            ('imputer', IterativeImputer(random_state=42, max_iter=10)),  # MICE imputation
            ('scaler', StandardScaler())
        ])
        
        # MinMax scaler for log-transformed features
        minmax_numerical_transformer = Pipeline([
            ('imputer', IterativeImputer(random_state=42, max_iter=10)),  # MICE imputation
            ('scaler', MinMaxScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('standard_num', standard_numerical_transformer, standard_scale_features),
                ('minmax_num', minmax_numerical_transformer, minmax_scale_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Add numerical feature names in the correct order
        feature_names.extend(standard_scale_features)
        feature_names.extend(minmax_scale_features)
        
        # Add one-hot encoded feature names
        if self.categorical_features:
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_features)
        
        # Add remaining features (binary and condition)
        remainder_features = [col for col in X.columns if col not in self.numerical_features and col not in self.categorical_features]
        feature_names.extend(remainder_features)
        
        # Create DataFrame with proper column names
        X_processed = pd.DataFrame(
            X_processed,
            columns=feature_names
        )
        
        return X_processed, y

    def get_k_fold_splits(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> List[Tuple]:
        """Generate k-fold cross-validation splits.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of folds
            
        Returns:
            List of (train_idx, val_idx) tuples for each fold
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(kf.split(X, y))

    def split_data(self, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  test_size: float = 0.2,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Note: Validation split will be handled by k-fold cross-validation
        """
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test 

    def plot_numerical_distribution(self, data: pd.DataFrame, feature_name: str, output_dir: str = 'results') -> None:
        """Plot the distribution of a specific numerical feature.
        
        Args:
            data: DataFrame containing the feature column
            feature_name: Name of the numerical feature column to plot
            output_dir: Directory to save the plots
        """
        if feature_name not in data.columns:
            print(f"\nWarning: Feature '{feature_name}' not found in data for plotting distribution.")
            return
            
        if data[feature_name].isnull().all():
            print(f"\nWarning: Feature '{feature_name}' is all null. Skipping distribution plot.")
            return
            
        plt.figure(figsize=(12, 6))
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_") # Make filename safe
        
        # Create main distribution plot
        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=feature_name, bins=30)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel(f'{feature_name}')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Create boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=data[feature_name])
        plt.title(f'{feature_name} Distribution Boxplot')
        plt.ylabel(f'{feature_name}')
        
        # Print statistics
        print(f"\n{feature_name} Statistics:")
        print(data[feature_name].describe())
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe_feature_name}_distribution.png'))
        plt.close()
        
        # Additional analysis: Check for potential outliers
        try:
            Q1 = data[feature_name].quantile(0.25)
            Q3 = data[feature_name].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_threshold_lower = Q1 - 1.5 * IQR
                outlier_threshold_upper = Q3 + 1.5 * IQR
                outliers = data[(data[feature_name] < outlier_threshold_lower) | (data[feature_name] > outlier_threshold_upper)]
                print(f"\nNumber of potential {feature_name} outliers (IQR method): {len(outliers)}")
                if len(outliers) > 0 and len(outliers) < 50: # Print details only for a few outliers
                    print(f"Outlier values for {feature_name}:")
                    print(outliers[feature_name].unique())
            else:
                 print(f"\nIQR for {feature_name} is zero, cannot calculate outliers.")
        except Exception as e:
            print(f"\nError calculating outliers for {feature_name}: {e}") 