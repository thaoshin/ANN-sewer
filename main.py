import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_preparation import DataPreparation
from model_architecture import SewerANN
from model_evaluation import ModelEvaluator
from sklearn.utils import class_weight #tried SMOTE/SMOTEEN before but class_weight seems to perform better
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

def main():
    """Main function to run the sewer pipe failure prediction pipeline."""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Set paths to the Excel files
    pipe_data_path = "data/felobs_tv_avlopp_250313_NY_1.xlsx"
    soil_data_path = "data/jordart_20250414.xlsx"
    
    print("Loading and merging data...")
    merged_data = data_prep.load_and_merge_data(pipe_data_path, soil_data_path)
    
    # Preprocess data
    print("Preprocessing data...")
    X_processed, y = data_prep.preprocess_data(merged_data)
    
    # Split into train and test sets (keeping test set aside)
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = data_prep.split_data(X_processed, y)
    
    # Print feature names for debugging
    print("\nFeature names after processing:")
    print(X_processed.columns.tolist())
    
    # Prepare for k-fold cross-validation
    n_splits = 5 # should I increase or decrease the fold nnumbers? 
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    # Get k-fold splits
    fold_splits = data_prep.get_k_fold_splits(X_train, y_train, n_splits=n_splits)
    
    # Lists to store metrics for each fold
    fold_histories = []
    fold_metrics = []
    
    # Define activation functions to try
    activations = ['relu', 'leakyrelu'] # i tried other activations before and leakyrelu seems to perform better
    
    # Train and evaluate on each fold
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, 1):
        print(f"\nTraining fold {fold_idx}/{n_splits}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Calculate class weights for this fold
        classes = np.unique(y_fold_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_fold_train
        )
        class_weights_dict = dict(zip(classes, weights))
        
        # Create and train model for this fold
        model_builder = SewerANN(input_dim=X_train.shape[1])
        
        # Alternate between ReLU and LeakyReLU for each fold
        current_activation = activations[fold_idx % len(activations)]
        print(f"Using activation function: {current_activation}")
        
        model = model_builder.create_model(activation=current_activation)
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=25, # should i increase the patience?
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_fold_train, y_fold_train,
            epochs=300,
            batch_size=64,
            validation_data=(X_fold_val, y_fold_val),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights_dict,
            verbose=1
        )
        
        # Create evaluator for this fold
        fold_evaluator = ModelEvaluator(model, X_fold_train, X_fold_val, y_fold_train, y_fold_val, X_processed.columns)
        
        # Plot training curves for this fold
        fold_evaluator.plot_training_history(history)
        
        # Generate predictions and plot evaluation metrics
        y_pred = model.predict(X_fold_val)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Plot evaluation metrics
        fold_evaluator.plot_confusion_matrix(y_fold_val, y_pred)
        fold_evaluator.plot_roc_curve(y_fold_val, y_pred)
        fold_evaluator.plot_precision_recall_curve(y_fold_val, y_pred)
        fold_evaluator.plot_ann_feature_importance()
        
        # Store training history
        fold_histories.append(history.history)
        
        # Calculate metrics
        metrics = {
            'fold': fold_idx,
            'activation': current_activation,
            'accuracy': accuracy_score(y_fold_val, y_pred_classes),
            'precision': precision_score(y_fold_val, y_pred_classes),
            'recall': recall_score(y_fold_val, y_pred_classes),
            'f1': f1_score(y_fold_val, y_pred_classes),
            'auc': roc_auc_score(y_fold_val, y_pred)
        }
        fold_metrics.append(metrics)
        
        print(f"\nFold {fold_idx} metrics:")
        for metric, value in metrics.items():
            if metric not in ['fold', 'activation']:
                print(f"{metric}: {value:.4f}")
    
    
    # Filter metrics for only relu and leakyrelu before creating plots
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df = metrics_df[metrics_df['activation'].isin(['relu', 'leakyrelu'])]
    
    # Plot average learning curves for each activation function
    plt.figure(figsize=(15, 10))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    for activation in ['relu', 'leakyrelu']:
        activation_indices = [i for i, metrics in enumerate(fold_metrics) 
                            if metrics['activation'] == activation]
        if activation_indices:
            histories = [fold_histories[i] for i in activation_indices]
            # Find minimum length across all histories
            min_length = min(len(h['loss']) for h in histories)
            
            # Truncate all histories to minimum length
            avg_loss = np.mean([h['loss'][:min_length] for h in histories], axis=0)
            avg_val_loss = np.mean([h['val_loss'][:min_length] for h in histories], axis=0)
            epochs = range(1, min_length + 1)
            plt.plot(epochs, avg_loss, label=f'{activation} (train)')
            plt.plot(epochs, avg_val_loss, label=f'{activation} (val)', linestyle='--')
    
    plt.title('Training and Validation Loss by Activation Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    for activation in ['relu', 'leakyrelu']:
        activation_indices = [i for i, metrics in enumerate(fold_metrics) 
                            if metrics['activation'] == activation]
        if activation_indices:
            histories = [fold_histories[i] for i in activation_indices]
            # Find minimum length across all histories
            min_length = min(len(h['accuracy']) for h in histories)
            
            # Truncate all histories to minimum length
            avg_acc = np.mean([h['accuracy'][:min_length] for h in histories], axis=0)
            avg_val_acc = np.mean([h['val_accuracy'][:min_length] for h in histories], axis=0)
            epochs = range(1, min_length + 1)
            plt.plot(epochs, avg_acc, label=f'{activation} (train)')
            plt.plot(epochs, avg_val_acc, label=f'{activation} (val)', linestyle='--')
    
    plt.title('Training and Validation Accuracy by Activation Function')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()  # Show the final comparison plot
    plt.close()
    
    print("\nMetrics by activation function:")
    print(metrics_df.groupby('activation')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean())
    
    # Save filtered metrics to CSV
    metrics_df.to_csv('results/cross_validation_metrics.csv', index=False)
    
if __name__ == "__main__":
    main() 