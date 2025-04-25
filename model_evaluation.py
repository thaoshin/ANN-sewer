import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
from tensorflow.keras.layers import Dense
import pandas as pd
import re
import os
from sklearn.preprocessing import MinMaxScaler

class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
    def plot_training_history(self, history):
        """Plot training and validation Loss and Accuracy."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Loss
        ax1 = axes[0]
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Accuracy
        ax2 = axes[1]
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()  # Show the plot
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        f1 = f1_score(y_true, y_pred.round())
        cm = confusion_matrix(y_true, y_pred.round())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (F1 Score: {f1:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()  # Show the plot
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve and calculate AUC."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()  # Show the plot
        return plt.gcf(), roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.show()  # Show the plot
        return plt.gcf()
    
    def plot_ann_feature_importance(self, main_categories_map=None):
        """Plot feature importance based on ANN weights, aggregated by main categories."""
        
        # Print all available feature names for debugging
        print("\nAvailable features:")
        print(self.feature_names)
        
        # Define main category mapping
        main_categories_map = {
            # Physical/Numerical properties
            'AGE': 'Age',
            'INND': 'Diameter',
            'LLANG': 'Length',
            'soil_change_dist': 'Distance to Soil Change',
            
            # Material type 
            'MATERIAL': 'Material Type',
            
            # Pipe type
            'FTYP_D': 'Pipe Type',
            'FTYP_S': 'Pipe Type',
            'FTYP_K': 'Pipe Type',
            
            # Environmental factors - Soil types (all under one category)
            'soil_1': 'Soil Type',
            'soil_5': 'Soil Type',
            'soil_9': 'Soil Type',
            'soil_10': 'Soil Type',
            'soil_16': 'Soil Type',
            'soil_17': 'Soil Type',
            'soil_24': 'Soil Type',
            'soil_28': 'Soil Type',
            'soil_31': 'Soil Type',
            'soil_33': 'Soil Type',
            'soil_40': 'Soil Type',
            'soil_50': 'Soil Type',
            'soil_55': 'Soil Type',
            'soil_91': 'Soil Type',
            'soil_95': 'Soil Type',
            'soil_200': 'Soil Type',
            'soil_890': 'Soil Type',
            'soil_9147': 'Soil Type',
            'soil_nan': 'Soil Type',
            
            # Maintenance
            'RENOV_AR': 'Renovation Year',

        }
        
        # Add soil change combinations dynamically based on feature names
        soil_change_features = [f for f in self.feature_names if f.startswith('soil_change_')]
        for feature in soil_change_features:
            if feature != 'soil_change_dist':  # Skip the distance feature
                main_categories_map[feature] = 'Soil Change Type'

        # Add any MATERIAL_ prefixed features dynamically
        material_features = [f for f in self.feature_names if f.startswith('MATERIAL_')]
        for feature in material_features:
            main_categories_map[feature] = 'Material Type'

        # Add any RENOV_METOD_ prefixed features dynamically
        renov_features = [f for f in self.feature_names if f.startswith('RENOV_METOD_')]
        for feature in renov_features:
            main_categories_map[feature] = 'Renovation Method'

        # Find first Dense layer
        first_dense = None
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                # Check input shape dynamically
                if hasattr(layer.input, 'shape') and len(layer.input.shape) > 1 and layer.input.shape[1] == len(self.feature_names):
                    first_dense = layer
                    break
        
        if first_dense is None:
             # Fallback: use the first Dense layer encountered if specific match fails
             for layer in self.model.layers:
                 if isinstance(layer, Dense):
                     first_dense = layer
                     break
             if first_dense is None:
                 raise ValueError("Could not find any Dense layer in the model.")


        # Get the weights
        weights = first_dense.get_weights()[0]
        
        # Check if weight matrix shape matches feature count
        if weights.shape[0] != len(self.feature_names):
             raise ValueError(f"Weight matrix input dimension ({weights.shape[0]}) does not match number of features ({len(self.feature_names)}). Please check model architecture and feature names.")
             
        # Calculate feature importance
        feature_importance = []
        for idx, feature_name in enumerate(self.feature_names):
            # Ensure index is within bounds
            if idx < weights.shape[0]:
                 importance = np.mean(np.abs(weights[idx]))
                 feature_importance.append(importance)
            else:
                 print(f"Warning: Feature index {idx} ({feature_name}) is out of bounds for weight matrix with shape {weights.shape}. Skipping.")
                 feature_importance.append(0) # Assign 0 importance if index is out of bounds

        # Aggregate importance by main category
        category_importance = {}
        unmatched_features = []
        
        # Process each feature
        for idx, feature_name in enumerate(self.feature_names):
            importance_value = feature_importance[idx]
            matched = False
            
            # Try exact matches
            if feature_name in main_categories_map:
                category = main_categories_map[feature_name]
                category_importance[category] = category_importance.get(category, 0) + importance_value
                matched = True
            
            if not matched:
                unmatched_features.append((feature_name, importance_value))
        
        # Print any unmatched features for debugging
        if unmatched_features:
            print("\nWarning: Some features were not matched to any category:")
            print("Feature Name | Importance Value")
            print("-" * 40)
            for name, value in unmatched_features:
                print(f"{name} | {value:.4f}")
        
        # Sort categories by importance
        sorted_importance = dict(sorted(category_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Print matched categories and their total importance
        print("\nMatched categories and their importance:")
        for category, importance in sorted_importance.items():
            print(f"{category}: {importance:.4f}")
        
        # Create horizontal bar plot
        plt.figure(figsize=(12, 8))
        categories = list(sorted_importance.keys())
        values = list(sorted_importance.values())
        
        # Create bars with a different color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        bars = plt.barh(categories, values, color=colors)
        
        # Customize plot
        plt.title('Feature Importance', fontsize=14, pad=20)
        plt.xlabel('Average Absolute Weight', fontsize=12)
        plt.ylabel('Feature Category', fontsize=12)
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', 
                    va='center', ha='left', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()

    def evaluate_model(self, history, save_dir='results'):
        """Comprehensive model evaluation with all plots."""
        print("\nGenerating evaluation plots...")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        precision = precision_score(self.y_test, y_pred_classes)
        recall = recall_score(self.y_test, y_pred_classes)
        f1 = f1_score(self.y_test, y_pred_classes)
        roc_auc_val = roc_auc_score(self.y_test, y_pred)
        
        print("\nTest Set Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc_val:.4f}")
        