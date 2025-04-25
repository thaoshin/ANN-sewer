import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import optuna
import numpy as np

class SewerANN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        
    def create_model(self, learning_rate=0.0005, batch_size=64, activation='relu'):
        """Create a neural network model with regularization and dropout.
        
        Args:
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            activation: Activation function ('relu', 'leakyrelu',)
        """
        # Define L2 regularization
        l2_reg = regularizers.l2(0.01)
        
        # Map activation function names to actual functions
        activation_map = {
            'relu': layers.ReLU(),
            'leakyrelu': layers.LeakyReLU(alpha=0.01)
        }
        
        activation_layer = activation_map.get(activation, layers.ReLU())
        
        # Create model
        inputs = Input(shape=(self.input_dim,))
        
        # First layer
        x = Dense(64, kernel_regularizer=l2_reg)(inputs)
        x = BatchNormalization()(x)
        x = activation_layer(x)
        x = Dropout(0.3)(x)
        
        # Second layer
        x = Dense(32, kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = activation_layer(x)
        x = Dropout(0.3)(x)
        
        # Output layer (always sigmoid for binary classification)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with fixed learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.F1Score(threshold=0.5, name='f1_score')
                    ]
        )
        
        self.model = model
        return model
    
    def bayesian_optimization(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Perform Bayesian optimization for hyperparameter tuning."""
        def objective(trial):
            # Define hyperparameter search space
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            l2_factor = trial.suggest_float('l2_factor', 1e-4, 1e-1, log=True)
            activation = trial.suggest_categorical('activation', ['relu', 'leakyrelu'])

            # Create model with trial hyperparameters
            l2_reg = regularizers.l2(l2_factor)
            
            # Map activation function names to actual functions
            activation_map = {
                'relu': layers.ReLU(),
                'leakyrelu': layers.LeakyReLU(alpha=0.01)
            }
            activation_layer = activation_map.get(activation, layers.ReLU())
            
            inputs = Input(shape=(self.input_dim,))
            x = Dense(64, kernel_regularizer=l2_reg)(inputs)
            x = BatchNormalization()(x)
            x = activation_layer(x)
            x = Dropout(dropout_rate)(x)
            x = Dense(32, kernel_regularizer=l2_reg)(x)
            x = BatchNormalization()(x)
            x = activation_layer(x)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            
            # Use fixed learning rate
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            val_loss = min(history.history.get('val_loss', [np.inf]))
            return val_loss
        
        # Create study object and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best trial parameters: {study.best_trial.params}")
        print(f"Best validation loss: {study.best_value}")

        # Create final model with best parameters found
        best_lr = study.best_trial.params['learning_rate']
        best_bs = study.best_trial.params['batch_size']
        best_activation = study.best_trial.params.get('activation', 'relu')
        
        self.create_model(learning_rate=best_lr, batch_size=best_bs, activation=best_activation)
        
        return study.best_trial.params 