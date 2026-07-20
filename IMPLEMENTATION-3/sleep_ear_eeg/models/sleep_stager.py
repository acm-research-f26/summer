# models/sleep_stager.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

class SleepStager:
    """
    Enhanced Random Forest classifier with hyperparameter tuning.
    """
    def __init__(self, tune_hyperparams=False):
        self.scaler = StandardScaler()
        self.tune_hyperparams = tune_hyperparams
        
        # Base model with better parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        if tune_hyperparams:
            # Hyperparameter grid for tuning
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

    def train(self, X_train, y_train):
        """
        Trains the sleep staging model with scaling and optional hyperparameter tuning.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.tune_hyperparams:
            print("  Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.model, 
                self.param_grid, 
                cv=3, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"  Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X_scaled, y_train)

    def predict(self, X_test):
        """
        Predicts sleep stages for new data.
        """
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)