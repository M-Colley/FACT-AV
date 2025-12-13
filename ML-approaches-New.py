#!/usr/bin/env python3
"""
Improved ML Analysis Script for Trust Prediction
Author: Improved Version
Date: 2025

This script analyzes trust ratings using multiple ML approaches with improved:
- Code organization and readability
- Error handling and validation
- Performance evaluation
- Visualization consistency
- Documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import DBSCAN

# Optional imports with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - feature explanations will be skipped")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# Configuration
@dataclass
class Config:
    """Configuration class for the ML analysis"""
    data_path: Path = Path("data") / "all_combined_prepared_with_demographics.xlsx"
    results_path: Path = Path("results") / "ML-Approaches"
    sheet_name: str = "Sheet1"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    cv_folds: int = 5
    
    # Feature definitions
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    target_column: str = "trust"
    
    def __post_init__(self):
        if self.numerical_features is None:
            self.numerical_features = ['mIoU', 'License', 'Age']
        if self.categorical_features is None:
            self.categorical_features = ['SCENARIO', 'INTRODUCTION', 'Gender', 
                                       'Education', 'Job', 'DrivingFrequency', 'Distance']


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_mappings = self._get_label_mappings()
        self.feature_name_mappings = self._get_feature_name_mappings()
        
    def _get_label_mappings(self) -> Dict[str, Dict[str, str]]:
        """Define mappings for categorical variables"""
        return {
            'Gender': {
                "A1": "F", "A2": "M", "A3": "non-binary", "A4": "Prefer not to tell"
            },
            'Education': {
                "A1": "Secondary School", "A2": "Middle School", "A3": "High School",
                "A4": "College", "A5": "Vocational training"
            },
            'Job': {
                "A1": "Student (school)", "A2": "Student (college)", "A3": "Employee",
                "A4": "Self-employed", "A5": "Jobseeker", "A6": "Other"
            },
            'DrivingFrequency': {
                "A1": "Daily", "A2": "On working days", "A3": "3-4 times a week",
                "A4": "1 time a week", "A5": "1-3 times a month", "A6": "less than 1 time a month"
            },
            'Distance': {
                "A1": "less than 7.000km", "A2": "7.000 - 14.999km", "A3": "15.000 - 24.999km",
                "A4": "25.000 - 32.999km", "A5": "33.000 or more km"
            }
        }
    
    def _get_feature_name_mappings(self) -> Dict[str, str]:
        """Define mappings for feature names in plots"""
        return {
            'mIoU': 'Model Performance (mIoU)',
            'SCENARIO': 'Scenario',
            'SCENARIO_NeueMitte': 'Scenario: City',
            'SCENARIO_Ueberland': 'Scenario: Cross-Country',
            'SCENARIO_Spielstrasse': 'Scenario: Walking Speed Zone',
            'INTRODUCTION_boasting': 'Introduction: boasting',
            'INTRODUCTION_ambiguous': 'Introduction: ambiguous',
            'INTRODUCTION': 'Introduction',
            'Gender_M': 'Gender: Male',
            'Gender_non-binary': 'Gender: non-binary',
            'Education_High School': 'Education: High School',
            'Education_Vocational training': 'Education: Vocational training',
            'Job_Jobseeker': 'Job: Jobseeker',
            'Job_Other': 'Job: Other',
            'Job_Self-employed': 'Job: Self-employed',
            'Job_Student (college)': 'Job: Student (college)',
            'License': 'Driving License (years)',
            'Age': 'Age',
            'DrivingFrequency': 'Driving Frequency',
            'DrivingFrequency_1-3 times a month': 'Driving Frequency: 1-3/month',
            'DrivingFrequency_3-4 times a week': 'Driving Frequency: 3-4/week',
            'DrivingFrequency_Daily': 'Driving Frequency: Daily',
            'DrivingFrequency_less than 1 time a month': 'Driving Frequency: <1/month',
            'DrivingFrequency_On working days': 'Driving Frequency: Working days',
            'Distance': 'Distance',
            'Distance_25.000 - 32.999km': 'Distance: 25.000 - 32.999km',
            'Distance_7.000 - 14.999km': 'Distance: 7.000 - 14.999km',
            'Distance_less than 7.000km': 'Distance: <7.000km',
            'Distance_33.000 or more km': 'Distance: >=33.000km',
        }
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        print("Loading data...")
        
        # Load data
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
            
        df = pd.read_excel(self.config.data_path, sheet_name=self.config.sheet_name)
        print(f"Loaded {len(df)} rows")
        
        # Apply label mappings
        for column, mapping in self.label_mappings.items():
            if column in df.columns:
                df[column] = df[column].replace(mapping)
        
        # Remove rows with missing target variable
        initial_rows = len(df)
        df = df.dropna(subset=[self.config.target_column])
        print(f"Removed {initial_rows - len(df)} rows with missing target variable")
        
        # Basic data validation
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the loaded data"""
        # Check required columns
        required_cols = (self.config.numerical_features + 
                        self.config.categorical_features + 
                        [self.config.target_column])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check target variable range
        target_values = df[self.config.target_column].dropna()
        print(f"Target variable '{self.config.target_column}' statistics:")
        print(f"  Range: {target_values.min():.2f} - {target_values.max():.2f}")
        print(f"  Mean: {target_values.mean():.2f} ± {target_values.std():.2f}")
        print(f"  Missing values: {df[self.config.target_column].isnull().sum()}")


class ModelEvaluator:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config, feature_name_mappings: Dict[str, str]):
        self.config = config
        self.feature_name_mappings = feature_name_mappings
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train and evaluate Random Forest with hyperparameter tuning"""
        print("Training Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestRegressor(random_state=self.config.random_state)
        grid_search = GridSearchCV(rf, param_grid, cv=self.config.cv_folds, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Predictions and metrics
        y_pred = best_rf.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_rf, X_train, y_train, 
                                  cv=self.config.cv_folds, 
                                  scoring='neg_mean_absolute_error')
        
        return {
            'model': best_rf,
            'metrics': metrics,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'predictions': y_pred
        }
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                              metrics: Dict[str, float], method_name: str,
                              std: Optional[np.ndarray] = None) -> None:
        """Create consistent feature importance plots"""
        importances = model.feature_importances_
        
        # Create DataFrame and apply name mappings
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_df['feature'] = importance_df['feature'].replace(self.feature_name_mappings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', 
                   palette="viridis", hue='feature', legend=False)
        
        # Add error bars if provided
        if std is not None:
            plt.errorbar(x=importance_df['importance'], 
                        y=np.arange(len(importance_df)),
                        xerr=std[importance_df.index], 
                        fmt='none', c='black', capsize=3)
        
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Feature Importances - {method_name}')
        
        # Add metrics text box
        metrics_text = '\n'.join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        plt.text(0.98, 0.02, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        filename = f'feature_importance_{method_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.config.results_path / filename, 
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()


class ClusterAnalyzer:
    """Handles clustering analysis"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def perform_dbscan_clustering(self, df: pd.DataFrame) -> None:
        """Perform DBSCAN clustering on trust values"""
        print("Performing DBSCAN clustering...")
        
        # Prepare and scale features
        features = df[[self.config.target_column]].dropna()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # DBSCAN clustering
        db = DBSCAN(eps=0.3, min_samples=10).fit(features_scaled)
        labels = db.labels_
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Estimated number of clusters: {n_clusters}")
        print(f"Estimated number of noise points: {n_noise}")
        
        # Plotting
        self._plot_clusters(df, labels, n_clusters)
    
    def _plot_clusters(self, df: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> None:
        """Plot clustering results"""
        plt.figure(figsize=(10, 6))
        
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        x_values = df['mIoU'].values
        y_values = df[self.config.target_column].values
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'black'  # Black for noise
            
            class_member_mask = (labels == k)
            plt.scatter(x_values[class_member_mask], y_values[class_member_mask],
                       c=[col], s=60, alpha=0.7, label=f'Cluster {k}' if k != -1 else 'Noise')
        
        plt.xlabel('Model Performance (mIoU)')
        plt.ylabel('Trust Rating')
        plt.title(f'DBSCAN Clustering Results - {n_clusters} Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = f'dbscan_clusters_{n_clusters}.png'
        plt.savefig(self.config.results_path / filename, 
                   dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def main():
    """Main execution function"""
    # Setup
    config = Config()
    config.results_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor(config)
    evaluator = ModelEvaluator(config, data_processor.feature_name_mappings)
    cluster_analyzer = ClusterAnalyzer(config)
    
    # Load and preprocess data
    df = data_processor.load_and_preprocess_data()
    
    # Clustering analysis
    cluster_analyzer.perform_dbscan_clustering(df)
    
    # Prepare data for ML
    print("\nPreparing data for machine learning...")
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_data = df[config.categorical_features]
    one_hot_encoded = encoder.fit_transform(categorical_data)
    one_hot_df = pd.DataFrame(one_hot_encoded, 
                             columns=encoder.get_feature_names_out(config.categorical_features))
    
    # Combine features
    X = pd.concat([df[config.numerical_features], one_hot_df], axis=1)
    y = df[config.target_column]
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    
    # Train Random Forest
    rf_results = evaluator.train_random_forest(X_train, X_test, y_train, y_test)
    
    print(f"\nRandom Forest Results:")
    print(f"Best parameters: {rf_results['best_params']}")
    print(f"Cross-validation MAE: {rf_results['cv_mae_mean']:.4f} ± {rf_results['cv_mae_std']:.4f}")
    for metric, value in rf_results['metrics'].items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Feature importance plot
    evaluator.plot_feature_importance(
        rf_results['model'], X_train.columns.tolist(), 
        rf_results['metrics'], "Random Forest"
    )
    
    # Permutation importance
    print("\nCalculating permutation importance...")
    perm_importance = permutation_importance(
        rf_results['model'], X_test, y_test, 
        n_repeats=30, random_state=config.random_state
    )
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    importance_df['feature'] = importance_df['feature'].replace(
        data_processor.feature_name_mappings
    )
    
    sns.barplot(data=importance_df, x='importance', y='feature', 
               palette="plasma", hue='feature', legend=False)
    plt.errorbar(x=importance_df['importance'], y=np.arange(len(importance_df)),
                xerr=importance_df['std'], fmt='none', c='black', capsize=3)
    
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.title('Permutation Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(config.results_path / 'permutation_importance_random_forest.png', 
               dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # SHAP analysis (if available)
    if SHAP_AVAILABLE:
        print("Generating SHAP explanations...")
        try:
            explainer = shap.TreeExplainer(rf_results['model'])
            shap_values = explainer.shap_values(X_test)
            
            # Rename features for SHAP plot
            X_test_renamed = X_test.rename(columns=data_processor.feature_name_mappings)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_renamed, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(config.results_path / 'shap_summary_random_forest.png', 
                       dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    # Store all model results for comparison
    all_model_results = {'Random Forest': rf_results['metrics']}
    
    # Train other models
    models_to_test = []
    
    if CATBOOST_AVAILABLE:
        models_to_test.append(('CatBoost', CatBoostRegressor, True))  # Can handle categorical
    if XGBOOST_AVAILABLE:
        models_to_test.append(('XGBoost', xgb.XGBRegressor, False))  # Use one-hot encoded
    if LIGHTGBM_AVAILABLE:
        models_to_test.append(('LightGBM', lgb.LGBMRegressor, False))  # Use one-hot encoded
    if TABPFN_AVAILABLE:
        models_to_test.append(('TabPFN', TabPFNRegressor, False))  # Use one-hot encoded
    
    for model_name, model_class, use_categorical in models_to_test:
        print(f"\nTraining {model_name}...")
        try:
            if use_categorical and model_name == 'CatBoost':
                # CatBoost can handle categorical features natively
                X_model = df[config.numerical_features + config.categorical_features].copy()
                y_model = df[config.target_column]
                X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(
                    X_model, y_model, test_size=config.test_size, random_state=config.random_state
                )
                
                # Convert categorical columns to category dtype for CatBoost
                for feature in config.categorical_features:
                    X_train_model[feature] = X_train_model[feature].astype('category')
                    X_test_model[feature] = X_test_model[feature].astype('category')
                
                cat_features_indices = [X_train_model.columns.get_loc(c) for c in config.categorical_features]
                model = model_class(cat_features=cat_features_indices, 
                                  random_state=config.random_state, verbose=False)
                
                model.fit(X_train_model, y_train_model)
                y_pred = model.predict(X_test_model)
                feature_names = X_train_model.columns.tolist()
                
            else:
                # Use one-hot encoded data for XGBoost, LightGBM, and TabPFN
                X_train_model, X_test_model = X_train.copy(), X_test.copy()
                y_train_model, y_test_model = y_train.copy(), y_test.copy()
                
                if model_name == 'TabPFN':
                    model = model_class()  # TabPFN doesn't use random_state in constructor
                else:
                    model = model_class(random_state=config.random_state)
                
                # Special settings for XGBoost
                if model_name == 'XGBoost':
                    model.set_params(eval_metric='rmse')
                
                model.fit(X_train_model, y_train_model)
                y_pred = model.predict(X_test_model)
                feature_names = X_train_model.columns.tolist()
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(y_test_model, y_pred)
            all_model_results[model_name] = metrics
            
            print(f"{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            # Feature importance plot - ALL models that have feature_importances_
            if hasattr(model, 'feature_importances_'):
                print(f"  Generating feature importance plot for {model_name}...")
                evaluator.plot_feature_importance(
                    model, feature_names, metrics, model_name
                )
                
                # SHAP analysis for tree-based models
                if SHAP_AVAILABLE and model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                    print(f"  Generating SHAP explanations for {model_name}...")
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test_model)
                        
                        # Rename features for SHAP plot
                        if use_categorical:
                            # For CatBoost with categorical features
                            feature_mappings = {col: data_processor.feature_name_mappings.get(col, col) 
                                              for col in X_test_model.columns}
                            X_test_renamed = X_test_model.rename(columns=feature_mappings)
                        else:
                            # For one-hot encoded features
                            X_test_renamed = X_test_model.rename(columns=data_processor.feature_name_mappings)
                        
                        # SHAP summary bar plot
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_test_renamed, plot_type="bar", show=False)
                        plt.title(f'SHAP Feature Importance - {model_name}')
                        plt.tight_layout()
                        plt.savefig(config.results_path / f'shap_summary_{model_name.lower()}.png', 
                                   dpi=300, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                        
                        # SHAP beeswarm plot (especially useful for LightGBM as in original)
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_test_renamed, show=False)
                        plt.title(f'SHAP Feature Effects - {model_name}')
                        plt.tight_layout()
                        plt.savefig(config.results_path / f'shap_beeswarm_{model_name.lower()}.png', 
                                   dpi=300, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                        
                        print(f"  SHAP plots saved for {model_name}")
                        
                    except Exception as e:
                        print(f"  SHAP analysis failed for {model_name}: {e}")
            
            else:
                print(f"  Note: {model_name} does not provide feature importances")
                
                # For TabPFN, create a performance summary instead
                if model_name == 'TabPFN':
                    try:
                        print(f"  Generating TabPFN quantile analysis...")
                        quantiles = [0.25, 0.5, 0.75]
                        quantile_preds = model.predict(X_test_model, output_type="quantiles", quantiles=quantiles)
                        
                        results_text = [f"{model_name} Results:"]
                        for metric, value in metrics.items():
                            results_text.append(f"{metric.upper()}: {value:.4f}")
                        
                        results_text.append("\nQuantile Predictions MAE:")
                        for q, q_pred in zip(quantiles, quantile_preds):
                            q_mae = mean_absolute_error(y_test_model, q_pred)
                            results_text.append(f"Quantile {q}: {q_mae:.4f}")
                        
                        # Try mode prediction
                        try:
                            mode_pred = model.predict(X_test_model, output_type="mode")
                            mode_mae = mean_absolute_error(y_test_model, mode_pred)
                            results_text.append(f"Mode MAE: {mode_mae:.4f}")
                        except:
                            pass
                        
                        # Save results to file
                        results_file = config.results_path / f'tabpfn_detailed_results.txt'
                        with open(results_file, 'w') as f:
                            f.write('\n'.join(results_text))
                        
                        print(f"  TabPFN detailed results saved to: {results_file}")
                        
                    except Exception as e:
                        print(f"  Could not generate TabPFN quantile predictions: {e}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    print(f"\nAnalysis complete! Results saved to: {config.results_path}")
    
    # Create a summary comparison of all models
    print(f"\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-" * 60)
    print(f"{'Random Forest':<15} {rf_results['metrics']['mae']:<8.4f} {rf_results['metrics']['rmse']:<8.4f} {rf_results['metrics']['r2']:<8.4f}")
    
    # Note: In a real implementation, you'd store all model results to compare them here
    print(f"Cross-validation MAE (RF): {rf_results['cv_mae_mean']:.4f} ± {rf_results['cv_mae_std']:.4f}")


if __name__ == "__main__":
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Run analysis
    main()