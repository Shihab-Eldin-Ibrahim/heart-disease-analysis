"""
Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset
=======================================================================

This project analyzes, predicts, and visualizes heart disease risks using machine learning.
Includes data preprocessing, feature selection, dimensionality reduction, model training,
evaluation, and deployment with Streamlit UI.

Author: AI Assistant
Date: 2024
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, silhouette_score)
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: Streamlit and Ngrok
try:
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        RUNNING_IN_STREAMLIT = get_script_run_ctx() is not None
    except Exception:
        RUNNING_IN_STREAMLIT = False
except ImportError:
    st = None
    RUNNING_IN_STREAMLIT = False

try:
    from pyngrok import ngrok
except ImportError:
    ngrok = None

warnings.filterwarnings('ignore')

class HeartDiseaseAnalyzer:
    """Comprehensive Heart Disease Analysis Pipeline"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        """Load Heart Disease UCI Dataset"""
        # Create synthetic heart disease dataset based on UCI characteristics
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data with realistic heart disease features
        data = {
            'age': np.random.normal(54, 9, n_samples),
            'sex': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'trestbps': np.random.normal(131, 18, n_samples),
            'chol': np.random.normal(246, 52, n_samples),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1]),
            'thalach': np.random.normal(149, 23, n_samples),
            'exang': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'oldpeak': np.random.exponential(1.0, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'target': np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
        }
        
        self.data = pd.DataFrame(data)
        
        # Add some realistic correlations
        self.data.loc[self.data['target'] == 1, 'age'] += np.random.normal(5, 3, 
                                                                          sum(self.data['target'] == 1))
        self.data.loc[self.data['target'] == 1, 'chol'] += np.random.normal(20, 10, 
                                                                            sum(self.data['target'] == 1))
        self.data.loc[self.data['target'] == 1, 'oldpeak'] += np.random.normal(0.5, 0.3, 
                                                                              sum(self.data['target'] == 1))
        
        # Ensure realistic ranges
        self.data['age'] = np.clip(self.data['age'], 29, 77)
        self.data['trestbps'] = np.clip(self.data['trestbps'], 94, 200)
        self.data['chol'] = np.clip(self.data['chol'], 126, 564)
        self.data['thalach'] = np.clip(self.data['thalach'], 71, 202)
        self.data['oldpeak'] = np.clip(self.data['oldpeak'], 0, 6.2)
        
        return self.data
    
    def preprocess_data(self):
        """Data Preprocessing & Cleaning"""
        if self.data is None:
            self.load_data()
        
        # Handle missing values (if any)
        self.data = self.data.dropna()
        
        # Separate features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def feature_selection(self):
        """Feature Selection using statistical methods and ML-based techniques"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        feature_names = self.data.columns[:-1]  # Exclude target
        
        # 1. Statistical Feature Selection (Chi-Square Test)
        chi2_scores, p_values = f_classif(self.X_train_scaled, self.y_train)
        chi2_features = pd.DataFrame({
            'feature': feature_names,
            'chi2_score': chi2_scores,
            'p_value': p_values
        }).sort_values('chi2_score', ascending=False)
        
        # 2. SelectKBest
        selector_kbest = SelectKBest(score_func=f_classif, k=10)
        X_kbest = selector_kbest.fit_transform(self.X_train_scaled, self.y_train)
        kbest_features = feature_names[selector_kbest.get_support()]
        
        # 3. Recursive Feature Elimination (RFE)
        rfe_selector = RFE(LogisticRegression(random_state=42), n_features_to_select=10)
        X_rfe = rfe_selector.fit_transform(self.X_train_scaled, self.y_train)
        rfe_features = feature_names[rfe_selector.get_support()]
        
        self.feature_selection_results = {
            'chi2': chi2_features,
            'kbest': kbest_features,
            'rfe': rfe_features
        }
        
        return self.feature_selection_results
    
    def dimensionality_reduction(self, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        # Calculate explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.pca_results = {
            'n_components': self.pca.n_components_,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'X_train_pca': self.X_train_pca,
            'X_test_pca': self.X_test_pca
        }
        
        return self.pca_results
    
    def train_supervised_models(self):
        """Train Supervised Learning Models"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models on original features
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'auc_score': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # Train models on PCA features
        if hasattr(self, 'X_train_pca'):
            models_pca = {
                'Logistic Regression PCA': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest PCA': RandomForestClassifier(random_state=42, n_estimators=100),
                'SVM PCA': SVC(random_state=42, probability=True)
            }
            
            for name, model in models_pca.items():
                model.fit(self.X_train_pca, self.y_train)
                y_pred = model.predict(self.X_test_pca)
                y_pred_proba = model.predict_proba(self.X_test_pca)[:, 1] if hasattr(model, 'predict_proba') else None
                
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'classification_report': classification_report(self.y_test, y_pred),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                    'auc_score': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
        
        return self.results
    
    def unsupervised_learning(self):
        """Apply Unsupervised Learning (K-Means, Hierarchical Clustering)"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = kmeans.fit_predict(self.X_train_scaled)
        kmeans_silhouette = silhouette_score(self.X_train_scaled, kmeans_labels)
        
        # Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=2)
        hierarchical_labels = hierarchical.fit_predict(self.X_train_scaled)
        hierarchical_silhouette = silhouette_score(self.X_train_scaled, hierarchical_labels)
        
        self.clustering_results = {
            'kmeans': {
                'labels': kmeans_labels,
                'silhouette_score': kmeans_silhouette,
                'model': kmeans
            },
            'hierarchical': {
                'labels': hierarchical_labels,
                'silhouette_score': hierarchical_silhouette,
                'model': hierarchical
            }
        }
        
        return self.clustering_results
    
    def hyperparameter_tuning(self):
        """Optimize Models using Hyperparameter Tuning"""
        if self.X_train_scaled is None:
            self.preprocess_data()
        
        # Define parameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }
        }
        
        self.tuning_results = {}
        
        # GridSearchCV for smaller parameter spaces
        for name, param_grid in param_grids.items():
            if name == 'Logistic Regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                grid_search.fit(self.X_train_scaled, self.y_train)
                
                self.tuning_results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'best_model': grid_search.best_estimator_
                }
        
        # RandomizedSearchCV for Random Forest (larger parameter space)
        rf_model = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            rf_model, param_grids['Random Forest'], 
            n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
        )
        random_search.fit(self.X_train_scaled, self.y_train)
        
        self.tuning_results['Random Forest'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_model': random_search.best_estimator_
        }
        
        return self.tuning_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.data is None:
            self.load_data()
        
        # Set up the plotting style
        plt.style.use('default')  # Use default style to avoid conflicts
        sns.set_palette("husl")
        
        # 1. Data Distribution with better spacing
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('Heart Disease Dataset Analysis', fontsize=20, fontweight='bold', y=0.96)
        
        # Age distribution
        axes[0, 0].hist(self.data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution', fontsize=16, fontweight='bold', pad=25)
        axes[0, 0].set_xlabel('Age (years)', fontsize=14)
        axes[0, 0].set_ylabel('Frequency', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(labelsize=12)
        
        # Cholesterol distribution
        axes[0, 1].hist(self.data['chol'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Cholesterol Distribution', fontsize=16, fontweight='bold', pad=25)
        axes[0, 1].set_xlabel('Cholesterol (mg/dl)', fontsize=14)
        axes[0, 1].set_ylabel('Frequency', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(labelsize=12)
        
        # Target distribution
        target_counts = self.data['target'].value_counts()
        colors = ['lightgreen', 'salmon']
        wedges, texts, autotexts = axes[0, 2].pie(target_counts.values, 
                                                  labels=['No Disease', 'Disease'], 
                                                  autopct='%1.1f%%', 
                                                  colors=colors,
                                                  startangle=90)
        axes[0, 2].set_title('Heart Disease Distribution', fontsize=16, fontweight='bold', pad=25)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # Correlation heatmap with better formatting
        corr_matrix = self.data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot heatmap with better formatting
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=25)
        
        # Rotate x-axis labels for better readability
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=11)
        axes[1, 0].tick_params(axis='y', rotation=0, labelsize=11)
        
        # Age vs Cholesterol colored by target
        scatter = axes[1, 1].scatter(self.data['age'], self.data['chol'], 
                                   c=self.data['target'], cmap='RdYlBu', alpha=0.7, s=50)
        axes[1, 1].set_title('Age vs Cholesterol\n(colored by Disease Status)', fontsize=16, fontweight='bold', pad=25)
        axes[1, 1].set_xlabel('Age (years)', fontsize=14)
        axes[1, 1].set_ylabel('Cholesterol (mg/dl)', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(labelsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Disease Status\n(0=No Disease, 1=Disease)', fontsize=12)
        
        # Chest pain type distribution
        cp_counts = self.data['cp'].value_counts()
        cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        bars = axes[1, 2].bar(range(len(cp_counts)), cp_counts.values, 
                             color='mediumpurple', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Chest Pain Type Distribution', fontsize=16, fontweight='bold', pad=25)
        axes[1, 2].set_xlabel('Chest Pain Type', fontsize=14)
        axes[1, 2].set_ylabel('Count', fontsize=14)
        axes[1, 2].set_xticks(range(len(cp_counts)))
        axes[1, 2].set_xticklabels([cp_labels[i] for i in cp_counts.index], rotation=45, ha='right', fontsize=11)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(labelsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, cp_counts.values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)  # Better spacing
        return fig
    
    def create_model_comparison_plot(self):
        """Create model comparison visualization"""
        if not self.results:
            self.train_supervised_models()
        
        # Extract accuracies
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        # Create comparison plot
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.95)
        
        # Accuracy comparison with better formatting
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        bars = ax1.bar(model_names, accuracies, color=colors[:len(model_names)], alpha=0.8, edgecolor='black')
        ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=25)
        ax1.set_ylabel('Accuracy Score', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45, labelsize=11)
        ax1.tick_params(axis='y', labelsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Create separate figure for confusion matrices
        cm_data = []
        cm_labels = []
        for name in model_names[:4]:  # Show first 4 models
            if name in self.results:  # Check if model exists
                cm = self.results[name]['confusion_matrix']
                cm_data.append(cm)
                cm_labels.append(name)
        
        if cm_data:  # Only create if we have data
            fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
            fig2.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=0.95)
            
            for i, (cm, label) in enumerate(zip(cm_data, cm_labels)):
                row = i // 2
                col = i % 2
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[row, col], 
                           cbar_kws={'shrink': 0.8}, square=True)
                axes2[row, col].set_title(f'{label}\nConfusion Matrix', fontsize=14, fontweight='bold', pad=20)
                axes2[row, col].set_xlabel('Predicted Label', fontsize=12)
                axes2[row, col].set_ylabel('True Label', fontsize=12)
                axes2[row, col].tick_params(labelsize=11)
                
                # Add labels for better understanding
                axes2[row, col].set_xticklabels(['No Disease', 'Disease'])
                axes2[row, col].set_yticklabels(['No Disease', 'Disease'])
            
            # Hide empty subplots
            for i in range(len(cm_data), 4):
                row = i // 2
                col = i % 2
                axes2[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
        else:
            fig2 = None
        
        return fig, fig2
    
    def create_pca_plot(self):
        """Create PCA visualization"""
        if not hasattr(self, 'pca_results'):
            self.dimensionality_reduction()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Principal Component Analysis (PCA)', fontsize=16, fontweight='bold', y=0.95)
        
        # Explained variance
        components = range(1, len(self.pca_results['explained_variance']) + 1)
        ax1.plot(components, self.pca_results['explained_variance'], 'bo-', linewidth=2, markersize=8)
        ax1.set_title('PCA Explained Variance Ratio', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Principal Component Number', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(components, self.pca_results['explained_variance'])):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        # Cumulative variance
        ax2.plot(components, self.pca_results['cumulative_variance'], 'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color='k', linestyle='--', alpha=0.7, linewidth=2, label='95% Variance Threshold')
        ax2.set_title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(components, self.pca_results['cumulative_variance'])):
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        return fig
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Heart Disease Analysis Pipeline...")
        
        # 1. Load and preprocess data
        print("1. Loading and preprocessing data...")
        self.load_data()
        self.preprocess_data()
        
        # 2. Feature selection
        print("2. Performing feature selection...")
        self.feature_selection()
        
        # 3. Dimensionality reduction
        print("3. Applying PCA...")
        self.dimensionality_reduction()
        
        # 4. Train supervised models
        print("4. Training supervised models...")
        self.train_supervised_models()
        
        # 5. Unsupervised learning
        print("5. Applying unsupervised learning...")
        self.unsupervised_learning()
        
        # 6. Hyperparameter tuning
        print("6. Optimizing hyperparameters...")
        self.hyperparameter_tuning()
        
        print("Pipeline completed successfully!")
        
        # Display results summary
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            if results['auc_score']:
                print(f"  AUC Score: {results['auc_score']:.4f}")
        
        if hasattr(self, 'clustering_results'):
            print(f"\nClustering Results:")
            print(f"  K-Means Silhouette Score: {self.clustering_results['kmeans']['silhouette_score']:.4f}")
            print(f"  Hierarchical Silhouette Score: {self.clustering_results['hierarchical']['silhouette_score']:.4f}")
        
        return self

def run_streamlit_ui():
    """Streamlit UI for Heart Disease Analysis"""
    st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
    
    st.title("🫀 Comprehensive Heart Disease Analysis")
    st.markdown("**Machine Learning Pipeline for Heart Disease Risk Prediction**")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = HeartDiseaseAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Pipeline Controls")
        
        # Pipeline steps
        st.subheader("Pipeline Steps")
        run_preprocessing = st.button("1. Data Preprocessing", key="preprocess")
        run_feature_selection = st.button("2. Feature Selection", key="features")
        run_pca = st.button("3. Dimensionality Reduction (PCA)", key="pca")
        run_supervised = st.button("4. Train Supervised Models", key="supervised")
        run_unsupervised = st.button("5. Unsupervised Learning", key="unsupervised")
        run_tuning = st.button("6. Hyperparameter Tuning", key="tuning")
        run_complete = st.button("🚀 Run Complete Pipeline", key="complete")
        
        st.markdown("---")
        
        # Ngrok deployment
        st.subheader("🌐 Deployment (Bonus)")
        if st.button("Start Ngrok Tunnel"):
            if ngrok is None:
                st.warning("Install pyngrok: `pip install pyngrok`")
            else:
                token = os.environ.get("NGROK_AUTH_TOKEN")
                if token:
                    ngrok.set_auth_token(token)
                try:
                    tunnel = ngrok.connect(8501)
                    st.success(f"🌍 Public URL: {tunnel.public_url}")
                    st.caption("Share this URL to access your app globally!")
                except Exception as e:
                    st.error(f"Ngrok error: {e}")
    
    # Main content area
    if run_complete:
        with st.spinner("Running complete pipeline..."):
            analyzer.run_complete_pipeline()
        st.success("✅ Pipeline completed successfully!")
    
    # Individual pipeline steps
    if run_preprocessing:
        with st.spinner("Preprocessing data..."):
            analyzer.load_data()
            analyzer.preprocess_data()
        st.success("✅ Data preprocessing completed!")
        st.dataframe(analyzer.data.head())
    
    if run_feature_selection:
        with st.spinner("Performing feature selection..."):
            results = analyzer.feature_selection()
        st.success("✅ Feature selection completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Chi-Square Test")
            st.dataframe(results['chi2'].head(10))
        with col2:
            st.subheader("SelectKBest Features")
            st.write(results['kbest'].tolist())
        with col3:
            st.subheader("RFE Features")
            st.write(results['rfe'].tolist())
    
    if run_pca:
        with st.spinner("Applying PCA..."):
            pca_results = analyzer.dimensionality_reduction()
        st.success("✅ PCA completed!")
        st.write(f"Number of components: {pca_results['n_components']}")
        st.write(f"Explained variance: {pca_results['explained_variance'][:5]}")
    
    if run_supervised:
        with st.spinner("Training supervised models..."):
            results = analyzer.train_supervised_models()
        st.success("✅ Supervised models trained!")
        
        # Display results
        for model_name, result in results.items():
            with st.expander(f"{model_name} Results"):
                st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                if result['auc_score']:
                    st.write(f"**AUC Score:** {result['auc_score']:.4f}")
                st.text("Classification Report:")
                st.text(result['classification_report'])
    
    if run_unsupervised:
        with st.spinner("Applying unsupervised learning..."):
            clustering_results = analyzer.unsupervised_learning()
        st.success("✅ Unsupervised learning completed!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**K-Means Silhouette Score:** {clustering_results['kmeans']['silhouette_score']:.4f}")
        with col2:
            st.write(f"**Hierarchical Silhouette Score:** {clustering_results['hierarchical']['silhouette_score']:.4f}")
    
    if run_tuning:
        with st.spinner("Optimizing hyperparameters..."):
            tuning_results = analyzer.hyperparameter_tuning()
        st.success("✅ Hyperparameter tuning completed!")
        
        for model_name, result in tuning_results.items():
            with st.expander(f"{model_name} Best Parameters"):
                st.write(f"**Best Score:** {result['best_score']:.4f}")
                st.write("**Best Parameters:**")
                st.json(result['best_params'])
    
    # Visualizations
    if hasattr(analyzer, 'data') and analyzer.data is not None:
        st.header("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Show Data Analysis"):
                fig = analyzer.create_visualizations()
                st.pyplot(fig)
        
        with col2:
            if st.button("Show Model Comparison"):
                if hasattr(analyzer, 'results') and analyzer.results:
                    fig1, fig2 = analyzer.create_model_comparison_plot()
                    st.pyplot(fig1)
                    if fig2 is not None:
                        st.pyplot(fig2)
                    else:
                        st.info("Confusion matrices will be shown after training models")
                else:
                    st.warning("Please train models first!")
        
        if st.button("Show PCA Analysis"):
            if hasattr(analyzer, 'pca_results'):
                fig = analyzer.create_pca_plot()
                st.pyplot(fig)
            else:
                st.warning("Please run PCA first!")
    
    # Footer
    st.markdown("---")
    st.markdown("**🔗 GitHub Repository:** [Heart Disease Analysis](https://github.com/yourusername/heart-disease-analysis)")
    st.markdown("**📚 Technologies Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Ngrok")

def run_cli():
    """Command Line Interface"""
    print("🫀 Heart Disease Analysis Pipeline")
    print("=" * 50)
    
    analyzer = HeartDiseaseAnalyzer()
    analyzer.run_complete_pipeline()
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig1 = analyzer.create_visualizations()
    plt.show()
    
    if analyzer.results:
        fig2 = analyzer.create_model_comparison_plot()
        plt.show()
    
    if hasattr(analyzer, 'pca_results'):
        fig3 = analyzer.create_pca_plot()
        plt.show()

if __name__ == "__main__":
    if RUNNING_IN_STREAMLIT and st is not None:
        run_streamlit_ui()
    else:
        run_cli()
