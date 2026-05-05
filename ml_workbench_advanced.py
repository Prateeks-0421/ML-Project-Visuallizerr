import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, mean_squared_error,
    mean_absolute_error, r2_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import VotingClassifier
import joblib
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced ML Workbench",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
    }
    h2 {
        color: #764ba2;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Title and Description
st.title("🤖 Advanced ML Workbench Pro")
st.markdown("""
**Premium Machine Learning Platform** with advanced hyperparameter tuning, 
decision boundary visualization, and comprehensive model analysis.
""")

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/200x100?text=ML+Workbench", use_column_width=True)
    st.title("🎛️ Navigation")
    
    page = st.radio(
        "Select Module:",
        ["📊 Data Exploration", "⚙️ Model Configuration", 
         "🎯 Training & Tuning", "📈 Visualization", "🔍 Model Analysis", 
         "💾 Model Management"]
    )

# ==================== PAGE 1: DATA EXPLORATION ====================
if page == "📊 Data Exploration":
    st.header("📊 Data Exploration & Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Data Source")
        data_source = st.radio("Select data source:", ["Built-in Dataset", "Upload CSV"])
        
        if data_source == "Built-in Dataset":
            dataset_type = st.selectbox(
                "Choose dataset type:",
                ["Classification", "Regression"]
            )
            
            if dataset_type == "Classification":
                dataset = st.selectbox(
                    "Select classification dataset:",
                    ["Iris", "Wine", "Breast Cancer", "Synthetic Data"]
                )
                
                if dataset == "Iris":
                    data = load_iris()
                    X, y = data.data, data.target
                    feature_names = data.feature_names
                    task_type = "classification"
                elif dataset == "Wine":
                    data = load_wine()
                    X, y = data.data, data.target
                    feature_names = data.feature_names
                    task_type = "classification"
                elif dataset == "Breast Cancer":
                    data = load_breast_cancer()
                    X, y = data.data, data.target
                    feature_names = data.feature_names
                    task_type = "classification"
                else:
                    n_samples = st.slider("Number of samples:", 100, 10000, 500)
                    n_features = st.slider("Number of features:", 2, 50, 10)
                    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                              n_informative=max(2, n_features-2), random_state=42)
                    feature_names = [f"Feature {i}" for i in range(n_features)]
                    task_type = "classification"
            else:
                n_samples = st.slider("Number of samples:", 100, 10000, 500)
                n_features = st.slider("Number of features:", 1, 50, 5)
                X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                      n_informative=max(2, n_features-2), random_state=42)
                feature_names = [f"Feature {i}" for i in range(n_features)]
                task_type = "regression"
        
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                
                cols = df.columns.tolist()
                target_col = st.selectbox("Select target column:", cols)
                
                X = df.drop(columns=[target_col]).values
                y = df[target_col].values
                feature_names = df.drop(columns=[target_col]).columns.tolist()
                
                task_type = st.selectbox("Task type:", ["classification", "regression"])
        
        if st.button("Load Data", key="load_data"):
            st.session_state.data = {
                'X': X, 'y': y, 
                'feature_names': feature_names,
                'task_type': task_type
            }
            st.success(f"✅ Data loaded! Shape: {X.shape}")
    
    with col2:
        st.subheader("🔍 Data Info")
        if st.session_state.data is not None:
            data_dict = st.session_state.data
            X = data_dict['X']
            y = data_dict['y']
            
            st.write(f"**Samples:** {X.shape[0]}")
            st.write(f"**Features:** {X.shape[1]}")
            st.write(f"**Task Type:** {data_dict['task_type'].title()}")
            
            if data_dict['task_type'] == 'classification':
                unique_classes = len(np.unique(y))
                st.write(f"**Classes:** {unique_classes}")
                st.write(f"**Class Distribution:**")
                class_dist = pd.Series(y).value_counts().sort_index()
                st.bar_chart(class_dist)
            else:
                st.write(f"**Target Range:** [{y.min():.2f}, {y.max():.2f}]")
                st.write(f"**Target Mean:** {y.mean():.2f}")
                st.write(f"**Target Std:** {y.std():.2f}")
    
    # Data Visualization
    if st.session_state.data is not None:
        st.subheader("📈 Feature Analysis")
        data_dict = st.session_state.data
        X = data_dict['X']
        feature_names = data_dict['feature_names']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for idx, ax in enumerate(axes.flat):
                if idx < len(feature_names):
                    ax.hist(X[:, idx], bins=30, edgecolor='black', alpha=0.7)
                    ax.set_title(f"Distribution: {feature_names[idx]}")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            corr_matrix = np.corrcoef(X.T)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       xticklabels=feature_names[:8], yticklabels=feature_names[:8], ax=ax)
            ax.set_title("Feature Correlation Matrix")
            st.pyplot(fig)
    
    # Preprocessing
    if st.session_state.data is not None:
        st.subheader("⚙️ Data Preprocessing")
        col1, col2 = st.columns(2)
        
        with col1:
            scaler_type = st.selectbox(
                "Select Scaler:",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )
        
        with col2:
            test_size = st.slider("Test Size (%):", 10, 50, 20) / 100
        
        if st.button("Preprocess Data"):
            X = st.session_state.data['X']
            y = st.session_state.data['y']
            
            if scaler_type != "None":
                if scaler_type == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_type == "MinMaxScaler":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            st.session_state.data['X_train'] = X_train
            st.session_state.data['X_test'] = X_test
            st.session_state.data['y_train'] = y_train
            st.session_state.data['y_test'] = y_test
            st.session_state.data['scaler'] = scaler if scaler_type != "None" else None
            
            st.success("✅ Data preprocessed and split!")
            st.write(f"Training set: {X_train.shape[0]} samples")
            st.write(f"Test set: {X_test.shape[0]} samples")

# ==================== PAGE 2: MODEL CONFIGURATION ====================
elif page == "⚙️ Model Configuration":
    st.header("⚙️ Model Configuration & Hyperparameters")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load and preprocess data first!")
    else:
        task_type = st.session_state.data['task_type']
        
        # Model Selection
        st.subheader("🎯 Select Model")
        
        if task_type == "classification":
            models_list = [
                "Logistic Regression", "Decision Tree", "Random Forest",
                "Gradient Boosting", "Support Vector Machine", "K-Nearest Neighbors",
                "AdaBoost", "Voting Classifier"
            ]
        else:
            models_list = [
                "Linear Regression", "Ridge Regression", "Lasso Regression",
                "Decision Tree Regressor", "Random Forest Regressor",
                "Gradient Boosting Regressor", "SVR", "KNN Regressor"
            ]
        
        selected_model = st.selectbox("Choose Model:", models_list)
        
        st.subheader(f"🔧 Hyperparameters: {selected_model}")
        
        hyperparams = {}
        
        # Classification Models
        if task_type == "classification":
            if selected_model == "Logistic Regression":
                hyperparams = {
                    'C': st.slider("C (Regularization strength):", 0.001, 100.0, 1.0),
                    'penalty': st.selectbox("Penalty:", ['l2', 'l1']),
                    'solver': st.selectbox("Solver:", ['lbfgs', 'liblinear', 'saga']),
                    'max_iter': st.slider("Max Iterations:", 100, 5000, 1000),
                    'random_state': 42
                }
            
            elif selected_model == "Decision Tree":
                hyperparams = {
                    'max_depth': st.slider("Max Depth:", 1, 50, 10),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'criterion': st.selectbox("Criterion:", ['gini', 'entropy']),
                    'splitter': st.selectbox("Splitter:", ['best', 'random']),
                    'random_state': 42
                }
            
            elif selected_model == "Random Forest":
                hyperparams = {
                    'n_estimators': st.slider("Number of Trees:", 10, 500, 100),
                    'max_depth': st.slider("Max Depth:", 1, 50, 15),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'max_features': st.selectbox("Max Features:", ['sqrt', 'log2', None]),
                    'criterion': st.selectbox("Criterion:", ['gini', 'entropy']),
                    'n_jobs': -1,
                    'random_state': 42
                }
            
            elif selected_model == "Gradient Boosting":
                hyperparams = {
                    'n_estimators': st.slider("Number of Estimators:", 10, 500, 100),
                    'learning_rate': st.slider("Learning Rate:", 0.001, 1.0, 0.1),
                    'max_depth': st.slider("Max Depth:", 1, 20, 3),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'subsample': st.slider("Subsample:", 0.1, 1.0, 0.8),
                    'random_state': 42
                }
            
            elif selected_model == "Support Vector Machine":
                hyperparams = {
                    'C': st.slider("C (Regularization):", 0.001, 100.0, 1.0),
                    'kernel': st.selectbox("Kernel:", ['linear', 'rbf', 'poly', 'sigmoid']),
                    'gamma': st.selectbox("Gamma:", ['scale', 'auto']),
                    'degree': st.slider("Degree (for poly):", 2, 10, 3),
                    'probability': True,
                    'random_state': 42
                }
            
            elif selected_model == "K-Nearest Neighbors":
                hyperparams = {
                    'n_neighbors': st.slider("Number of Neighbors:", 1, 30, 5),
                    'weights': st.selectbox("Weights:", ['uniform', 'distance']),
                    'metric': st.selectbox("Distance Metric:", ['euclidean', 'manhattan', 'minkowski']),
                    'p': st.slider("Power Parameter (for Minkowski):", 1, 5, 2),
                    'n_jobs': -1
                }
            
            elif selected_model == "AdaBoost":
                hyperparams = {
                    'n_estimators': st.slider("Number of Estimators:", 10, 500, 50),
                    'learning_rate': st.slider("Learning Rate:", 0.1, 2.0, 1.0),
                    'algorithm': st.selectbox("Algorithm:", ['SAMME', 'SAMME.R']),
                    'random_state': 42
                }
            
            elif selected_model == "Voting Classifier":
                hyperparams = {
                    'voting': st.selectbox("Voting:", ['hard', 'soft']),
                    'n_jobs': -1
                }
        
        # Regression Models
        else:
            if selected_model == "Linear Regression":
                hyperparams = {
                    'fit_intercept': st.checkbox("Fit Intercept", True),
                    'n_jobs': -1
                }
            
            elif selected_model == "Ridge Regression":
                hyperparams = {
                    'alpha': st.slider("Alpha (Regularization):", 0.001, 100.0, 1.0),
                    'fit_intercept': st.checkbox("Fit Intercept", True),
                    'solver': st.selectbox("Solver:", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
                }
            
            elif selected_model == "Lasso Regression":
                hyperparams = {
                    'alpha': st.slider("Alpha (Regularization):", 0.001, 10.0, 0.1),
                    'fit_intercept': st.checkbox("Fit Intercept", True),
                    'max_iter': st.slider("Max Iterations:", 100, 10000, 1000)
                }
            
            elif selected_model == "Decision Tree Regressor":
                hyperparams = {
                    'max_depth': st.slider("Max Depth:", 1, 50, 10),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'criterion': st.selectbox("Criterion:", ['squared_error', 'absolute_error']),
                    'random_state': 42
                }
            
            elif selected_model == "Random Forest Regressor":
                hyperparams = {
                    'n_estimators': st.slider("Number of Trees:", 10, 500, 100),
                    'max_depth': st.slider("Max Depth:", 1, 50, 15),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'max_features': st.selectbox("Max Features:", ['sqrt', 'log2', None]),
                    'criterion': st.selectbox("Criterion:", ['squared_error', 'absolute_error']),
                    'n_jobs': -1,
                    'random_state': 42
                }
            
            elif selected_model == "Gradient Boosting Regressor":
                from sklearn.ensemble import GradientBoostingRegressor
                hyperparams = {
                    'n_estimators': st.slider("Number of Estimators:", 10, 500, 100),
                    'learning_rate': st.slider("Learning Rate:", 0.001, 1.0, 0.1),
                    'max_depth': st.slider("Max Depth:", 1, 20, 3),
                    'min_samples_split': st.slider("Min Samples Split:", 2, 20, 2),
                    'min_samples_leaf': st.slider("Min Samples Leaf:", 1, 20, 1),
                    'subsample': st.slider("Subsample:", 0.1, 1.0, 0.8),
                    'random_state': 42
                }
            
            elif selected_model == "SVR":
                hyperparams = {
                    'C': st.slider("C (Regularization):", 0.001, 100.0, 1.0),
                    'kernel': st.selectbox("Kernel:", ['linear', 'rbf', 'poly', 'sigmoid']),
                    'gamma': st.selectbox("Gamma:", ['scale', 'auto']),
                    'degree': st.slider("Degree (for poly):", 2, 10, 3),
                    'epsilon': st.slider("Epsilon:", 0.001, 1.0, 0.1)
                }
            
            elif selected_model == "KNN Regressor":
                hyperparams = {
                    'n_neighbors': st.slider("Number of Neighbors:", 1, 30, 5),
                    'weights': st.selectbox("Weights:", ['uniform', 'distance']),
                    'metric': st.selectbox("Distance Metric:", ['euclidean', 'manhattan', 'minkowski']),
                    'n_jobs': -1
                }
        
        # Store configuration
        st.session_state.model_config = {
            'model_name': selected_model,
            'hyperparams': hyperparams,
            'task_type': task_type
        }
        
        # Display hyperparameters summary
        st.subheader("📋 Configuration Summary")
        config_df = pd.DataFrame(list(hyperparams.items()), columns=["Parameter", "Value"])
        st.dataframe(config_df, use_container_width=True)

# ==================== PAGE 3: TRAINING & TUNING ====================
elif page == "🎯 Training & Tuning":
    st.header("🎯 Model Training & Hyperparameter Tuning")
    
    if st.session_state.data is None or 'X_train' not in st.session_state.data:
        st.warning("⚠️ Please load and preprocess data first!")
    elif 'model_config' not in st.session_state:
        st.warning("⚠️ Please configure model first!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Training Options")
            training_mode = st.radio(
                "Select training mode:",
                ["Standard Training", "Hyperparameter Tuning"]
            )
        
        with col2:
            st.subheader("📊 Cross Validation")
            cv_folds = st.slider("CV Folds:", 2, 10, 5)
        
        if training_mode == "Standard Training":
            st.subheader("📈 Train Model")
            
            if st.button("🚀 Train Model", key="train_model"):
                with st.spinner("Training model..."):
                    data = st.session_state.data
                    config = st.session_state.model_config
                    
                    X_train = data['X_train']
                    y_train = data['y_train']
                    X_test = data['X_test']
                    y_test = data['y_test']
                    
                    # Initialize model
                    model_name = config['model_name']
                    hyperparams = config['hyperparams']
                    task_type = config['task_type']
                    
                    if task_type == "classification":
                        if model_name == "Logistic Regression":
                            model = LogisticRegression(**hyperparams)
                        elif model_name == "Decision Tree":
                            model = DecisionTreeClassifier(**hyperparams)
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier(**hyperparams)
                        elif model_name == "Gradient Boosting":
                            model = GradientBoostingClassifier(**hyperparams)
                        elif model_name == "Support Vector Machine":
                            model = SVC(**hyperparams)
                        elif model_name == "K-Nearest Neighbors":
                            model = KNeighborsClassifier(**hyperparams)
                        elif model_name == "AdaBoost":
                            model = AdaBoostClassifier(**hyperparams)
                        elif model_name == "Voting Classifier":
                            lr = LogisticRegression(max_iter=1000, random_state=42)
                            rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            svm = SVC(probability=True, random_state=42)
                            model = VotingClassifier(
                                estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
                                voting=hyperparams.get('voting', 'soft')
                            )
                    else:
                        if model_name == "Linear Regression":
                            model = LinearRegression(**hyperparams)
                        elif model_name == "Ridge Regression":
                            model = Ridge(**hyperparams)
                        elif model_name == "Lasso Regression":
                            model = Lasso(**hyperparams)
                        elif model_name == "Decision Tree Regressor":
                            model = DecisionTreeRegressor(**hyperparams)
                        elif model_name == "Random Forest Regressor":
                            model = RandomForestRegressor(**hyperparams)
                        elif model_name == "Gradient Boosting Regressor":
                            from sklearn.ensemble import GradientBoostingRegressor
                            model = GradientBoostingRegressor(**hyperparams)
                        elif model_name == "SVR":
                            model = SVR(**hyperparams)
                        elif model_name == "KNN Regressor":
                            model = KNeighborsRegressor(**hyperparams)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    if task_type == "classification":
                        metrics = {
                            'Train Accuracy': accuracy_score(y_train, y_pred_train),
                            'Test Accuracy': accuracy_score(y_test, y_pred_test),
                            'Precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                        }
                        if len(np.unique(y_test)) == 2:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
                    else:
                        metrics = {
                            'Train R² Score': r2_score(y_train, y_pred_train),
                            'Test R² Score': r2_score(y_test, y_pred_test),
                            'Train MAE': mean_absolute_error(y_train, y_pred_train),
                            'Test MAE': mean_absolute_error(y_test, y_pred_test),
                            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
                        }
                    
                    # Store results
                    st.session_state.model = model
                    st.session_state.model_name = model_name
                    st.session_state.predictions = {
                        'y_pred_train': y_pred_train,
                        'y_pred_test': y_pred_test
                    }
                    st.session_state.metrics = metrics
                    
                    # Cross-validation scores
                    if task_type == "classification":
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    
                    st.session_state.cv_scores = cv_scores
                    st.success("✅ Model trained successfully!")
        
        else:  # Hyperparameter Tuning
            st.subheader("🔍 Hyperparameter Tuning")
            
            tuning_method = st.selectbox(
                "Select tuning method:",
                ["Grid Search", "Random Search"]
            )
            
            st.write("**Note:** Select parameters to tune. Leave others unchanged for better performance.")
            
            data = st.session_state.data
            config = st.session_state.model_config
            
            model_name = config['model_name']
            task_type = config['task_type']
            X_train = data['X_train']
            y_train = data['y_train']
            
            # Common parameter grids
            param_grids = {}
            
            if model_name == "Logistic Regression":
                param_grids = {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l2', 'l1'],
                    'max_iter': [1000, 2000]
                }
            
            elif model_name == "Random Forest":
                param_grids = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            
            elif model_name == "Gradient Boosting":
                param_grids = {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            
            elif model_name == "Support Vector Machine":
                param_grids = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            
            elif model_name == "K-Nearest Neighbors":
                param_grids = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            
            else:
                st.info("Predefined tuning parameters not available for this model. Using default parameters.")
                param_grids = {}
            
            if param_grids:
                st.write("**Parameter Grid:**")
                st.json(param_grids)
                
                if st.button("🔍 Start Tuning"):
                    with st.spinner("Tuning hyperparameters... This may take a while..."):
                        # Initialize base model
                        hyperparams = config['hyperparams'].copy()
                        
                        if task_type == "classification":
                            if model_name == "Logistic Regression":
                                base_model = LogisticRegression(random_state=42)
                            elif model_name == "Random Forest":
                                base_model = RandomForestClassifier(random_state=42)
                            elif model_name == "Gradient Boosting":
                                base_model = GradientBoostingClassifier(random_state=42)
                            elif model_name == "Support Vector Machine":
                                base_model = SVC(probability=True, random_state=42)
                            elif model_name == "K-Nearest Neighbors":
                                base_model = KNeighborsClassifier()
                        
                        # Perform tuning
                        if tuning_method == "Grid Search":
                            search = GridSearchCV(base_model, param_grids, cv=cv_folds, 
                                               scoring='accuracy' if task_type == "classification" else 'r2',
                                               n_jobs=-1)
                        else:
                            search = RandomizedSearchCV(base_model, param_grids, cv=cv_folds, 
                                                      scoring='accuracy' if task_type == "classification" else 'r2',
                                                      n_iter=10, random_state=42, n_jobs=-1)
                        
                        search.fit(X_train, y_train)
                        
                        st.session_state.model = search.best_estimator_
                        st.session_state.model_name = model_name
                        
                        # Display results
                        st.success("✅ Tuning completed!")
                        st.write(f"**Best Parameters:** {search.best_params_}")
                        st.write(f"**Best CV Score:** {search.best_score_:.4f}")
                        
                        # Results dataframe
                        results_df = pd.DataFrame(search.cv_results_)
                        st.dataframe(results_df[['param_' + k for k in search.best_params_.keys()] + 
                                               ['mean_test_score', 'std_test_score']], use_container_width=True)
        
        # Display metrics
        if st.session_state.metrics:
            st.subheader("📊 Model Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = st.session_state.metrics
            metric_items = list(metrics.items())
            
            for idx, (metric_name, value) in enumerate(metric_items[:4]):
                with [col1, col2, col3, col4][idx]:
                    st.metric(metric_name, f"{value:.4f}")
            
            # Additional metrics
            for idx, (metric_name, value) in enumerate(metric_items[4:]):
                st.metric(metric_name, f"{value:.4f}")
            
            # CV Scores
            if 'cv_scores' in st.session_state:
                st.write("**Cross-Validation Scores:**")
                cv_col1, cv_col2, cv_col3 = st.columns(3)
                with cv_col1:
                    st.metric("Mean CV Score", f"{st.session_state.cv_scores.mean():.4f}")
                with cv_col2:
                    st.metric("Std CV Score", f"{st.session_state.cv_scores.std():.4f}")
                with cv_col3:
                    st.metric("Min CV Score", f"{st.session_state.cv_scores.min():.4f}")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(st.session_state.cv_scores)), st.session_state.cv_scores, 
                       alpha=0.7, edgecolor='black')
                ax.axhline(y=st.session_state.cv_scores.mean(), color='r', 
                          linestyle='--', label='Mean')
                ax.set_xlabel("Fold")
                ax.set_ylabel("Score")
                ax.set_title("Cross-Validation Scores")
                ax.legend()
                st.pyplot(fig)

# ==================== PAGE 4: VISUALIZATION ====================
elif page == "📈 Visualization":
    st.header("📈 Advanced Visualization")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        data = st.session_state.data
        X_test = data['X_test']
        y_test = data['y_test']
        task_type = data['task_type']
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization:",
            ["Decision Boundary (2D)", "Feature Importance", "Residual Plot", 
             "Confusion Matrix", "ROC Curve", "Precision-Recall Curve", 
             "Prediction vs Actual"]
        )
        
        # Decision Boundary (2D)
        if viz_type == "Decision Boundary (2D)":
            st.subheader("🎯 Decision Boundary Visualization")
            
            if X_test.shape[1] >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    feature_1 = st.selectbox(
                        "Select First Feature:",
                        range(X_test.shape[1]),
                        format_func=lambda x: data['feature_names'][x] if x < len(data['feature_names']) else f"Feature {x}"
                    )
                
                with col2:
                    feature_2 = st.selectbox(
                        "Select Second Feature:",
                        range(X_test.shape[1]),
                        format_func=lambda x: data['feature_names'][x] if x < len(data['feature_names']) else f"Feature {x}",
                        index=1 if X_test.shape[1] > 1 else 0
                    )
                
                if st.button("🎨 Generate Decision Boundary"):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    X = X_test[:, [feature_1, feature_2]]
                    
                    # Create mesh
                    h = 0.02
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                        np.arange(y_min, y_max, h))
                    
                    # Create prediction surface
                    mesh_data = np.zeros((xx.shape[0] * xx.shape[1], X_test.shape[1]))
                    mesh_data[:, feature_1] = xx.ravel()
                    mesh_data[:, feature_2] = yy.ravel()
                    
                    # Copy other features from test set mean
                    for i in range(X_test.shape[1]):
                        if i not in [feature_1, feature_2]:
                            mesh_data[:, i] = X_test[:, i].mean()
                    
                    Z = st.session_state.model.predict(mesh_data)
                    Z = Z.reshape(xx.shape)
                    
                    # Plot decision boundary
                    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
                    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, levels=5)
                    
                    # Plot data points
                    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_test, cmap='RdYlBu', 
                                       edgecolors='black', s=100, alpha=0.7)
                    
                    feat_name_1 = data['feature_names'][feature_1] if feature_1 < len(data['feature_names']) else f"Feature {feature_1}"
                    feat_name_2 = data['feature_names'][feature_2] if feature_2 < len(data['feature_names']) else f"Feature {feature_2}"
                    
                    ax.set_xlabel(feat_name_1, fontsize=12)
                    ax.set_ylabel(feat_name_2, fontsize=12)
                    ax.set_title(f"Decision Boundary: {feat_name_1} vs {feat_name_2}", fontsize=14, fontweight='bold')
                    
                    plt.colorbar(scatter, ax=ax, label='Class')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("Need at least 2 features for decision boundary visualization!")
        
        # Feature Importance
        elif viz_type == "Feature Importance":
            st.subheader("🌟 Feature Importance")
            
            if hasattr(st.session_state.model, 'feature_importances_'):
                importances = st.session_state.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                feature_names_short = [data['feature_names'][i][:20] if i < len(data['feature_names']) 
                                      else f"Feature {i}" for i in range(len(importances))]
                
                ax.bar(range(len(importances)), importances[indices], alpha=0.7, edgecolor='black')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names_short[i] for i in indices], rotation=45, ha='right')
                ax.set_ylabel("Importance")
                ax.set_title("Feature Importance", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display as table
                importance_df = pd.DataFrame({
                    'Feature': [data['feature_names'][i][:50] if i < len(data['feature_names']) else f"Feature {i}" 
                               for i in indices],
                    'Importance': importances[indices]
                })
                st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("Model does not support feature importance extraction.")
        
        # Residual Plot (Regression)
        elif viz_type == "Residual Plot":
            if task_type == "regression":
                y_pred = st.session_state.predictions['y_pred_test']
                residuals = y_test - y_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
                    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residual Plot")
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_xlabel("Residuals")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Residuals")
                    st.pyplot(fig)
            else:
                st.info("Residual plot is only available for regression models.")
        
        # Confusion Matrix
        elif viz_type == "Confusion Matrix":
            if task_type == "classification":
                y_pred = st.session_state.predictions['y_pred_test']
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix", fontweight='bold')
                st.pyplot(fig)
                
                # Classification report
                st.text(classification_report(y_test, y_pred))
            else:
                st.info("Confusion matrix is only available for classification models.")
        
        # ROC Curve
        elif viz_type == "ROC Curve":
            if task_type == "classification" and len(np.unique(y_test)) == 2:
                y_pred_proba = st.session_state.model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve', fontweight='bold')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.info("ROC Curve is only available for binary classification.")
        
        # Precision-Recall Curve
        elif viz_type == "Precision-Recall Curve":
            if task_type == "classification" and len(np.unique(y_test)) == 2:
                y_pred_proba = st.session_state.model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(recall, precision, color='blue', lw=2)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve', fontweight='bold')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("Precision-Recall Curve is only available for binary classification.")
        
        # Prediction vs Actual
        elif viz_type == "Prediction vs Actual":
            y_pred = st.session_state.predictions['y_pred_test']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', s=100)
            
            if task_type == "regression":
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Prediction vs Actual", fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ==================== PAGE 5: MODEL ANALYSIS ====================
elif page == "🔍 Model Analysis":
    st.header("🔍 Comprehensive Model Analysis")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Model Summary", "Detailed Metrics", "Learning Curves", "Prediction Analysis"]
        )
        
        if analysis_type == "Model Summary":
            st.subheader("📋 Model Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Information:**")
                st.write(f"- **Model Name:** {st.session_state.model_name}")
                st.write(f"- **Task Type:** {st.session_state.data['task_type'].title()}")
                st.write(f"- **Training Samples:** {st.session_state.data['X_train'].shape[0]}")
                st.write(f"- **Testing Samples:** {st.session_state.data['X_test'].shape[0]}")
                st.write(f"- **Number of Features:** {st.session_state.data['X_train'].shape[1]}")
            
            with col2:
                st.write("**Model Parameters:**")
                st.json(st.session_state.model.get_params())
            
            st.subheader("📊 Performance Metrics")
            metrics_df = pd.DataFrame(
                list(st.session_state.metrics.items()),
                columns=["Metric", "Value"]
            )
            st.dataframe(metrics_df, use_container_width=True)
        
        elif analysis_type == "Detailed Metrics":
            st.subheader("🎯 Detailed Performance Metrics")
            
            data = st.session_state.data
            y_test = data['y_test']
            y_pred = st.session_state.predictions['y_pred_test']
            task_type = data['task_type']
            
            if task_type == "classification":
                # Multi-class metrics
                st.write("**Per-Class Metrics:**")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Overall metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                with col2:
                    st.metric("Macro Precision", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                with col3:
                    st.metric("Macro Recall", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                with col4:
                    st.metric("Macro F1", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                with col5:
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = st.session_state.model.predict_proba(data['X_test'])[:, 1]
                        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
            
            else:  # Regression
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                with col2:
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
                with col3:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                with col4:
                    st.metric("MAPE", f"{np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.4f}%")
                with col5:
                    st.metric("Max Error", f"{np.max(np.abs(y_test - y_pred)):.4f}")
        
        elif analysis_type == "Learning Curves":
            st.subheader("📚 Learning Curves")
            
            data = st.session_state.data
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            task_type = data['task_type']
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores_mean = []
            test_scores_mean = []
            
            for size in train_sizes:
                indices = np.random.choice(len(X_train), int(size * len(X_train)), replace=False)
                X_train_subset = X_train[indices]
                y_train_subset = y_train[indices]
                
                clone_model = st.session_state.model.__class__(**st.session_state.model.get_params())
                clone_model.fit(X_train_subset, y_train_subset)
                
                if task_type == "classification":
                    train_score = clone_model.score(X_train_subset, y_train_subset)
                    test_score = clone_model.score(X_test, y_test)
                else:
                    train_score = r2_score(y_train_subset, clone_model.predict(X_train_subset))
                    test_score = r2_score(y_test, clone_model.predict(X_test))
                
                train_scores_mean.append(train_score)
                test_scores_mean.append(test_score)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_sizes * len(X_train), train_scores_mean, 'o-', label='Training Score', linewidth=2)
            ax.plot(train_sizes * len(X_train), test_scores_mean, 'o-', label='Test Score', linewidth=2)
            ax.set_xlabel("Training Set Size")
            ax.set_ylabel("Score")
            ax.set_title("Learning Curves", fontweight='bold')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        elif analysis_type == "Prediction Analysis":
            st.subheader("🔎 Prediction Analysis")
            
            data = st.session_state.data
            y_test = data['y_test']
            y_pred = st.session_state.predictions['y_pred_test']
            task_type = data['task_type']
            
            if task_type == "classification":
                # Correct vs incorrect predictions
                correct = y_test == y_pred
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correct Predictions", f"{correct.sum()}/{len(y_test)}")
                with col2:
                    st.metric("Accuracy", f"{correct.mean():.2%}")
                with col3:
                    st.metric("Incorrect", f"{(~correct).sum()}")
                
                # Show misclassified samples
                if (~correct).sum() > 0:
                    st.write("**Sample of Misclassified Data:**")
                    misclassified_idx = np.where(~correct)[0][:10]
                    
                    results = []
                    for idx in misclassified_idx:
                        results.append({
                            'Actual': y_test[idx],
                            'Predicted': y_pred[idx],
                            'Confidence': max(st.session_state.model.predict_proba(data['X_test'])[idx])
                        })
                    
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
            
            else:  # Regression
                errors = np.abs(y_test - y_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Error", f"{errors.mean():.4f}")
                with col2:
                    st.metric("Median Error", f"{np.median(errors):.4f}")
                with col3:
                    st.metric("Std Error", f"{errors.std():.4f}")
                with col4:
                    st.metric("Max Error", f"{errors.max():.4f}")
                
                # Error distribution
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(errors.mean(), color='r', linestyle='--', label='Mean Error')
                ax.set_xlabel("Absolute Error")
                ax.set_ylabel("Frequency")
                ax.set_title("Error Distribution")
                ax.legend()
                st.pyplot(fig)

# ==================== PAGE 6: MODEL MANAGEMENT ====================
elif page == "💾 Model Management":
    st.header("💾 Model Management & Export")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Save Model")
            
            model_name = st.text_input(
                "Model Name:",
                value=f"{st.session_state.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            if st.button("🔽 Download Model (PKL)"):
                model_bytes = joblib.dumps(st.session_state.model)
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"
                )
                st.success("✅ Model ready for download!")
        
        with col2:
            st.subheader("📊 Model Report")
            
            report = f"""
# ML Model Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Model Type:** {st.session_state.model_name}
- **Task:** {st.session_state.data['task_type'].title()}

## Dataset
- **Training Samples:** {st.session_state.data['X_train'].shape[0]}
- **Test Samples:** {st.session_state.data['X_test'].shape[0]}
- **Features:** {st.session_state.data['X_train'].shape[1]}

## Performance Metrics
"""
            for metric, value in st.session_state.metrics.items():
                report += f"- **{metric}:** {value:.4f}\n"
            
            report += f"""
## Model Parameters
```
{st.session_state.model.get_params()}
```
"""
            
            st.download_button(
                label="📥 Download Report (MD)",
                data=report,
                file_name=f"{model_name}_report.md",
                mime="text/markdown"
            )
        
        # Model comparison history
        st.subheader("📈 Model Training History")
        st.info("Model comparison and history features coming soon!")
        
        # Export predictions
        st.subheader("📤 Export Predictions")
        
        data = st.session_state.data
        y_pred_test = st.session_state.predictions['y_pred_test']
        
        export_df = pd.DataFrame({
            'Actual': data['y_test'],
            'Predicted': y_pred_test
        })
        
        if data['task_type'] == 'classification' and hasattr(st.session_state.model, 'predict_proba'):
            proba = st.session_state.model.predict_proba(data['X_test'])
            for i in range(proba.shape[1]):
                export_df[f'Probability_Class_{i}'] = proba[:, i]
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{model_name}_predictions.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 <b>Advanced ML Workbench Pro</b> v1.0</p>
    <p style='color: gray;'>Premium Machine Learning Platform with Advanced Hyperparameter Tuning & Visualization</p>
</div>
""", unsafe_allow_html=True)
