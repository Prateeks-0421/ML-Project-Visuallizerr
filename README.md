# 🤖 Advanced ML Workbench Pro

> **Premium Machine Learning Platform** with Advanced Hyperparameter Tuning, Decision Boundary Visualization, and Comprehensive Model Analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Key Features

### 1. **📊 Data Exploration & Preprocessing**
- Load built-in datasets (Iris, Wine, Breast Cancer, Synthetic)
- Upload custom CSV files
- Automatic data splitting with configurable test size
- Multiple scaling options:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
- Feature correlation analysis
- Data distribution visualization

### 2. **⚙️ Model Configuration**
- **Classification Models** (8 algorithms):
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
  - K-Nearest Neighbors
  - AdaBoost
  - Voting Classifier

- **Regression Models** (8 algorithms):
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor
  - KNN Regressor

- **Full Hyperparameter Control**: Every available hyperparameter with:
  - Interactive sliders
  - Dropdown selectors
  - Real-time validation

### 3. **🎯 Training & Tuning**
- **Standard Training**: Quick model training with custom hyperparameters
- **Advanced Hyperparameter Tuning**:
  - Grid Search with exhaustive parameter testing
  - Random Search for efficient exploration
  - Cross-validation support (2-10 folds)
- Real-time training feedback
- Automatic cross-validation score computation

### 4. **📈 Advanced Visualization**
- **Decision Boundary (2D)**:
  - Interactive feature selection
  - Contour plots with decision regions
  - Color-coded class separation visualization
  - Mesh-based boundary computation

- **Feature Importance**:
  - Automatic extraction from tree-based models
  - Sortable importance ranking
  - Bar chart visualization

- **Performance Plots**:
  - Confusion Matrix (Classification)
  - ROC Curve (Binary Classification)
  - Precision-Recall Curve (Binary Classification)
  - Residual Plots (Regression)
  - Prediction vs Actual (All tasks)

### 5. **🔍 Comprehensive Model Analysis**
- Model summary with architecture details
- Detailed per-class metrics (Classification)
- Learning curves analysis
- Prediction analysis with error statistics
- Misclassification detection
- Error distribution analysis (Regression)

### 6. **💾 Model Management**
- Download trained models (PKL format)
- Export prediction results (CSV)
- Probability predictions export (Classification)
- Automatic model reports (Markdown)
- Training history tracking

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository** (or download files)
```bash
git clone <your-repo-url>
cd ml-workbench
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run ml_workbench_advanced.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 User Guide

### Step 1: Load Data 📊
1. Navigate to **📊 Data Exploration**
2. Choose data source:
   - Built-in datasets (Iris, Wine, Breast Cancer, Synthetic)
   - Upload custom CSV file
3. Configure preprocessing:
   - Select scaler type
   - Set train-test split ratio
4. Click "Preprocess Data"

### Step 2: Configure Model ⚙️
1. Navigate to **⚙️ Model Configuration**
2. Select model type from the list
3. Adjust hyperparameters using interactive controls
4. Review configuration summary

### Step 3: Train & Tune 🎯
1. Navigate to **🎯 Training & Tuning**
2. Choose training mode:
   - **Standard Training**: Train with configured hyperparameters
   - **Hyperparameter Tuning**: 
     - Grid Search (exhaustive)
     - Random Search (efficient)
3. Set cross-validation folds
4. Click train/tune button
5. Monitor real-time progress and metrics

### Step 4: Visualize Results 📈
1. Navigate to **📈 Visualization**
2. Select visualization type:
   - Decision Boundary (2D)
   - Feature Importance
   - Confusion Matrix / ROC Curve / Precision-Recall
   - Residual Plots (Regression)
   - Prediction vs Actual
3. Configure visualization parameters
4. Generate and analyze plots

### Step 5: Analyze Model 🔍
1. Navigate to **🔍 Model Analysis**
2. Select analysis type:
   - Model Summary
   - Detailed Metrics
   - Learning Curves
   - Prediction Analysis
3. Review detailed statistics and insights

### Step 6: Export & Save 💾
1. Navigate to **💾 Model Management**
2. Options:
   - Download trained model (PKL)
   - Download predictions (CSV)
   - Export model report (Markdown)

---

## 🎨 Hyperparameter Details

### Logistic Regression
- **C**: Regularization strength (0.001 - 100)
- **Penalty**: L1 or L2 regularization
- **Solver**: Optimization algorithm
- **Max Iterations**: Training iterations

### Random Forest
- **n_estimators**: Number of trees (10 - 500)
- **max_depth**: Maximum tree depth (1 - 50)
- **min_samples_split**: Minimum samples for split (2 - 20)
- **min_samples_leaf**: Minimum samples in leaf
- **max_features**: Feature selection strategy
- **Criterion**: Split quality measure

### Gradient Boosting
- **n_estimators**: Boosting stages (10 - 500)
- **learning_rate**: Shrinkage coefficient (0.001 - 1.0)
- **max_depth**: Tree depth (1 - 20)
- **subsample**: Training data fraction (0.1 - 1.0)
- **min_samples_split**: Minimum split samples
- **min_samples_leaf**: Minimum leaf samples

### Support Vector Machine
- **C**: Regularization parameter (0.001 - 100)
- **Kernel**: Linear, RBF, Polynomial, Sigmoid
- **Gamma**: Kernel coefficient
- **Degree**: Polynomial degree (for poly kernel)

### K-Nearest Neighbors
- **n_neighbors**: Number of neighbors (1 - 30)
- **weights**: Uniform or distance-weighted
- **metric**: Distance metric (euclidean, manhattan, minkowski)

---

## 📊 Metrics Explained

### Classification Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / all positive predictions
- **Recall**: True positives / all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (binary classification)
- **Confusion Matrix**: True/False Positives/Negatives

### Regression Metrics
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Max Error**: Maximum prediction error

---

## 🔧 Advanced Features

### Decision Boundary Visualization
- Supports 2D visualization of any two features
- Automatic mesh grid generation
- Contour plots showing decision regions
- Color-coded class representation
- Data points overlaid on boundary

### Hyperparameter Tuning
- **Grid Search**: 
  - Exhaustive parameter search
  - All combinations tested
  - Best parameter discovery
  
- **Random Search**:
  - Efficient exploration
  - Random sampling from parameter space
  - Faster than Grid Search for large spaces

### Cross-Validation
- k-fold cross-validation (2-10 folds)
- Automatic score computation
- Statistical summary (mean, std, min)
- Visual representation

---

## 🏗️ Project Structure

```
ml-workbench/
│
├── ml_workbench_advanced.py    # Main application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .streamlit/
│   └── config.toml             # Streamlit configuration
│
└── models/                      # (Optional) Saved models directory
    ├── model_1.pkl
    ├── model_2.pkl
    └── predictions_1.csv
```

---

## 🚀 Deployment

### Deploy on Streamlit Cloud

1. **Push code to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign up with GitHub
   - Click "New App"
   - Select your repository and branch
   - Deploy!

### Deploy on Heroku

1. **Create Procfile**
```
web: streamlit run ml_workbench_advanced.py --logger.level=error
```

2. **Create .streamlit/config.toml**
```toml
[server]
port = $PORT
enableCORS = false
headless = true
```

3. **Deploy**
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

## 💡 Tips & Best Practices

1. **Feature Scaling**: Always enable scaling for distance-based models (KNN, SVM)

2. **Imbalanced Data**: Use sampling techniques or adjust class weights

3. **Overfitting**: 
   - Use regularization (L1/L2)
   - Increase min_samples_split/leaf
   - Reduce model complexity

4. **Cross-Validation**: Higher folds = more reliable but slower

5. **Hyperparameter Tuning**:
   - Start with grid search for small parameter spaces
   - Use random search for large spaces
   - Focus on most impactful parameters

6. **Model Selection**: Use decision boundary visualization to understand model behavior

---

## 🐛 Troubleshooting

### Issue: "Data not loaded"
**Solution**: Click "Load Data" button in Data Exploration section

### Issue: "Model not trained"
**Solution**: Complete training in Training & Tuning section

### Issue: "Decision boundary not available"
**Solution**: Need at least 2 features; try synthetic data

### Issue: "Feature importance not available"
**Solution**: Only available for tree-based models; use feature importance visualization

---

## 📚 Additional Resources

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ML Best Practices](https://developers.google.com/machine-learning/guides)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Advanced ML Workbench Pro**
- Version: 1.0
- Created: 2024
- Maintained by: AI Development Team

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML Algorithms from [Scikit-Learn](https://scikit-learn.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

---

## ⭐ If you found this helpful, please star the repository!

---

**Last Updated**: 2024 | **Version**: 1.0.0

