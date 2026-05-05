# 🚀 Advanced Features Guide

## ML Workbench Pro - Technical Documentation

---

## Table of Contents

1. [Decision Boundary Visualization](#decision-boundary-visualization)
2. [Hyperparameter Tuning Strategies](#hyperparameter-tuning-strategies)
3. [Model Selection Guide](#model-selection-guide)
4. [Performance Optimization](#performance-optimization)
5. [Custom Data Handling](#custom-data-handling)

---

## Decision Boundary Visualization

### How it Works

The decision boundary visualization creates a 2D representation of how a trained model classifies data points. This is particularly useful for understanding:

- Model behavior and decision regions
- Whether the model is overfitting or underfitting
- How different features interact

### Technical Implementation

```python
1. Feature Selection: Choose 2 features from your dataset
2. Create Mesh Grid: Generate a dense grid of points in the feature space
   - Range: [min-1, max+1] for each feature
   - Resolution: 0.02 (controlled by parameter h)
3. Prediction: Get model predictions for all mesh points
4. Plotting:
   - Contourf: Shows decision regions with colors
   - Contour: Shows decision boundaries
   - Scatter: Overlays actual data points
```

### Best Practices

- Use features with good separation for clearer boundaries
- Classification is best visualized (regression shows continuous values)
- Tree-based models show sharp boundaries; distance-based models show smooth transitions
- For high-dimensional data, use PCA or t-SNE to reduce to 2D

### Interpreting Boundaries

- **Sharp Boundaries**: Tree-based models (Decision Tree, Random Forest)
- **Smooth Boundaries**: Distance-based models (KNN, SVM with RBF kernel)
- **Linear Boundaries**: Linear models (Logistic Regression, SVM with linear kernel)
- **Complex Patterns**: Ensemble models (Gradient Boosting, Voting Classifier)

---

## Hyperparameter Tuning Strategies

### 1. Grid Search

#### When to Use
- Small parameter spaces (< 1000 combinations)
- When you know the approximate optimal range
- Exhaustive exploration needed

#### Advantages
- Guarantees finding the best combination in the grid
- Parallel computation supported
- Reproducible results

#### Disadvantages
- Can be slow for large parameter spaces
- Requires manual specification of search space

#### Example Configuration
```
n_estimators: [50, 100, 200]
max_depth: [5, 10, 15]
min_samples_split: [2, 5, 10]
Total combinations: 3 × 3 × 3 = 27
```

### 2. Random Search

#### When to Use
- Large parameter spaces (> 1000 combinations)
- When you're unsure about optimal ranges
- Quick exploration needed

#### Advantages
- More efficient than Grid Search for large spaces
- Can explore a broader range
- Often finds comparable results with less computation

#### Disadvantages
- May miss optimal combination
- Less systematic than Grid Search

#### Strategy
```
- Run with n_iter=10-50 first
- Identify promising regions
- Run Grid Search in those regions
```

### 3. Bayesian Optimization (Advanced)

#### Implementation Pattern
```python
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score

def objective(params):
    model = RandomForestClassifier(
        n_estimators=params[0],
        max_depth=params[1],
        min_samples_split=params[2]
    )
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return -score  # Negative because we minimize

space = [(50, 300), (3, 20), (2, 10)]
result = gp_minimize(objective, space, n_calls=30)
```

---

## Model Selection Guide

### Classification Problems

#### Logistic Regression
**Best for:**
- Binary classification
- Linear separability
- Interpretability needed

**Hyperparameters to tune:**
- C (regularization strength)
- penalty (L1 vs L2)

**When to use:**
- Baseline model
- Fast training needed
- Feature importance critical

---

#### Decision Tree
**Best for:**
- Non-linear relationships
- Feature interactions
- Interpretability

**Hyperparameters to tune:**
- max_depth
- min_samples_split
- min_samples_leaf

**When to use:**
- Small datasets
- Mixed feature types
- Interpretability required

---

#### Random Forest
**Best for:**
- Mixed feature types
- Non-linear patterns
- Robust predictions

**Hyperparameters to tune:**
- n_estimators (50-500)
- max_depth (5-30)
- min_samples_split (2-10)

**Tips:**
- Higher n_estimators is usually better (diminishing returns after 200)
- Reduce max_depth if overfitting
- Increase min_samples_split if overfitting

---

#### Gradient Boosting
**Best for:**
- High accuracy needed
- Competition/production systems
- Complex patterns

**Hyperparameters to tune:**
- n_estimators (50-500)
- learning_rate (0.01-0.3)
- max_depth (3-8)

**Tips:**
- Lower learning_rate requires more estimators
- Usually max_depth of 3-7 is best
- subsample < 1.0 improves generalization

---

#### Support Vector Machine (SVM)
**Best for:**
- High-dimensional data
- Binary classification
- Clear margin between classes

**Hyperparameters to tune:**
- C (regularization)
- kernel (rbf for most cases)
- gamma (kernel coefficient)

**Tips:**
- Scale features with StandardScaler
- RBF kernel is usually best for non-linear
- Larger C means tighter fit (overfitting risk)

---

#### K-Nearest Neighbors
**Best for:**
- Small datasets
- Non-linear relationships
- Baseline comparisons

**Hyperparameters to tune:**
- n_neighbors (3-15)
- weights (uniform vs distance)
- metric (euclidean is default)

**Tips:**
- Always scale features
- distance weighting often better than uniform
- Odd n_neighbors for binary classification

---

### Regression Problems

#### Linear Regression
**Best for:**
- Linear relationships
- Interpretability
- Baseline model

**When to use:**
- Simple linear patterns
- Feature coefficients needed
- Fast predictions

---

#### Ridge & Lasso Regression
**Best for:**
- Multicollinearity
- Feature selection (Lasso)
- Regularization needed

**Ridge:**
- Keeps all features
- Better for prediction
- Alpha: 0.001 to 100

**Lasso:**
- Feature selection (coefficients = 0)
- Sparse models
- Alpha: 0.001 to 10

---

#### Tree-Based Regression
**Best for:**
- Non-linear relationships
- Mixed feature types
- Feature interactions

**Hyperparameters:**
- Same as classification
- Criterion: squared_error or absolute_error
- absolute_error more robust to outliers

---

## Performance Optimization

### Data Preprocessing

```python
# 1. StandardScaler - Default choice
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 2. MinMaxScaler - For bounded features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)

# 3. RobustScaler - Outlier-resistant
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Training Optimization

```python
# Parallel Processing
model = RandomForestClassifier(n_jobs=-1)  # Use all cores

# Early Stopping
model = GradientBoostingClassifier(
    n_estimators=100,
    validation_fraction=0.2,
    n_iter_no_change=10
)
```

### Model Evaluation

```python
# K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

# Stratified K-Fold (for imbalanced data)
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

---

## Custom Data Handling

### Data Format Requirements

**CSV Format:**
```csv
feature1,feature2,feature3,target
1.5,2.3,0.8,0
2.1,1.5,0.9,1
3.0,3.2,1.1,0
...
```

**DataFrame Format:**
```python
import pandas as pd
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values
```

### Handling Missing Values

```python
# Before uploading, clean your data:

import pandas as pd
df = pd.read_csv('data.csv')

# Option 1: Drop rows with NaN
df = df.dropna()

# Option 2: Fill with mean
df = df.fillna(df.mean())

# Option 3: Fill with median (more robust)
df = df.fillna(df.median())

df.to_csv('cleaned_data.csv', index=False)
```

### Feature Scaling in Code

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

---

## Advanced Recipes

### 1. Handling Imbalanced Classification

```python
# Method 1: Adjust class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
model.fit(X_train, y_train, sample_weight=class_weights)

# Method 2: Use stratified splitting
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Method 3: SMOTE (Synthetic Minority Oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

### 2. Feature Engineering

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
```

### 3. Ensemble Techniques

```python
# Voting Classifier
from sklearn.ensemble import VotingClassifier

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)
svm = SVC(probability=True)

voting = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
    voting='soft'  # Use probabilities
)
voting.fit(X_train, y_train)
```

---

## Troubleshooting Guide

### Problem: Model Overfitting
**Indicators:**
- High training accuracy, low test accuracy
- Large gap in cross-validation scores

**Solutions:**
1. Increase regularization (higher C for SVM, lower learning_rate for GB)
2. Reduce model complexity (max_depth, n_estimators)
3. Increase min_samples_split and min_samples_leaf
4. Use more training data
5. Add feature scaling
6. Use cross-validation to detect early

### Problem: Model Underfitting
**Indicators:**
- Low training and test accuracy
- Simple decision boundaries

**Solutions:**
1. Increase model complexity
2. Lower regularization
3. Use more features
4. Increase training time/estimators
5. Try a more complex model

### Problem: Poor Generalization
**Indicators:**
- Model works well on training, poorly on test

**Solutions:**
1. Use cross-validation (not just train-test split)
2. Increase training data size
3. Feature scaling and normalization
4. Hyperparameter tuning
5. Ensemble methods

---

## Performance Benchmarks

### Typical Accuracies (Iris Dataset)
- Logistic Regression: ~97%
- Decision Tree: ~98%
- Random Forest: ~98%
- Gradient Boosting: ~99%
- SVM: ~98%
- KNN: ~96%

### Training Time (per fold, 1000 samples, 10 features)
- Logistic Regression: < 1ms
- Decision Tree: < 10ms
- Random Forest: 50-200ms
- Gradient Boosting: 100-500ms
- SVM: 10-100ms
- KNN: < 1ms (at prediction time)

---

## References & Further Reading

1. **Scikit-Learn Docs**: https://scikit-learn.org/stable/documentation.html
2. **Hyperparameter Tuning**: https://scikit-learn.org/stable/modules/grid_search.html
3. **Model Selection**: https://scikit-learn.org/stable/modules/model_evaluation.html
4. **Feature Scaling**: https://scikit-learn.org/stable/modules/preprocessing.html

---

**Last Updated**: 2024
**ML Workbench Pro v1.0**

