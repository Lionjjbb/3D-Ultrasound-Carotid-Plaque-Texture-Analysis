import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import LeaveOneOut
from collections import Counter
from scipy.stats import ttest_ind

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Step 1: Load Data
data_path = r"C:\Users\Wing on\Desktop\model_X.xlsx"
labels_path = r"C:\Users\Wing on\Desktop\FYP-Label.xlsx"
data_df = pd.read_excel(data_path)
labels_df = pd.read_excel(labels_path)

# Step 2: Align Data and Labels
data_df.set_index('Unnamed: 0', inplace=True)
labels_df.set_index('Unnamed: 0', inplace=True)
common_indices = data_df.index.intersection(labels_df.index)
X = data_df.loc[common_indices].values  # (54, 376)
y = labels_df.loc[common_indices]['Label'].values  # (54,)
feature_names = data_df.columns

# Step 3: Skip Feature Scaling
X_unscaled = X

# Step 4: Define Models with Manually Tuned Parameters
models = {
    'Logistic Regression': LogisticRegression(
        penalty='l1', solver='liblinear', C=1.0, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=3, random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        objective='binary', num_leaves=3, learning_rate=0.01, 
        min_data_in_leaf=5, lambda_l1=1.0, n_estimators=100, 
        verbose=-1, random_state=42
    ),
    'K-NN': KNeighborsClassifier(
        n_neighbors=5, weights='distance', metric='euclidean'
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, probability=True, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.01,
        min_child_weight=5, gamma=0.1, random_state=42
    )

}

# Step 5: LOOCV with Feature Selection and 1D Mapping
loo = LeaveOneOut()
results = {}
feature_selection_counts = {name: Counter() for name in models}
one_d_outputs = {name: [] for name in models}  # Store 1D outputs for each model
true_labels = []

for name, model in models.items():
    for train_idx, test_idx in loo.split(X_unscaled):
        X_train, X_test = X_unscaled[train_idx], X_unscaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature Selection (Lasso on training data)
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
        l1_model.fit(X_train, y_train)
        top_5_features = np.argsort(np.abs(l1_model.coef_[0]))[::-1][:5]
        for idx in top_5_features:
            feature_selection_counts[name][feature_names[idx]] += 1

        # Subset data
        X_train_reduced = X_train[:, top_5_features]
        X_test_reduced = X_test[:, top_5_features]

        # Train model and get 1D output (probability)
        model.fit(X_train_reduced, y_train)
        one_d_output = model.predict_proba(X_test_reduced)[0][1]  # Probability of class 1
        one_d_outputs[name].append(one_d_output)
        
        if name == 'Logistic Regression':
            true_labels.append(int(y_test.item()))

    # Compute p-value using t-test between 1D outputs of different classes
    outputs_class0 = np.array([out for out, label in zip(one_d_outputs[name], true_labels) if label == 0])
    outputs_class1 = np.array([out for out, label in zip(one_d_outputs[name], true_labels) if label == 1])
    _, p_value = ttest_ind(outputs_class0, outputs_class1, alternative='two-sided', equal_var=False)
    results[name] = {'p_value': p_value}
    print(f"{name}: p-value between classes = {p_value:.4e}")

# Step 6: Display Results
print("\nFinal Results (p-values):")
for name, res in results.items():
    print(f"{name}: p-value = {res['p_value']:.4e}")

# Step 7: Feature Selection Summary
for name in models:
    print(f"\nTop 5 Features for {name}:")
    for feature, count in feature_selection_counts[name].most_common(5):
        print(f"- {feature}: {count}/{len(y)} folds")

# Step 7.5: Calculate and Display Average L1 Coefficients
print("\nAverage L1 Coefficients across all folds:")
l1_coefficients = {}

for train_idx, test_idx in loo.split(X_unscaled):
    X_train, X_test = X_unscaled[train_idx], X_unscaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Get L1 coefficients
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    l1_model.fit(X_train, y_train)
    coefs = l1_model.coef_[0]
    
    # Store coefficients
    for idx, coef in enumerate(coefs):
        feature_name = feature_names[idx]
        if feature_name not in l1_coefficients:
            l1_coefficients[feature_name] = []
        l1_coefficients[feature_name].append(coef)

# Calculate and display average coefficients
avg_coefficients = {feature: np.mean(coefs) for feature, coefs in l1_coefficients.items()}
sorted_features = sorted(avg_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 5 Features by Absolute Coefficient Value:")
for feature, coef in sorted_features[:5]:
    print(f"- {feature}: {coef:.4f}")

# Step 8: Display 1D Outputs
print("\n1D Outputs for each model:")
for name in models:
    print(f"{name} outputs: {[f'{x:.4f}' for x in one_d_outputs[name]]}")
print(f"True Labels: {true_labels}")



# Step 9: Class Distribution
print(f"\nClass Distribution: 0 (VB) = {np.sum(y == 0)}, 1 (Placebo) = {np.sum(y == 1)}")