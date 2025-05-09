# 3D ultrasound assessment of the effects of Vitamin B therapy on carotid plaque texture using machine learning

Project Overview

This project explores how Vitamin B therapy affects carotid artery plaque (fatty buildup in arteries) using 3D ultrasound images and machine learning. We aim to see if Vitamin B can help reduce the risk of heart disease, especially in areas with limited healthcare.

Goals

--Analyze 3D ultrasound images to study plaque texture.

--Compare Vitamin B treatment with a placebo using machine learning.

--Find key image features that show treatment differences.

--Suggest Vitamin B use to prevent heart disease.

Dataset

--The dataset includes 56 patients (27 on Vitamin B, 29 on placebo) with 224 artery records (112 left, 112 right; half before treatment, half after). It contains:

--Idd files: Measure blood flow and pressure.

--I3d files: Show 3D artery structure. After cleaning the data, we used 54 complete records (26 Vitamin B, 28 placebo).

Installation

Requirements

--Python 3.8 or higher

--Python packages: pandas, numpy, scikit-learn, lightgbm, xgboost, scipy, pyvista

External tools:

1. 3D Quantify

2. 3DImageAnalysis.exe

3. ParaView

4. ITK-SNAP

5. BChiuCarotidAnalysisProgram

6. MATLAB (for analyzing image textures)

Data Preparation:

Follow the steps from VesselWallTexturalAnalysisProcotol file to perform feature extraction from the raw data and preprocess the data 

Machine Learning Models (6models fpy.py):

Train and evaluate models (Logistic Regression, Random Forest, LightGBM, K-NN, SVM, XGBoost):

This script:

--Applies Leave-One-Out Cross-Validation (LOOCV).

--Selects top5 features using Lasso regularization.

--Trains models and predicts outcomes.

--Evaluates results with Welch's t-test.

--Evaluates on the selected features

Methodology

Feature Extraction

Nine methods extract 376 textural features:





1. Gray-level distribution (34 features)



2. Gray-level co-occurrence matrix (78 features)



3. Gray-level run-length matrix (66 features)



4. Gray-level difference statistics (12 features)



5. Neighborhood gray tone difference matrix (10 features)



6. Laws texture (105 features)



7. Local binary pattern (27 features)



8. Gaussian filter bank (24 features)



9. Structure tensor (20 features)

Models

Logistic Regression: Uses L1 regularization for feature selection.

Random Forest: Captures non-linear relationships.

LightGBM: Boosting for high-dimensional data.

K-NN: Distance-based classification.

SVM: RBF kernel for non-linear separation.

XGBoost: Boosting with regularization.

Evaluation



Validation: LOOCV (54 folds).
Statistical Test: Welch's t-test on model outputs.



Results:


--SVM: p = 8.6884e-11 (highly significant)

--Logistic Regression: p = 1.8706e-05

--K-NN: p = 2.1375e-03

--Random Forest, LightGBM, XGBoost: Not significant (p > 0.05)

Why Machine Learning Before t-Test?

Machine learning is used before the t-test to:

1. Integrate multiple features for complex pattern detection.

2. Select the most discriminative features (e.g., top 5 via Lasso).

3. Handle high-dimensional data and non-linear relationships.

4. Generate a 1D output (probability of placebo) for robust t-test evaluation. This enhances the t-test's focus on relevant features, improving statistical power and clinical relevance.

Results:


--SVM excelled due to clear class separation and RBF kernel.



--Top Features: Laws_LER_absmean, Laws_LER_mean, Laws_LLS_mean, FirstOrder_Maximum, NGTDM_60_Complexity.



--Conclusion: Vitamin B shows effects beyond placebo, supporting its role in CVD prevention.

Limitations:





--Small dataset (54 samples) limits generalizability.



--Incomplete files due to contouring and system errors.



--Tree-based models (LightGBM, XGBoost) struggled with small data and linear-biased feature selection.
