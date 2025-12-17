# Project-ML

Clustering &amp; Classification Data Analysis

---

##  **01_data_overview.ipynb - Data Overview & EDA + Evaluation**

This notebook handles dataset preparation, exploratory data analysis, and model evaluation.

### **Features:**

1. **Dataset Selection & Loading**
   - Loads classification dataset (`diabetes.csv`)
   - Loads clustering dataset (`retail_store_sales.csv`)
   - Reads raw datasets from `data/raw/`

2. **Exploratory Data Analysis (EDA)**
   - Dataset shape overview
   - Missing values analysis and summary
   - Duplicates detection and overview
   - Basic visualizations:
     - Distribution plots (Glucose distribution)
     - Scatter plots (BMI vs Glucose)
     - Correlation matrix heatmap for diabetes features

3. **Model Evaluation** 
   - **Accuracy Score** - Calculates accuracy for all supervised models
   - **Precision Score** - Computes weighted precision for each model
   - **Recall Score** - Computes weighted recall for each model
   - **F1 Score** - Computes weighted F1 score for each model
   - **ROC-AUC Score** - Calculates ROC-AUC for binary classification
   - **Confusion Matrix** - Creates visualizations for all models
   - **Classification Report** - Generates detailed per-class metrics

### **Input Files:**
- `data/raw/diabetes.csv` - Classification dataset
- `data/raw/retail_store_sales.csv` - Clustering dataset
- `data/processed/X_test_diabetes.csv` - Test features
- `data/processed/y_test_diabetes.csv` - Test labels
- `data/processed/y_pred_lr.csv` - Logistic Regression predictions
- `data/processed/y_pred_nb.csv` - Naive Bayes predictions
- `data/processed/y_pred_dt.csv` - Decision Tree predictions
- `data/processed/y_pred_svm.csv` - SVM predictions
- `data/processed/y_proba_lr.csv` - Logistic Regression probabilities
- `data/processed/y_proba_nb.csv` - Naive Bayes probabilities
- `data/processed/y_proba_dt.csv` - Decision Tree probabilities
- `data/processed/y_proba_svm.csv` - SVM probabilities

### **Output:**
- Evaluation results displayed in notebook
- Confusion matrix visualizations (2x2 grid)
- Classification reports for each model

### **Dependencies:**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (for evaluation metrics: `accuracy_score`, `f1_score`, `confusion_matrix`, `classification_report`)

---

##  **06_supervised_models.ipynb - Supervised Model Training**

This notebook trains supervised learning models with hyperparameter tuning and generates predictions.

### **Features:**

1. **Hyperparameter Tuning**
   - Uses **RandomizedSearchCV** and **GridSearchCV** for optimal parameter selection
   - Optimizes using **F1 score** (better for imbalanced datasets)
   - 5-fold cross-validation for robust parameter selection
   - All models use `class_weight='balanced'` to handle class imbalance

2. **Model Training**
   - Trains 4 supervised classification models with tuned hyperparameters:
     - **Logistic Regression** - Linear classification with L2 penalty, tuned C values
     - **Naive Bayes** - Probabilistic classifier with optimized var_smoothing
     - **Decision Tree** - Tree-based classifier with tuned depth, min_samples, and criterion
     - **SVM** - Support Vector Machine with tuned C, kernel, and gamma parameters

3. **Prediction Generation**
   - Generates predictions on test set for each model
   - Generates prediction probabilities for ROC-AUC calculation
   - Saves predictions and probabilities to CSV files for evaluation

4. **Model Saving**
   - Saves all trained models as pickle files using `joblib`
   - Models saved in `models/` directory for future use

5. **Model Status Tracking**
   - Displays best hyperparameters found for each model
   - Shows cross-validation scores
   - Simple comparison table showing training status
   - **Note**: This notebook does NOT calculate evaluation metrics

### **Input Files:**
- `data/processed/X_train_diabetes.csv` - Training features
- `data/processed/X_test_diabetes.csv` - Test features
- `data/processed/y_train_diabetes.csv` - Training labels
- `data/processed/y_test_diabetes.csv` - Test labels

### **Output Files:**
- `data/processed/y_pred_lr.csv` - Logistic Regression predictions
- `data/processed/y_pred_nb.csv` - Naive Bayes predictions
- `data/processed/y_pred_dt.csv` - Decision Tree predictions
- `data/processed/y_pred_svm.csv` - SVM predictions
- `data/processed/y_proba_lr.csv` - Logistic Regression probabilities
- `data/processed/y_proba_nb.csv` - Naive Bayes probabilities
- `data/processed/y_proba_dt.csv` - Decision Tree probabilities
- `data/processed/y_proba_svm.csv` - SVM probabilities
- `models/logistic_regression.pkl` - Saved Logistic Regression model
- `models/naive_bayes.pkl` - Saved Naive Bayes model
- `models/decision_tree.pkl` - Saved Decision Tree model
- `models/svm.pkl` - Saved SVM model

### **Dependencies:**
- `pandas`, `numpy`, `joblib`
- `scikit-learn` (for models: `LogisticRegression`, `GaussianNB`, `DecisionTreeClassifier`, `SVC`)
- `scikit-learn` (for tuning: `GridSearchCV`, `RandomizedSearchCV`)

### **Important Notes:**
-  **No evaluation metrics** are calculated in this notebook
- All evaluation (Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix) is handled by `01_data_overview.ipynb`
- Predictions and probabilities are saved as CSV files to be loaded and evaluated by the evaluation notebook
- Models are saved as pickle files for deployment or future predictions

---

##  **Data Flow Between Notebooks**

```
06_supervised_models.ipynb (Training):
    Hyperparameter Tuning → Train Models → Generate Predictions & Probabilities → Save to CSV & PKL
         ↓
    data/processed/y_pred_*.csv, y_proba_*.csv
    models/*.pkl
         ↓
01_data_overview.ipynb (Evaluation):
    Load Predictions & Probabilities → Calculate Metrics → Visualize Results
```

### **Step-by-Step Process:**

1. **Run `06_supervised_models.ipynb`:**
   - Performs hyperparameter tuning using RandomizedSearchCV/GridSearchCV
   - Trains all 4 models with optimal hyperparameters on training data
   - Generates predictions and probabilities on test set
   - Saves predictions and probabilities to CSV files in `data/processed/`
   - Saves trained models as pickle files in `models/` directory

2. **Run evaluation section in `01_data_overview.ipynb`:**
   - Loads predictions and probabilities from CSV files
   - Calculates Accuracy, Precision, Recall, F1 Score, and ROC-AUC for each model
   - Creates Confusion Matrix visualizations
   - Generates detailed Classification Reports

---

##  **Key Features - Updated Structure**

### **Separation of Concerns:**

-  **`01_data_overview.ipynb`** handles **Data Overview, EDA, and Evaluation** (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
-  **`06_supervised_models.ipynb`** focuses on **Hyperparameter Tuning and Model Training** (no evaluation metrics)
-  Clear separation: Training vs. Evaluation

### **Model Training Improvements:**

- **Hyperparameter Tuning**: All models use RandomizedSearchCV/GridSearchCV with F1 score optimization
- **Class Imbalance Handling**: All models use `class_weight='balanced'` for better performance on imbalanced data
- **Model Persistence**: All trained models are saved as pickle files for future use
- **5-Fold Cross-Validation**: Robust parameter selection using cross-validation

### **Benefits:**

- Better code organization and modularity
- Evaluation notebook has complete overview of data and results
- Training notebook focuses on hyperparameter tuning and model implementation
- Models are saved for deployment or future predictions
- Optimized for imbalanced binary classification tasks
- Easier maintenance and code review

---

# Clustering Preprocessing

## Data Preprocessing for Clustering Analysis

This notebook prepares the **cat breeds dataset** for clustering analysis. The preprocessing pipeline focuses on cleaning dirty data, handling missing and invalid values, removing outliers, encoding categorical features, and scaling numerical features. Optional visualizations are included to verify data quality after preprocessing.

---

## Dataset

* **Name:** Cat Breeds Dataset
* **File:** `cat_breeds_dirty.csv`
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/iflixxe16/cat-breeds](https://www.kaggle.com/datasets/iflixxe16/cat-breeds)
* **Rows × Columns:** 1103 × 17 (raw)

### Main Columns

* Breed
* Age_in_years, Age_in_months
* Gender
* Neutered_or_spayed
* Body_length, Weight
* Fur_colour_dominant, Fur_pattern, Eye_colour
* Allowed_outdoor
* Preferred_food
* Owner_play_time_minutes
* Sleep_time_hours
* Country
* Latitude, Longitude

---

## Steps Performed

### 1. **Handle Missing Values & Remove Duplicates**

* Replaced invalid negative values in `Age_in_years` and `Age_in_months` with `NaN`.
* Filled missing numeric values using **median**.
* Filled missing categorical values using **mode**.
* Removed duplicate rows.
* Resulted in a clean dataset with no missing values.

---

### 2. **Outlier Detection & Removal**

* Applied the **IQR (Interquartile Range) method** to numeric columns:

  * `Weight`
  * `Body_length`
* Removed extreme outliers to reduce skewness and improve clustering quality.
* Dataset shape after this step: **998 × 15**.

---

### 3. **Encoding Categorical Columns**

* Applied **One-Hot Encoding** using `pd.get_dummies(drop_first=True)`.
* Encoded categorical features such as:

  * Breed
  * Gender
  * Fur attributes
  * Country
  * Lifestyle-related attributes
* Converted the dataset into a fully numeric format suitable for clustering.

---

### 4. **Feature Scaling**

* Applied **StandardScaler** to all features.
* Ensured all variables contribute equally to distance-based clustering algorithms (e.g. K-Means).

---

### 5. **Visualizations & Saving Processed Data**

* Boxplots for `Weight` and `Body_length` to confirm outlier removal.
* Histograms for all numeric features after cleaning.
* Correlation heatmap for numeric columns to inspect relationships.
* Saved final clustering-ready dataset to:

```
processed/final_clustering_ready.csv
```

---

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## My Role Description

* Designed and implemented a complete preprocessing pipeline for clustering.
* Cleaned dirty categorical and numeric data.
* Handled missing, invalid, and duplicate values.
* Removed outliers using the IQR method.
* Encoded categorical features and scaled numeric features.
* Generated visualizations to validate preprocessing steps.
* Saved the final clustering-ready dataset for downstream modeling.

---

## References / Sources

* Feature scaling & preprocessing techniques:
  [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)

* How to detect outliers using IQR and Boxplots — MachineLearningPlus:
  [https://www.machinelearningplus.com/machine-learning/how-to-detect-outliers-using-iqr-and-boxplots/](https://www.machinelearningplus.com/machine-learning/how-to-detect-outliers-using-iqr-and-boxplots/)

* General best practices for data cleaning & clustering preparation
_____________________________________________________________________

# Clustering Model Training

## Description
This notebook trains unsupervised clustering models and saves cluster assignments for analysis and visualization.

## Features

### Model Training
- K-Means – Distance-based centroid clustering  
- DBSCAN – Density-based clustering  
- Gaussian Mixture Model (GMM) – Probabilistic mixture model  
- HDBSCAN / Agglomerative – Hierarchical density-based clustering or fallback  

### Parameter Tuning
- Finds optimal parameters for each model using **Silhouette Score**:  
  - K for K-Means  
  - eps for DBSCAN  
  - n_components for GMM  
  - min_cluster_size for HDBSCAN or n_clusters for Agglomerative  

### Model Training & Saving
- Trains each model with optimal parameters  
- Saves trained models using **joblib**  
- Saves cluster labels as CSV for further analysis  
- Saves dataset augmented with cluster assignments  

### Visualization
- Generates scatter plots of clusters for all models  
- Uses **matplotlib** to visualize results  

## Input Files
- `data/processed/final_preprocessed_clustering.csv` – Preprocessed dataset for clustering  

## Output Files

### Models
- `models/kmeans_model.pkl`  
- `models/dbscan_model.pkl`  
- `models/gmm_model.pkl`  
- `models/hdbscan_model.pkl` or `models/agglomerative_model.pkl`  

### Cluster Labels
- `data/processed/kmeans_cluster_labels.csv`  
- `data/processed/dbscan_cluster_labels.csv`  
- `data/processed/gmm_cluster_labels.csv`  
- `data/processed/hdbscan_cluster_labels.csv` or `agglomerative_cluster_labels.csv`  

### Clustered Data
- `data/processed/clustered_data.csv`  
- `data/processed/dbscan_clustered_data.csv`  
- `data/processed/gmm_clustered_data.csv`  
- `data/processed/hdbscan_clustered_data.csv` or `agglomerative_clustered_data.csv`  

## Dependencies
- pandas, numpy  
- scikit-learn (KMeans, DBSCAN, GaussianMixture, AgglomerativeClustering, silhouette_score)  
- hdbscan (optional)  
- joblib  
- matplotlib  

## Notes
- No supervised evaluation metrics (Accuracy, F1) are calculated  
- Silhouette Score is used as an internal quality metric  
- Cluster labels can be used for downstream analysis or supervised learning integration  

## Data Flow Between Notebooks
Clustering Notebook (Training):  
Train Models → Save Models → Save Cluster Labels → Save Clustered Data  

Models and labels can be used for visualization, analysis, or integration with other notebooks  

## Step-by-Step Process
1. Load preprocessed dataset  
2. Train K-Means, DBSCAN, and GMM (and HDBSCAN if available)  
3. Test parameters for each model to maximize Silhouette Score  
4. Save models, labels, and clustered datasets  
5. Visualize clusters using scatter plots  

## Benefits
- Clear separation of clustering workflow  
- Modular and maintainable code: training, parameter tuning, saving, visualization  
- Easy to update or extend with new clustering methods  

## My Role Description
- **Data Preprocessing:** Cleaned and prepared the dataset for clustering  
- **Model Implementation:** Implemented K-Means, DBSCAN, GMM, and HDBSCAN/Agglomerative models  
- **Parameter Tuning:** Tested multiple parameters and selected optimal values using Silhouette Score  
- **Model Saving & Reporting:** Saved trained models, cluster labels, and clustered datasets  
- **Visualization:** Created scatter plots for cluster analysis and comparison  
- **Documentation:** Updated README  

## Evaluation
The clustering results were evaluated using **Silhouette Score** as the primary quality metric.  

### Results Summary
| Model                  | Silhouette Score | Additional Metrics          |
|------------------------|----------------|----------------------------|
| K-Means                | 0.207          | –                          |
| DBSCAN                 | -0.062         | Noise points: high         |
| GMM                    | 0.154          | AIC: -15920460.74, BIC: -15583072.66 |
| HDBSCAN / Agglomerative | 0.175          | Noise points (if HDBSCAN): moderate |

**Observations:**  
- K-Means achieved the highest Silhouette Score, suggesting relatively better cluster separation.  
- DBSCAN resulted in a negative Silhouette Score, indicating clusters were poorly defined.  
- GMM performed moderately, with AIC and BIC values indicating model fit.  
- HDBSCAN/Agglomerative provided intermediate performance, suitable for hierarchical or density-based structure analysis.  

## Comparison
All models were compared based on Silhouette Score and practical clustering behavior.  

| Model                  | Strengths                                           | Limitations                                       |
|------------------------|---------------------------------------------------|--------------------------------------------------|
| K-Means                | Simple, interpretable, best Silhouette Score     | Sensitive to outliers, assumes spherical clusters |
| DBSCAN                 | Detects arbitrary-shaped clusters, handles noise | Poor performance with current eps/min_samples    |
| GMM                    | Probabilistic clustering, soft assignments       | Moderate separation, sensitive to initialization |
| HDBSCAN / Agglomerative | Can detect hierarchical/density structures       | Requires careful parameter tuning, slightly lower score than K-Means |

**Key Takeaways:**  
- K-Means is currently the most effective clustering model for this dataset.  
- DBSCAN needs further parameter optimization to improve clustering quality.  
- GMM provides probabilistic insights but clusters are less distinct.  
- HDBSCAN/Agglomerative is useful for hierarchical analysis and handling density variations.  

## References
- scikit-learn Clustering Documentation: https://scikit-learn.org/stable/modules/clustering.html  
- HDBSCAN Documentation: https://hdbscan.readthedocs.io/en/latest/  
- K-Means Clustering Theory: https://en.wikipedia.org/wiki/K-means_clustering  
- DBSCAN Clustering Theory: https://en.wikipedia.org/wiki/DBSCAN  
- Gaussian Mixture Models: https://scikit-learn.org/stable/modules/mixture.html  
- Silhouette Score Explanation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html  


--------------------------------------------------------------------------------------------------
# Classification Preprocessing

Data Preprocessing for Classification Analysis
This notebook prepares the diabetes dataset for binary classification analysis. Steps include handling missing values (treating zeros as missing), imputation, outlier handling, feature scaling, train/test split, and class balancing using SMOTE.

## Dataset

- **diabetes.csv** (768 rows × 9 columns)  
- Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome (target variable)
- Binary classification task: Predict diabetes (Outcome: 0 = No Diabetes, 1 = Diabetes)

## Steps

1. **Missing Values Handling**

   * Identified zero values in medical features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) as missing values
   * Converted zeros to NaN for these columns
   * Applied median imputation using SimpleImputer

2. **Outlier Handling**

   * Applied IQR (Interquartile Range) method to cap outliers
   * Capped outliers for all numeric features to prevent extreme values from affecting the model

3. **Feature/Target Split**

   * Separated features (X) from target variable (y = Outcome)
   * Features shape: (768, 8)
   * Target shape: (768,)

4. **Feature Scaling**

   * Applied StandardScaler to normalize all features
   * Ensures all features are on the same scale for better model performance

5. **Train/Test Split**

   * Split dataset into training (80%) and testing (20%) sets
   * Used stratified split to maintain class distribution
   * Final shapes:
     * X_train: (614, 8)
     * X_test:  (154, 8)
     * y_train: (614,)
     * y_test:  (154,)

6. **Class Balancing (SMOTE)**

   * Applied SMOTE (Synthetic Minority Oversampling Technique) to balance classes in training data
   * After SMOTE:
     * X_train: (800, 8)
     * y_train: (800,)
   * Balanced class distribution for better model performance

7. **Save Preprocessed Data**

   * Saved final preprocessed datasets:
     * `data/processed/X_train_diabetes.csv`
     * `data/processed/X_test_diabetes.csv`
     * `data/processed/y_train_diabetes.csv`
     * `data/processed/y_test_diabetes.csv`

8. **Visualizations**

   * Glucose distribution histogram
   * Outcome (target) distribution count plot
   * Correlation heatmap for all features

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

## My Role

* Implemented preprocessing pipeline for binary classification: missing value handling, outlier capping, scaling, splitting, and class balancing.
* Generated visualizations to verify preprocessing and understand data distribution.
* Saved intermediate and final datasets for model training.

## References / Sources

* Dataset: diabetes.csv (Pima Indians Diabetes Dataset)
* Feature scaling & preprocessing techniques: [scikit-learn documentation](https://scikit-learn.org/stable/)
* Handling class imbalance using SMOTE: [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
* Outlier detection using IQR: [MachineLearningPlus](https://www.machinelearningplus.com/)

-------------------------------------------------------

# Regression Preprocessing
# FIFA 21 Player Regression Preprocessing

---

## Dataset
(https://www.kaggle.com/datasets/yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring)
**fifa21_raw_data.csv** (18,979 rows × 77 columns)  
Mix of numerical (55) and categorical/object (22) features.

---

## Steps Performed

### 1. Column Cleaning

Removed special symbols (↓) and extra spaces from column names.

Dropped unnecessary columns: photoUrl, playerUrl, LongName.

### 2. Height & Weight Standardization

Converted height from feet'inches" to centimeters.

Converted weight from lbs or kg to kilograms.

### 3. Currency Cleaning

Converted Value, Wage, and Release Clause to numeric values in euros.

Handled M for millions and K for thousands.

### 4. Ratings & Hits Cleaning

Removed stars (★) from columns: W/F, SM, IR.

Converted Hits to integers, standardizing K notation.

### 5. Team & Contract Parsing

Extracted Team, Contract_Start, and Contract_End from Team & Contract.

Identified 714 unique teams

### 6. Player Position Categorization

Extracted Primary_Position and mapped into broader Position_Category (GK, DEF, MID, ATT)

### 7. Loan Status

Created On_Loan binary column from Loan Date End

### 8. Missing Value Handling

Filled numerical columns with median values.

Filled categorical columns with mode or 'Unknown'.

Result: 0 missing values

### 9. Outlier Handling

Capped Height and Weight using IQR method with multiplier 2.5

### 10. Feature Engineering

Age categories: Young, Prime, Experienced, Veteran.

Performance indices: Attack_Score, Defense_Score, Passing_Score, Physical_Score.

Growth metrics: Potential_Growth, Value_per_Rating.

BMI calculation.

Contract metrics: Contract_Length, Years_Remaining.

Star player indicator: Is_Star (OVA >= 85)

### 11. Encoding

Binary encoding: foot_encoded.

Ordinal encoding: AttackRate, DefenseRate from work rates.

One-Hot encoding: Position_Category.

Frequency encoding: Nationality, Team.

Label encoding: Primary_Position.


## Dependencies

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- re  

---

## My Role

- Implemented a complete preprocessing pipeline for regression modeling.
- Handled missing values, duplicates, outliers, categorical encoding, and feature scaling.
- Extracted features and engineered new metrics for performance analysis.

---

## References / Sources
- Dataset: `fifa21_raw_data.csv`  
- Feature scaling & preprocessing techniques: scikit-learn documentation  
- Outlier detection using IQR: MachineLearningPlus ([Link](https://www.machinelearningplus.com/machine-learning/how-to-detect-outliers-using-iqr-and-boxplots/))

------------------------------------------------------------------------------------------------------
# FIFA Player Value Prediction – Regression Models
In this section of the project, we used dataset (FIFA Players Dataset) to predict values of players using 4 regression models. The expected outcome is a comparison between the 4 models to determine which model is best.

## Project Objectives

-Use a preprocessed dataset provided by a teammate

-Train and compare the following regression models:

-*Linear Regression*

-*Ridge Regression*

-*Lasso Regression*

-*Elastic Net Regression*

-*Evaluate models using standard regression metrics*

-*Identify the best-performing model*

## The target variable is 'Value'

## Technologies & Libraries Used

-Python

-Pandas

-NumPy

-Scikit-learn

-Matplotlib / Seaborn

-Jupyter Notebook (.ipynb)

## Machine Learning Workflow

-Load the cleaned CSV dataset

-Define target variable (Value)

-Drop non-feature and leakage columns

-One-hot encode remaining categorical features

-Split data into training and testing sets

-Apply feature scaling using StandardScaler

-Train four regression models

-Evaluate models using:

-MAE (Mean Absolute Error)

-MSE (Mean Squared Error)

-RMSE (Root Mean Squared Error)

-R² Score

-Compare and analyze model performance

## More info on models implemented
Linear Regression -->	Baseline regression model
Ridge Regression -->	L2 regularization
Lasso Regression -->	L1 regularization with feature selection
Elastic Net	Combination of --> L1 & L2 regularization

## Results Summary

-All four models were successfully trained and evaluated.
-Regularized models required feature scaling for optimal performance.
-Elastic Net required increased iterations and stronger regularization to converge properly.
-Final comparison was done using R² Score and RMSE.

# How to run?
1. Find the Jupyter Notebook:
BONUS_regression

2. Open the .ipynb file

3. Run all cells from top to bottom

4. View evaluation metrics and model comparison results

## Notes

-Models were trained inside the notebook environment.

-Trained models were not saved since deployment was not required for this project.

-The focus of this work is on model comparison and performance evaluation.

# SOURCES
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

