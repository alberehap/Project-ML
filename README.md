# Project-ML

Clustering &amp; Classification Data Analysis

---

##  **01_data_overview.ipynb - Data Overview & EDA + Evaluation**

This notebook handles dataset preparation, exploratory data analysis, and model evaluation.

### **Features:**

1. **Dataset Selection & Loading**
   - Loads classification dataset (`dirty_cafe_sales.csv`)
   - Loads clustering dataset (`retail_store_sales.csv`)
   - Reads raw datasets from `data/raw/`

2. **Exploratory Data Analysis (EDA)**
   - Dataset shape overview
   - Missing values analysis and summary
   - Duplicates detection and overview
   - Basic visualizations:
     - Distribution plots (histograms)
     - Scatter plots (Quantity vs Total Spent)
     - Correlation matrix heatmap

3. **Model Evaluation** 
   - **Accuracy Score** - Calculates accuracy for all supervised models
   - **F1 Score** - Computes weighted F1 score for each model
   - **Confusion Matrix** - Creates visualizations for all models
   - **Classification Report** - Generates detailed per-class metrics

### **Input Files:**
- `data/raw/dirty_cafe_sales.csv` - Classification dataset
- `data/raw/retail_store_sales.csv` - Clustering dataset
- `data/processed/X_test_classification.csv` - Test features
- `data/processed/y_test_classification.csv` - Test labels
- `data/processed/y_pred_lr.csv` - Logistic Regression predictions
- `data/processed/y_pred_nb.csv` - Naive Bayes predictions
- `data/processed/y_pred_dt.csv` - Decision Tree predictions
- `data/processed/y_pred_svm.csv` - SVM predictions

### **Output:**
- Evaluation results displayed in notebook
- Confusion matrix visualizations (2x2 grid)
- Classification reports for each model

### **Dependencies:**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (for evaluation metrics: `accuracy_score`, `f1_score`, `confusion_matrix`, `classification_report`)

---

##  **06_supervised_models.ipynb - Supervised Model Training**

This notebook trains supervised learning models and generates predictions.

### **Features:**

1. **Model Training**
   - Trains 4 supervised classification models:
     - **Logistic Regression** - Linear classification model
     - **Naive Bayes** - Probabilistic classifier
     - **Decision Tree** - Tree-based classifier
     - **SVM** - Support Vector Machine

2. **Prediction Generation**
   - Generates predictions on test set for each model
   - Saves predictions to CSV files for evaluation

3. **Model Status Tracking**
   - Simple comparison table showing training status
   - **Note**: This notebook does NOT calculate evaluation metrics

### **Input Files:**
- `data/processed/X_train_classification.csv` - Training features
- `data/processed/X_test_classification.csv` - Test features
- `data/processed/y_train_classification.csv` - Training labels
- `data/processed/y_test_classification.csv` - Test labels

### **Output Files:**
- `data/processed/y_pred_lr.csv` - Logistic Regression predictions
- `data/processed/y_pred_nb.csv` - Naive Bayes predictions
- `data/processed/y_pred_dt.csv` - Decision Tree predictions
- `data/processed/y_pred_svm.csv` - SVM predictions

### **Dependencies:**
- `pandas`, `numpy`
- `scikit-learn` (for models: `LogisticRegression`, `GaussianNB`, `DecisionTreeClassifier`, `SVC`)

### **Important Notes:**
-  **No evaluation metrics** are calculated in this notebook
- All evaluation (Accuracy, F1, Confusion Matrix) is handled by `01_data_overview.ipynb`
- Predictions are saved as CSV files to be loaded and evaluated by the evaluation notebook

---

##  **Data Flow Between Notebooks**

```
06_supervised_models.ipynb (Training):
    Train Models → Generate Predictions → Save to CSV
         ↓
    data/processed/y_pred_*.csv
         ↓
01_data_overview.ipynb (Evaluation):
    Load Predictions → Calculate Metrics → Visualize Results
```

### **Step-by-Step Process:**

1. **Run `06_supervised_models.ipynb`:**
   - Trains all 4 models on training data
   - Generates predictions on test set
   - Saves predictions to CSV files in `data/processed/`

2. **Run evaluation section in `01_data_overview.ipynb`:**
   - Loads predictions from CSV files
   - Calculates Accuracy and F1 Score for each model
   - Creates Confusion Matrix visualizations
   - Generates detailed Classification Reports

---

##  **Key Features - Updated Structure**

### **Separation of Concerns:**

-  **`01_data_overview.ipynb`** handles **Evaluation** (Accuracy, F1, Confusion Matrix)
-  **`06_supervised_models.ipynb`** focuses only on **Model Training** (no evaluation metrics)
-  Clear separation: Training vs. Evaluation

### **Benefits:**

- Better code organization and modularity
- Evaluation notebook has complete overview of data and results
- Training notebook focuses solely on model implementation
- Easier maintenance and code review

---


# Clustering &amp; Classification Data Analysis

# Clustering Preprocessing

Data Preprocessing for Clustering Analysis  
This notebook prepares the retail store sales dataset for clustering analysis. Steps include handling missing values, removing outliers, encoding categorical data, and feature scaling. Optional visualizations are provided for data inspection.

## Dataset

- [retail_store_sales.csv](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) (12575 rows × 11 columns)  
- Columns: Transaction ID, Customer ID, Category, Item, Price Per Unit, Quantity, Total Spent, Payment Method, Location, Transaction Date, Discount Applied  

## Steps Performed  

1. **Handle Missing Values & Remove Duplicates**  
   - Removed rows with missing `Item`.  
   - Filled missing numeric values with median.  
   - Filled missing categorical values with mode.  
   - Removed duplicate rows.  
   - Saved cleaned dataset: `cleaned1_clustering.csv`.  

2. **Outlier Detection & Removal**  
   - Applied IQR method to numeric columns: Price Per Unit, Quantity, Total Spent.  
   - Removed outliers (56 rows from Total Spent).  
   - Saved dataset: `cleaned2_no_outliers.csv`.  

3. **Encoding Categorical Columns**  
   - Applied one-hot encoding for categorical columns: Category, Item, Payment Method, Location, Discount Applied.  
   - Saved encoded dataset: `encoded_clustering.csv`.  

4. **Feature Scaling**  
   - Applied StandardScaler to numeric columns (Price Per Unit, Quantity, Total Spent).  
   - Saved final preprocessed dataset: `final_preprocessed_clustering.csv`.  

5. **Visualizations**  
   - Histogram for `Total Spent` after outlier removal.  
   - Correlation heatmap for numeric columns after scaling.  

## Dependencies

- pandas, numpy, matplotlib, seaborn, scikit-learn  

## My Role Description

- Implemented the data preprocessing pipeline for clustering analysis.  
- Handled missing values, outliers, categorical encoding, and feature scaling.  
- Generated optional visualizations to verify preprocessing.  
- Saved intermediate and final processed datasets and pushed files to GitHub.

### References / Sources

- Dataset: [retail_store_sales.csv](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning)  
- Feature scaling & preprocessing techniques: [scikit-learn documentation](https://scikit-learn.org/stable/)  
- How to detect outliers using IQR and Boxplots? — MachineLearningPlus  
  (https://www.machinelearningplus.com/machine-learning/how-to-detect-outliers-using-iqr-and-boxplots/)

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
This notebook prepares the cafe sales dataset for classification analysis. Steps include handling missing values, encoding categorical data, balancing classes, and splitting the dataset. Optional visualizations are included.

## Dataset

- [cafe_sales.csv](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training) (10000 rows × 8 columns)  
- Columns: Transaction ID, Customer ID, Payment Method, Location, Quantity, Total Spent, Item Categories (Coffee, Tea, Juice, etc.), Transaction Date

## Steps

1. **Missing Values & Duplicates**

   * Removed rows with missing essential values.
   * Filled missing numeric values with median, categorical with mode.
   * Removed duplicate rows.
   * Saved cleaned dataset.

2. **Encoding Categorical Data**

   * One-hot encoding for categorical columns: Item Categories, Location.
   * Converted target `Payment Method` to numeric labels.
   * Saved encoded dataset.

3. **Feature Scaling**

   * Applied StandardScaler to numeric columns: Quantity, Total Spent.
   * Saved final preprocessed dataset.

4. **Train/Test Split & Class Balancing**

   * Split dataset into training (80%) and testing (20%) sets.
   * Applied SMOTE to balance classes in training data.
   * Final shapes:

     * X_train: (8960, 23)
     * X_test:  (1447, 23)
     * y_train: (8960,)
     * y_test:  (1447,)

5. **Visualizations**

   * Histogram for `Total Spent`.
   * Scatter plot of `Quantity` vs `Total Spent`.
   * Correlation heatmap for numeric columns.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

## My Role

* Implemented preprocessing pipeline for classification: missing values, encoding, scaling, splitting, and class balancing.
* Generated optional visualizations to verify preprocessing.
* Saved intermediate and final datasets and pushed files to GitHub.

## References / Sources

* Dataset: [cafe_sales.csv](https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training)
* Feature scaling & preprocessing techniques: [scikit-learn documentation](https://scikit-learn.org/stable/)
* Handling class imbalance using SMOTE: [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
* Outlier detection & preprocessing tips: [MachineLearningPlus](https://www.machinelearningplus.com/)

-------------------------------------------------------

# Regression Preprocessing
# FIFA 21 Player Regression Preprocessing

This repository contains a complete data preprocessing pipeline for the FIFA 21 player dataset, preparing it for regression modeling.

---

## Dataset

**fifa21_raw_data.csv** (18,979 rows × 77 columns)  

**Columns include:** Player ID, Name, Nationality, Positions, Age, OVA, POT, Team & Contract, Height, Weight, Foot, Ratings, Performance Stats, Work Rates, Contract Info, Hits, Value, Wage, Release Clause, etc.

---

## Steps Performed

### 1. Handle Missing Values & Remove Duplicates
- Filled missing numeric values with median.
- Filled missing categorical values with mode or `'Unknown'`.
- Removed duplicate rows.

### 2. Column Cleaning
- Removed special characters (`↓`) from column names.
- Stripped extra whitespace from column names.
- Dropped unnecessary columns: `photoUrl`, `playerUrl`, `LongName`.

### 3. Height & Weight Cleaning
- Converted `Height` from feet/inches to centimeters.
- Converted `Weight` from lbs/kg to kilograms.

### 4. Currency Columns Cleaning
- Converted `Value`, `Wage`, and `Release Clause` to numeric (from strings like €1.2M, €500K).

### 5. Rating Columns Cleaning
- Removed stars from `W/F`, `SM`, `IR` and converted to integers.

### 6. Hits Column Cleaning
- Cleaned `Hits` column and converted values like '372K' to integers.

### 7. Team & Contract Extraction
- Extracted `Team`, `Contract_Start`, and `Contract_End` from the `Team & Contract` column.
- Extracted 714 unique teams.

### 8. Player Position Categorization
- Extracted primary position from `Positions`.
- Mapped positions to general categories: GK, DEF, MID, ATT.

### 9. Loan Status
- Created a binary column `On_Loan` indicating whether a player is on loan (1) or not (0).
- Total players on loan: 1013.

### 10. Drop Unnecessary Columns
- Dropped `Team & Contract`, `Positions`, `Loan Date End`.

### 11. Handling Missing Values
- Ensured no remaining missing values after filling numeric and categorical columns.

### 12. Outlier Handling
- Capped outliers in `Height` and `Weight` using the IQR method.

### 13. Feature Engineering
- **Age Category:** Young, Prime, Experienced, Veteran.
- **Performance Scores:** `Attack_Score`, `Defense_Score`, `Passing_Score`, `Physical_Score`.
- **Growth Metrics:** `Potential_Growth = POT - OVA`, `Value_per_Rating`.
- **BMI Calculation:** `Weight / (Height/100)^2`.
- **Contract Metrics:** `Contract_Length`, `Years_Remaining`.
- **Star Player Flag:** `Is_Star` for OVA >= 85.

### 14. Encoding
- **Binary Encoding:** `foot_encoded` (Right=1, Left=0).
- **Ordinal Encoding:** AttackRate (`A/W`) and DefenseRate (`D/W`) as Low=1, Medium=2, High=3.
- **One-Hot Encoding:** `Position_Category`.
- **Frequency Encoding:** `Nationality_freq`, `Team_freq`.
- **Label Encoding:** `Primary_Position`.

---

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

