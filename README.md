# Project-ML

Clustering &amp; Classification Data Analysis

---

## 📁 **01_data_overview.ipynb - Data Overview & EDA + Evaluation**

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

3. **Model Evaluation** ⭐
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

## 📁 **06_supervised_models.ipynb - Supervised Model Training**

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
- ⚠️ **No evaluation metrics** are calculated in this notebook
- All evaluation (Accuracy, F1, Confusion Matrix) is handled by `01_data_overview.ipynb`
- Predictions are saved as CSV files to be loaded and evaluated by the evaluation notebook

---

## 🔄 **Data Flow Between Notebooks**

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

## 🎯 **Key Features - Updated Structure**

### **Separation of Concerns:**

- ✅ **`01_data_overview.ipynb`** handles **Evaluation** (Accuracy, F1, Confusion Matrix)
- ✅ **`06_supervised_models.ipynb`** focuses only on **Model Training** (no evaluation metrics)
- ✅ Clear separation: Training vs. Evaluation

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

