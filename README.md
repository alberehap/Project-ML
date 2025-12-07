# Project-ML

Clustering &amp; Classification Data Analysis

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

