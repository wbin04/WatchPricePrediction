# Luxury Watch Price Prediction

This project aims to predict the price of luxury watches based on features such as brand, model, case material, strap material, movement type, and other factors. Below is an overview of the data processing workflow and project structure.

## Data Processing Workflow

### 1. Data Preprocessing
The `Data Preprocessing.ipynb` notebook handles data cleaning and initial preparation. Key steps include:
- **Data Loading**: Data is loaded from `datasets/luxury_watches_preprocessed.csv`.
- **Handling Missing Values**:
  - Columns with missing values, such as `Complications` and `Power Reserve`, are filled with default values or medians.
- **Normalization**:
  - Case material, strap material, dial color, and glass material are normalized to ensure consistency.
- **Feature Transformation**:
  - `Water Resistance` is converted to an integer value.
  - `Price` is cleaned and converted to a numeric format.
  - `Power Reserve` is normalized to hours.
- **Complexity Scoring**:
  - `Complication_Score` is calculated based on the types of watch complications.

### 2. Exploratory Data Analysis (EDA)
EDA is performed to understand the data and visualize key insights. Key steps include:
- **Distribution Analysis**:
  - Boxplots are used to identify outliers in numerical columns.
- **Correlation Between Numerical Columns**:
  - Scatterplots are used to show relationships between pairs of numerical columns.
  - Heatmaps are used to display the correlation matrix of numerical columns.
  - Pairplots are used to visualize distributions and relationships between numerical columns.
- **Correlation Between Categorical Columns and Price**:
  - Boxplots are used to show the relationship between categorical columns and the `Price` column.

### 3. Feature Engineering
Feature engineering is performed to enhance the data for modeling. Key steps include:

- **Outlier Clipping**:
  - Outliers in numerical columns such as `Water Resistance`, `Case Diameter`, `Case Thickness`, `Band Width`, `Power Reserve`, `Complication_Score`, and `Price` are clipped using the interquartile range (IQR) method.

- **Feature Creation**:
  - `Brand_Tier` and `Model_Tier` are created based on predefined categories of brands and models.
  - `Luxury_Index` is calculated as a composite score using `Case Material`, `Strap Material`, `Crystal Material`, and `Complication_Score`.
  - `Material_Match` indicates whether the case material matches the strap material.
  - `Movement_Complexity` is derived from the type of movement (e.g., automatic or manual).
  - `Water_Tier` categorizes water resistance into `Basic`, `Standard`, and `Professional` tiers.
  - `Case_Proportion` is calculated as the ratio of `Case Diameter` to `Case Thickness`.
  - `Case_Size_Category` categorizes watches based on their case diameter.
  - `Dial_Score` assigns a score to dial colors based on their perceived luxury.
  - `Has_Complication` is a binary feature indicating the presence of complications.

- **Encoding Categorical Variables**:
  - Label encoding and target encoding are applied to `Brand_Tier` and `Model_Tier` for use in machine learning models.

- **Interaction Features**:
  - `Brand_Case` combines `Brand_Tier` and `Case_Proportion`.
  - `Brand_Crystal` combines `Brand_Tier` and the presence of sapphire crystal.
  - `Material_Crystal_Movement` combines `Material_Match`, the presence of sapphire crystal, and `Movement_Complexity`.

- **Feature Reordering**:
  - Columns are reordered to ensure logical grouping and ease of analysis.

### 4. Regression Models
Various regression models are implemented to predict watch prices. Below are detailed explanations of the models used:

- **CatBoost Regression**:
  - CatBoost is a gradient boosting algorithm specifically designed to handle categorical features efficiently. It eliminates the need for extensive preprocessing, such as one-hot encoding, by natively supporting categorical variables.
  - The `CatBoost Regression.ipynb` notebook demonstrates how this model is trained to predict watch prices. The training process involves iterative learning, where the model optimizes its predictions by minimizing the loss function over multiple iterations.
  - CatBoost also provides detailed logs during training, which help in monitoring the reduction of error and understanding the model's convergence behavior.

- **XGBoost Regression**:
  - XGBoost (Extreme Gradient Boosting) is a powerful and flexible machine learning algorithm based on decision trees. It is known for its speed and performance, especially in structured/tabular data.
  - The `XGBoost Regression.ipynb` notebook focuses on training an XGBoost model with hyperparameter tuning. Techniques such as grid search are used to find the optimal values for parameters like `n_estimators`, `max_depth`, and `learning_rate`.
  - Additionally, the notebook includes feature importance analysis using permutation importance, which helps identify the most influential features in predicting watch prices.

- **Random Forest Regression**:
  - Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data, and the final prediction is obtained by averaging the outputs of all trees.
  - The `RandomForest Regression.ipynb` notebook illustrates the training process for this model. Random Forest is particularly robust to noise and overfitting due to its averaging mechanism, making it a reliable choice for regression tasks.
  - This model is also easy to interpret, as it provides insights into feature importance and the contribution of individual features to the predictions.

## Evaluation

To assess the performance of the regression models, the following evaluation metrics are used:

- **R² (Coefficient of Determination)**  
  Measures the proportion of variance in the dependent variable that is predictable from the independent variables.  
  Ranges from 0 to 1, where a higher value indicates better model performance.  
  Formula:

  $$
  R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
  $$

- **RMSE (Root Mean Squared Error)**  
  Represents the square root of the average squared differences between predicted and actual values.  
  Provides a measure of the model's prediction error in the same units as the target variable.  
  Formula:

  $$
  RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
  $$

- **MAE (Mean Absolute Error)**  
  Represents the average of the absolute differences between predicted and actual values.  
  Provides a straightforward measure of prediction accuracy.  
  Formula:

  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$

These metrics are calculated for each regression model to compare their performance and select the best model for predicting luxury watch prices.


## Project Structure
- `datasets/`
  - Contains raw and preprocessed datasets.
- `Data Preprocessing.ipynb`
  - Notebook for data cleaning and preparation.
- `EDA.ipynb`
  - Notebook for data analysis and visualization.
- `Feature Engineering.ipynb`
  - Notebook for feature creation and transformation.
- `Regression Models`
  - Contains scripts and notebooks for training and evaluating regression models.

## How to Run the Project
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Execute notebooks in order:
   - `Data Preprocessing.ipynb`
   - `EDA.ipynb`
   - `Feature Engineering.ipynb`
   - Regression notebooks (`CatBoost`, `XGBoost`, `RandomForest`).

## Conclusion
This project provides a comprehensive workflow for predicting luxury watch prices, from data preprocessing to model evaluation. The structured approach ensures reusability and scalability for future improvements.