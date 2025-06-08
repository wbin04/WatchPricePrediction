# 🕰️ Watch Price Prediction Project

## Tổng quan dự án

Dự án này xây dựng một hệ thống hoàn chỉnh bao gồm thu thập dữ liệu, tiền xử lý, phân tích khám phá, kỹ thuật đặc trưng đến xây dựng ETL pipeline và mô hình dự đoán giá đồng hồ cao cấp sử dụng Machine Learning.

## 📁 Cấu trúc dự án

```
WatchPricePrediction/
├── 🕷️ Data Collection
│   └── Crawler.py                                   # Web scraping với Scrapy
│
├── 🔄 ETL Pipeline
│   └── ETL_Pipeline.py                              # Complete ETL process
│
├── 📊 Data Analysis & Visualization
│   ├── Descriptive_Statistics_Visualization.ipynb   # Thống kê mô tả & trực quan hóa
│   ├── ML_Pipeline_1_Preprocessing.ipynb            # Tiền xử lý dữ liệu
│   ├── ML_Pipeline_2_EDA.ipynb                      # Phân tích khai phá dữ liệu  
│   └── ML_Pipeline_3_Feature_Engineering.ipynb      # Kỹ thuật đặc trưng
│
├── 🤖 Machine Learning Models
│   ├── ML_Pipeline_4_CatBoost.ipynb                 # Mô hình CatBoost
│   ├── ML_Pipeline_4_XGBoost.ipynb                  # Mô hình XGBoost
│   └── ML_Pipeline_4_LightGBM.ipynb                 # Mô hình LightGBM
│
├── 📂 Data Storage
│   ├── data_lake/                                   # Data Lake (Parquet format)
│   │   └── watch_dl.parquet
│   ├── data_warehouse/                              # Data Warehouse (SQLite)
│   │   └── watch_dwh.db
│   ├── datasets_etl/                                # ETL processed data
│   │   ├── data_raw.csv
│   │   └── data_transformed.csv
│   └── datasets_ml/                                 # ML processed data
│       ├── data_raw.csv
│       ├── data_preprocessed.csv
│       └── data_featured.csv
│
└── 📋 Documentation
    └── README.md                                    # Project documentation
```

## 🚀 Quy trình thực hiện

### 1. 🕷️ Thu thập dữ liệu (Data Collection)

**File:** [`Crawler.py`](Crawler.py)

#### Mô tả:
- Sử dụng Scrapy framework để thu thập dữ liệu từ website: [watchbase.com](https://watchbase.com)
- Thu thập thông tin chi tiết về đồng hồ từ 10 thương hiệu nổi tiếng
- Spider class có thể tùy chỉnh brands và số lượng models tối đa

#### Các thương hiệu được crawl:
- Rolex, Omega, Tag Heuer, Tudor, Longines
- IWC, Breitling, Cartier, Panerai, Patek Philippe

#### Dữ liệu thu thập:
- **Thông tin cơ bản:** Brand, Family, Reference, Name, Movement
- **Thông tin vỏ:** Case Material, Glass, Case Back, Case Shape, Case Diameter
- **Thông số kỹ thuật:** Water Resistance, Lug Width
- **Thông tin mặt số:** Dial Color, Dial Finish, Dial Indexes, Dial Hands
- **Thông tin sản xuất:** Produced, Limited
- **Giá cả:** Price (từ price chart API)

#### Cách chạy:
```bash
python Crawler.py
```

#### Kết quả:
- File output: [`datasets_etl/data_raw.csv`](datasets_etl/data_raw.csv) và [`datasets_ml/data_raw.csv`](datasets_ml/data_raw.csv)
- Số lượng: ~6,300 mẫu dữ liệu

---

### 2. 🔄 ETL Pipeline

**File:** [`ETL_Pipeline.py`](ETL_Pipeline.py)

#### Mô tả:
ETL Pipeline hoàn chỉnh thực hiện Extract, Transform, Load data với các thành phần:

#### Các bước ETL:

##### Extract:
- Đọc dữ liệu raw từ CSV
- Lưu backup vào Data Lake (Parquet format)

##### Transform:
- **Data Preprocessing:** Xử lý missing values, chuẩn hóa dữ liệu
- **Feature Engineering:** Tạo features mới, grouping categorical variables

##### Load:
- Lưu vào Data Warehouse (SQLite database)
- Export processed data cho ML pipeline

#### Cách chạy:
```python
python ETL_Pipeline.py
```

#### Kết quả:
- **Data Lake:** [`data_lake/watch_dl.parquet`](data_lake/watch_dl.parquet)
- **Data Warehouse:** [`data_warehouse/watch_dwh.db`](data_warehouse/watch_dwh.db)
- **Processed Data:** [`datasets_etl/data_transformed.csv`](datasets_etl/data_transformed.csv)

---

### 3. 📊 Thống kê mô tả & Trực quan hóa

**File:** [`Descriptive_Statistics_Visualization.ipynb`](Descriptive_Statistics_Visualization.ipynb)

#### Mục đích:
- Hiểu tổng quan về dataset
- Phát hiện missing values, outliers
- Thống kê mô tả cơ bản với visualizations

#### Nội dung chính:
- Kiểm tra missing values và data types
- Thống kê mô tả cho biến số
- Phân bố của biến phân loại
- Tạo visualizations đa dạng (histograms, boxplots, heatmaps)

---

### 4. 🧹 Tiền xử lý dữ liệu (Data Preprocessing)

**File:** [`ML_Pipeline_1_Preprocessing.ipynb`](ML_Pipeline_1_Preprocessing.ipynb)

#### Các bước xử lý:

##### 4.1 Xử lý missing values:
- **Case Material:** Thay thế NaN bằng "Stainless Steel" (phổ biến nhất)
- **Water Resistance:** Chuyển đổi format và điền bằng mode
- **Dial Indexes, Dial Hands:** Điền bằng mode
- **Glass, Case Back, Case Shape:** Điền bằng mode
- **Case Diameter:** Chuyển đổi format và điền bằng mode
- **Dial Color:** Điền bằng mode

##### 4.2 Loại bỏ columns không cần thiết:
- `Produced`: Quá nhiều missing values
- `Lug Width`: Quá nhiều missing values  
- `Dial Finish`: Quá nhiều missing values
- `Reference`, `Name`: Không cần thiết cho modeling

##### 4.3 Chuẩn hóa dữ liệu:
- Chuyển đổi `Water Resistance` từ "30 m" → 30.0
- Chuyển đổi `Case Diameter` từ "40 mm" → 40.0
- Chuyển đổi `Price` thành float
- Rename `Family` → `Model`

##### 4.4 Xử lý biến Limited:
- Tách lấy phần đầu tiên từ chuỗi phức tạp

#### Output:
- File: `datasets_ml/data_preprocessed.csv`
- Dataset sạch, sẵn sàng cho EDA

---

### 5. 🔍 Phân tích khai phá dữ liệu (Exploratory Data Analysis - EDA)

**File:** [`ML_Pipeline_2_EDA.ipynb`](ML_Pipeline_2_EDA.ipynb)

#### Mục đích:
- Hiểu sâu về phân bố dữ liệu
- Khai phá mối quan hệ giữa các biến
- Phát hiện patterns và insights

#### Phân tích chính:

##### 5.1 Phân tích biến số:
- **Case Diameter:** Phân bố lệch phải, tập trung 35-45mm
- **Water Resistance:** Phân bố đa modal, có clusters
- **Price:** Phân bố lệch phải mạnh, nhiều outliers

##### 5.2 Phân tích biến phân loại:
- **Brand:** Phân bố không đều, Omega và Rolex chiếm ưu thế
- **Model:** Đa dạng, mỗi brand có nhiều model
- **Case Material:** Stainless Steel phổ biến nhất
- **Dial Color:** Black và White/Silver chiếm ưu thế

##### 5.3 Mối quan hệ giữa các biến:
- Correlation matrix cho biến số
- Cross-tabulation cho biến phân loại
- Price distribution theo từng nhóm

#### Insights quan trọng:
- Brand là yếu tố quan trọng nhất ảnh hưởng đến giá
- Case Diameter có correlation với Price
- Limited edition có giá cao hơn
- Case Material ảnh hưởng lớn đến Price

---

### 6. ⚙️ Kỹ thuật đặc trưng (Feature Engineering)

**File:** [`ML_Pipeline_3_Feature_Engineering.ipynb`](ML_Pipeline_3_Feature_Engineering.ipynb)

#### Mục đích:
- Tạo features mới từ dữ liệu gốc
- Xử lý outliers
- Chuẩn bị dữ liệu cho modeling

#### Các kỹ thuật áp dụng:

##### 6.1 Outlier Treatment:
- **Case Diameter:** Loại bỏ values > 60mm (pocket watches, errors)
- Sử dụng boxplot và histogram để detect outliers

##### 6.2 Feature Grouping:

**Case Material Grouping:**
```python
case_material_mapping = {
    'Steel': ['Stainless Steel', 'Steel'],
    'Gold Variants': ['Yellow Gold', 'Rose Gold', 'Red Gold', 'White Gold', 'Gold'],
    'Platinum': ['Platinum'],
    'Titanium': ['Titanium'],
    'Mixed/Other': ['Bronze', 'Ceramic', 'Carbon Fiber', 'Others']
}
```

**Case Diameter Grouping:**
```python
diameter_bins = [0, 35, 40, 45, 50, float('inf')]
diameter_labels = ['S', 'M', 'L', 'XL', 'XXL']
```

**Water Resistance Level:**
```python
resistance_bins = [0, 30, 100, 200, 500, float('inf')]
resistance_labels = ['Low', 'Basic', 'Standard', 'Professional', 'Extreme']
```

**Dial Color Grouping:**
```python
color_mapping = {
    'Black': ['Black'],
    'White/Silver': ['White', 'Silver'],
    'Blue': ['Blue'],
    'Colorful': ['Green', 'Brown', 'Red', 'Gold', 'Orange'],
    'Grey/Brown': ['Grey', 'Anthracite']
}
```

##### 6.3 Target Variable Transformation:
- **Log Transformation:** `LogPrice = log(Price)` 
- Giảm skewness của Price distribution
- Cải thiện model performance

##### 6.4 Text Processing:
- Chuyển brand names về lowercase
- Standardize categorical values

#### Output:
- File: `datasets_ml/data_featured.csv`
- Features engineered, ready for modeling

---

### 7. 🤖 Machine Learning Models


#### Train/Validation/Test Split:
- **Training:** 70% 
- **Validation:** 20%
- **Test:** 10%


#### 7.1 📈 CatBoost Regression

**File:** [`ML_Pipeline_4_CatBoost.ipynb`](ML_Pipeline_4_CatBoost.ipynb)

##### CatBoost:
- **Native categorical support:** Xử lý trực tiếp categorical features
- **Robust to overfitting:** Built-in regularization
- **High performance:** State-of-the-art gradient boosting
- **No need for encoding:** Tự động xử lý categorical variables

##### Data Preparation:
```python
# Categorical columns
cat_columns = ['Brand', 'Model', 'Limited', 'CaseMaterialGrouped', 
               'Glass', 'Case Shape', 'CaseDiameterGrouped', 
               'WaterResistanceLevel', 'DialColorGrouped', 
               'DialHandsGrouped', 'Dial Indexes']

# Convert to category type
df[cat_columns] = df[cat_columns].astype('category')

# Target variable
X = df.drop(columns=['Url', 'LogPrice'])
y = df['LogPrice']
```

##### Model Configuration:
```python
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_columns,
    random_state=42,
    verbose=100
)
```

##### Performance Results:
- **R² Score:** 0.86
- **RMSE:** 0.40
- **MAE:** 0.29

---

#### 7.2 🚀 XGBoost Regression

**File:** [`ML_Pipeline_4_XGBoost.ipynb`](ML_Pipeline_4_XGBoost.ipynb)

##### XGBoost:
- **High performance:** Excellent for structured data
- **Feature importance:** Built-in feature importance
- **Regularization:** L1/L2 regularization
- **Parallel processing:** Fast training

##### Data Preparation:
```python
# Target Encoding for categorical variables
from category_encoders import TargetEncoder

cat_columns = ['Brand', 'Model', 'Limited', 'CaseMaterialGrouped', 
               'Glass', 'Case Shape', 'CaseDiameterGrouped', 
               'WaterResistanceLevel', 'DialColorGrouped', 
               'DialHandsGrouped', 'Dial Indexes']

encoder = TargetEncoder(cols=cat_columns)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_val_enc = encoder.transform(X_val)
X_test_enc = encoder.transform(X_test)
```

##### Model Configuration:
```python
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='rmse'
)
```

##### Training với Early Stopping:
```python
model.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],
    early_stopping_rounds=50,
    verbose=100
)
```

##### Performance Results:
- **R² Score:** 0.87
- **RMSE:** 0.39
- **MAE:** 0.27

---

#### 7.3 ⚡ LightGBM Regression 

**File:** [`ML_Pipeline_4_LightGBM.ipynb`](ML_Pipeline_4_LightGBM.ipynb)

##### LightGBM:
- **Light Gradient Boosting Machine:** Tối ưu hóa tốc độ và bộ nhớ
- **High performance:** Nhanh hơn XGBoost và CatBoost
- **Memory efficient:** Sử dụng ít bộ nhớ hơn
- **Feature importance:** Built-in feature importance analysis
- **GPU support:** Hỗ trợ training trên GPU

##### Data Preparation:
```python
# Target Encoding for categorical variables
cat_columns = ['Brand', 'Model', 'Limited', 'CaseMaterialGrouped', 
               'Glass', 'Case Shape', 'CaseDiameterGrouped', 
               'WaterResistanceLevel', 'DialColorGrouped', 
               'DialHandsGrouped', 'Dial Indexes']

encoder = TargetEncoder(cols=cat_columns)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_val_enc = encoder.transform(X_val)
X_test_enc = encoder.transform(X_test)
```

##### Model Configuration:
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val]
)
```

##### Performance Results:
- **R² Score:** 0.87
- **RMSE:** 0.39
- **MAE:** 0.27

##### Feature Importance Visualization:
```python
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
```

---

### 8. 📊 So sánh Models

#### Performance Comparison:

| Model | R² Score | RMSE | MAE | Ưu điểm | Nhược điểm |
|-------|----------|------|-----|---------|------------|
| **CatBoost** | 0.86 | 0.40 | 0.29 | - Native categorical support<br>- Robust overfitting prevention<br>- No encoding needed | - Slower training<br>- More memory usage |
| **XGBoost** | ~0.87 | ~0.39 | ~0.27 | - Fast training<br>- Excellent feature importance<br>- Wide adoption | - Requires encoding<br>- More hyperparameter tuning |
| **LightGBM** | 0.87 | 0.39 | 0.27 | - Fastest training speed<br>- Memory efficient<br>- Good performance | - Requires encoding<br>- Can overfit with small datasets |

#### Feature Importance Insights:
1. **Model:** Yếu tố quan trọng nhất (30-40% importance)
2. **Brand:** Quan trọng thứ hai (15-20% importance)  
3. **Case Material:** Ảnh hưởng lớn đến giá (10-15% importance)
4. **Case Diameter:** Kích thước quan trọng (8-12% importance)
5. **Limited Edition:** Premium factor (5-8% importance)

---

## 🛠️ Cài đặt và Chạy dự án

### Requirements:

Tất cả dependencies được định nghĩa trong file [`requirements.txt`](requirements.txt).

#### Cài đặt nhanh:
```powershell
# Clone project và cài đặt dependencies
pip install -r requirements.txt
```

#### Hoặc cài đặt thủ công:
```powershell
# Core data processing
pip install pandas>=1.5.0 numpy>=1.21.0

# Data visualization  
pip install matplotlib>=3.5.0 seaborn>=0.11.0

# Web scraping
pip install scrapy>=2.6.0 requests>=2.28.0

# Machine learning
pip install scikit-learn>=1.1.0
pip install catboost>=1.2.0 xgboost>=1.6.0 lightgbm>=3.3.0
pip install category-encoders>=2.5.0

# Database & ETL
pip install sqlalchemy>=1.4.0

# Jupyter notebook
pip install jupyter>=1.0.0 ipykernel>=6.15.0

# Additional utilities
pip install pyarrow>=9.0.0 tqdm>=4.64.0
```

#### Kiểm tra cài đặt:
```powershell
# Verify installations
python -c "import pandas, numpy, matplotlib, seaborn, scrapy, sklearn, catboost, xgboost, lightgbm, category_encoders, sqlalchemy, pyarrow; print('All packages installed successfully!')"
```

### Chạy toàn bộ pipeline:

#### 1. Thu thập dữ liệu:
```powershell
# Run spider
python Crawler.py
```

#### 2. ETL Pipeline:
```powershell
# Run complete ETL process
python ETL_Pipeline.py
```

#### 3. Data Analysis & Preprocessing:
```powershell
# Jupyter notebooks
jupyter notebook "Descriptive_Statistics_Visualization.ipynb"
jupyter notebook "ML_Pipeline_1_Preprocessing.ipynb"
jupyter notebook "ML_Pipeline_2_EDA.ipynb"
jupyter notebook "ML_Pipeline_3_Feature_Engineering.ipynb"
```

#### 4. Machine Learning Models:
```powershell
# Train models
jupyter notebook "ML_Pipeline_4_CatBoost.ipynb"
jupyter notebook "ML_Pipeline_4_XGBoost.ipynb"
jupyter notebook "ML_Pipeline_4_LightGBM.ipynb"
```

---

## 📈 Kết quả và Insights

### Architecture Insights:

#### Data Lake & Data Warehouse:
- **Data Lake:** Lưu trữ raw data dạng Parquet (columnar format)
- **Data Warehouse:** Structured data trong SQLite cho analytics
- **ETL Pipeline:** Automated transformation process

#### ML Pipeline Design:
- **Modular approach:** Tách biệt preprocessing, EDA, feature engineering
- **Reproducibility:** Consistent data flow từ raw → processed → featured
- **Model comparison:** Multiple algorithms với same data preparation

### Business Insights:

1. **Brand Premium:** Patek Philippe và Rolex có giá cao nhất
2. **Material Impact:** Platinum > Gold variants > Steel về giá trị
3. **Size Matters:** Đồng hồ L/XL có giá cao hơn
4. **Limited Edition Premium:** Giá trị tăng 20-50%
5. **Water Resistance:** Diving watches có giá trị cao

### Technical Insights:

1. **Log Transformation:** Cải thiện model performance đáng kể
2. **Feature Engineering:** Grouping categorical variables hiệu quả
3. **Target Encoding:** Hiệu quả cho XGBoost và LightGBM với high cardinality categories
4. **ETL Design:** Scalable architecture cho future data updates

---

## 🔮 Hướng phát triển

### Near-term:
- Hyperparameter tuning chi tiết hơn với GridSearchCV/Optuna
- Cross-validation với time-series split
- Ensemble methods (CatBoost + XGBoost + LightGBM)
- SHAP analysis cho model interpretability
- Real-time ETL pipeline với Apache Airflow

### Long-term:
- Real-time price tracking system
- Deep Learning models (Neural Networks, Transformers)
- Computer Vision cho watch image analysis
- Web application deployment với Flask/FastAPI
- API service cho price prediction
- Cloud deployment (AWS/GCP/Azure)
- Microservices architecture

### Data Engineering:
- Apache Kafka cho real-time data streaming
- Apache Spark cho big data processing
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline với GitHub Actions

---

## 📄 Kiến trúc hệ thống

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Web Scraper │───>│ Data Lake   │───>│ ETL Pipeline│
│ (Scrapy)    │    │ (Parquet)   │    │ (Python)    │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ML Models   │<───│   Analytics │<───│ Data WH     │
│ (Ensemble)  │    │ (Jupyter)   │    │ (SQLite)    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 📄 License

This project is for educational purposes only. Data is collected from public sources for academic research.

---

**Lưu ý:** Dự án này chỉ mang tính chất học thuật và nghiên cứu. Kết quả dự đoán không nên được sử dụng cho các quyết định tài chính thực tế.
