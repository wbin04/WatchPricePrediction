# 🕰️ Watch Price Prediction Project

## Tổng quan dự án

Dự án này xây dựng một hệ thống dự đoán giá đồng hồ cao cấp sử dụng kỹ thuật Machine Learning. Hệ thống bao gồm toàn bộ pipeline từ thu thập dữ liệu, tiền xử lý, phân tích khám phá, kỹ thuật đặc trưng đến xây dựng mô hình dự đoán.

## 📁 Cấu trúc dự án

```
WatchPricePrediction/
├── 📊 Data Collection & Processing
│   ├── watchbase_crawler_scrapy.py                             # Web scraping script
│   ├── Descriptive_Statistics_Visualization.ipynb              # Thống kê mô tả
│   └── Data Preprocessing.ipynb                                # Tiền xử lý dữ liệu
│
├── 🔍 Exploratory Data Analysis  
│   ├── EDA.ipynb                                               # Phân tích khám phá dữ liệu
│   └── Feature Engineering.ipynb                               # Kỹ thuật đặc trưng
│
├── 🤖 Machine Learning Models
│   ├── CatBoost Regression.ipynb                               # Mô hình CatBoost
│   ├── XGBoost Regression.ipynb                                # Mô hình XGBoost
│   └── LightGBM.ipynb                                          # Mô hình LightGBM
│
└── 📂 datasets/
    ├── watchbase_data_raw_scrapy.csv                           # Dữ liệu thô
    ├── watchbase_data_preprocessed_scrapy.csv                  # Dữ liệu đã tiền xử lý
    └── watchbase_data_featured_scrapy.csv                      # Dữ liệu với feature engineering
```

## 🚀 Quy trình thực hiện

### 1. 🕷️ Thu thập dữ liệu (Data Collection)

**File:** `watchbase_crawler_scrapy.py`

#### Mô tả:
- Sử dụng Scrapy framework để thu thập dữ liệu từ website: [watchbase.com](watchbase.com)
- Thu thập thông tin chi tiết về đồng hồ từ 10 thương hiệu nổi tiếng

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
python watchbase_crawler_scrapy.py
```

#### Kết quả:
- File output: `watchbase_data_raw_scrapy.csv`
- Số lượng: ~6,300 mẫu dữ liệu

---

### 2. 📊 Thống kê mô tả (Descriptive Statistics)

**File:** `Descriptive_Statistics_Visualization.ipynb`

#### Mục đích:
- Hiểu tổng quan về dataset
- Phát hiện missing values, outliers
- Thống kê mô tả cơ bản

#### Nội dung chính:
- Kiểm tra missing values và data types
- Thống kê mô tả cho biến số
- Phân bố của biến phân loại
- Tạo visualizations cơ bản

---

### 3. 🧹 Tiền xử lý dữ liệu (Data Preprocessing)

**File:** `Data Preprocessing.ipynb`

#### Các bước xử lý:

##### 3.1 Xử lý missing values:
- **Case Material:** Thay thế NaN bằng "Stainless Steel" (phổ biến nhất)
- **Water Resistance:** Chuyển đổi format và điền bằng mode
- **Dial Indexes, Dial Hands:** Điền bằng mode
- **Glass, Case Back, Case Shape:** Điền bằng mode
- **Case Diameter:** Chuyển đổi format và điền bằng mode
- **Dial Color:** Điền bằng mode

##### 3.2 Loại bỏ columns không cần thiết:
- `Produced`: Quá nhiều missing values
- `Lug Width`: Quá nhiều missing values  
- `Dial Finish`: Quá nhiều missing values
- `Reference`, `Name`: Không cần thiết cho modeling

##### 3.3 Chuẩn hóa dữ liệu:
- Chuyển đổi `Water Resistance` từ "30 m" → 30.0
- Chuyển đổi `Case Diameter` từ "40 mm" → 40.0
- Chuyển đổi `Price` thành float
- Rename `Family` → `Model`

##### 3.4 Xử lý biến Limited:
- Tách lấy phần đầu tiên từ chuỗi phức tạp

#### Output:
- File: `watchbase_data_preprocessed_scrapy.csv`
- Dataset sạch, sẵn sàng cho EDA

---

### 4. 🔍 Phân tích khám phá dữ liệu (EDA)

**File:** `EDA.ipynb`

#### Mục đích:
- Hiểu sâu về phân bố dữ liệu
- Khám phá mối quan hệ giữa các biến
- Phát hiện patterns và insights

#### Phân tích chính:

##### 4.1 Phân tích biến số:
- **Case Diameter:** Phân bố lệch phải, tập trung 35-45mm
- **Water Resistance:** Phân bố đa modal, có clusters
- **Price:** Phân bố lệch phải mạnh, nhiều outliers

##### 4.2 Phân tích biến phân loại:
- **Brand:** Phân bố không đều, Omega và Rolex chiếm ưu thế
- **Model:** Đa dạng, mỗi brand có nhiều model
- **Case Material:** Stainless Steel phổ biến nhất
- **Dial Color:** Black và White/Silver chiếm ưu thế

##### 4.3 Mối quan hệ giữa các biến:
- Correlation matrix cho biến số
- Cross-tabulation cho biến phân loại
- Price distribution theo từng nhóm

#### Insights quan trọng:
- Brand là yếu tố quan trọng nhất ảnh hưởng đến giá
- Case Diameter có correlation với Price
- Limited edition có giá cao hơn
- Case Material ảnh hưởng lớn đến Price

---

### 5. ⚙️ Kỹ thuật đặc trưng (Feature Engineering)

**File:** `Feature Engineering.ipynb`

#### Mục đích:
- Tạo features mới từ dữ liệu gốc
- Xử lý outliers
- Chuẩn bị dữ liệu cho modeling

#### Các kỹ thuật áp dụng:

##### 5.1 Outlier Treatment:
- **Case Diameter:** Loại bỏ values > 60mm (pocket watches, errors)
- Sử dụng boxplot và histogram để detect outliers

##### 5.2 Feature Grouping:

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
resistance_mapping = {
    'Basic': [30, 50],
    'Sports': [100, 200, 300],
    'Diving': [500, 1000, 2000, 3900]
}
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

##### 5.3 Target Variable Transformation:
- **Log Transformation:** `LogPrice = log(Price)` 
- Giảm skewness của Price distribution
- Cải thiện model performance

##### 5.4 Text Processing:
- Chuyển brand names về lowercase
- Standardize categorical values

#### Output:
- File: `watchbase_data_featured_scrapy.csv`
- Features engineered, ready for modeling

---

### 6. 🤖 Machine Learning Models

#### 6.1 📈 CatBoost Regression

**File:** `CatBoost Regression.ipynb`

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

##### Train/Validation/Test Split:
- **Training:** 70% 
- **Validation:** 20%
- **Test:** 10%

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

##### Evaluation Metrics:
- **R² Score:** Coefficient of determination
- **RMSE:** Root Mean Square Error
- **MAE:** Mean Absolute Error

##### Feature Importance Analysis:
- Permutation importance
- SHAP values (nếu có)
- Categorical feature importance

---

#### 6.2 🚀 XGBoost Regression  

**File:** `XGBoost Regression.ipynb`

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

##### Train/Validation/Test Split:
- **Training:** 70%
- **Validation:** 10% 
- **Test:** 20%

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

---

#### 6.3 ⚡ LightGBM Regression

**File:** `LightGBM.ipynb`

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

##### Train/Validation/Test Split:
- **Training:** 70%
- **Validation:** 10%
- **Test:** 20%

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

##### LightGBM Dataset:
```python
lgb_train = lgb.Dataset(X_train_enc, y_train)
lgb_val = lgb.Dataset(X_val_enc, y_val, reference=lgb_train)
```

##### Feature Importance Visualization:
```python
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
```

---

### 7. 📊 So sánh Models

#### Performance Comparison:

| Model | R² Score | RMSE | MAE | Ưu điểm | Nhược điểm |
|-------|----------|------|-----|---------|------------|
| **CatBoost** | ~0.85 | ~0.42 | ~0.30 | - Native categorical support<br>- Robust overfitting prevention<br>- No encoding needed | - Slower training<br>- More memory usage |
| **XGBoost** | ~0.87 | ~0.39 | ~0.27 | - Fast training<br>- Excellent feature importance<br>- Wide adoption | - Requires encoding<br>- More hyperparameter tuning |
| **LightGBM** | ~0.87 | ~0.39 | ~0.27 | - Fastest training speed<br>- Memory efficient<br>- Good performance | - Requires encoding<br>- Can overfit with small datasets |

#### Feature Importance Insights:
1. **Brand:** Yếu tố quan trọng nhất (30-40% importance)
2. **Model:** Quan trọng thứ hai (15-20% importance)  
3. **Case Material:** Ảnh hưởng lớn đến giá (10-15% importance)
4. **Case Diameter:** Kích thước quan trọng (8-12% importance)
5. **Limited Edition:** Premium factor (5-8% importance)

---

## 🛠️ Cài đặt và Chạy dự án

### Requirements:
```bash
pip install pandas numpy matplotlib seaborn
pip install scrapy requests
pip install scikit-learn
pip install catboost xgboost lightgbm
pip install category-encoders
```

### Chạy toàn bộ pipeline:

1. **Thu thập dữ liệu:**
```bash
python watchbase_crawler_scrapy.py
```

2. **Tiền xử lý và EDA:**
```bash
jupyter notebook "Data Preprocessing.ipynb"
jupyter notebook "EDA.ipynb"
```

3. **Feature Engineering:**
```bash
jupyter notebook "Feature Engineering.ipynb"
```

4. **Training Models:**
```bash
jupyter notebook "CatBoost Regression.ipynb"
jupyter notebook "XGBoost Regression.ipynb"
jupyter notebook "LightGBM.ipynb"
```

---

## 📈 Kết quả và Insights

### Business Insights:

1. **Brand Premium:** Patek Philippe và Rolex có giá cao nhất
2. **Material Impact:** Platinum > Gold > Steel về giá trị
3. **Size Matters:** Đồng hồ XL/XXL có giá cao hơn
4. **Limited Edition Premium:** Tăng giá 20-50%
5. **Water Resistance:** Diving watches có giá premium

### Technical Insights:

1. **Log Transformation:** Cải thiện model performance đáng kể
2. **Feature Engineering:** Grouping categorical variables hiệu quả
3. **Model Selection:** CatBoost performs tốt hơn với categorical data
4. **Target Encoding:** Hiệu quả cho XGBoost và LightGBM với high cardinality categories
5. **Training Speed:** LightGBM nhanh nhất, theo sau là XGBoost, sau cùng là CatBoost

---

## 🔮 Hướng phát triển

### Near-term:
- Hyperparameter tuning chi tiết hơn
- Cross-validation với time-series split
- Ensemble methods (CatBoost + XGBoost + LightGBM)
- SHAP analysis cho interpretability

### Long-term:
- Real-time price tracking
- Deep Learning models (Neural Networks)
- Computer Vision cho watch image analysis
- Web application deployment
- API service cho price prediction

---

## 📄 License

This project is for educational purposes only. Data is collected from public sources for academic research.

---

**Lưu ý:** Dự án này chỉ mang tính chất học thuật và nghiên cứu. Kết quả dự đoán không nên được sử dụng cho các quyết định tài chính thực tế.
