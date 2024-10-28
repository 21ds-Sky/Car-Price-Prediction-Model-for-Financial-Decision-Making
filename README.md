<h1 align="center">🚗 Car Price Prediction</h1>
<p align="center">
  <strong>Analyze and predict used car prices using regression models</strong>
</p>

---

## 📋 Objective
This project aims to analyze and visualize used car prices using a dataset from Kaggle, employing various linear regression models to predict the most probable car prices. The main models used include **Linear Regression**, **Ridge Regression**, **Lasso Regression**, and **ElasticNet Regression**.

Dataset: <a href="https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho">Vehicle Dataset from Cardekho</a>

---

## 🗃️ Dataset Description
The dataset includes **301 records** and **9 features** related to used cars listed on <a href="https://www.cardekho.com/">Cardekho.com</a>.

### Features:
- **Car_Name**: The name of the car.
- **Year**: Year of purchase.
- **Selling_Price**: Selling price of the car.
- **Present_Price**: Current ex-showroom price.
- **Kms_Driven**: Kilometers driven.
- **Fuel_Type**: Fuel type (Diesel, Petrol, CNG).
- **Seller_Type**: Dealer or Individual.
- **Transmission**: Manual or Automatic.
- **Owner**: Number of previous owners.

---

## 🚀 Steps to Achieve the Objective

### 1. Data Loading
- Load the dataset from Kaggle.
- Inspect for missing values and clean data as needed.

### 2. Data Exploration
- Perform **exploratory data analysis (EDA)** to understand data distribution.
- Visualize data trends and patterns.

### 3. Feature Engineering
- Convert categorical features using **one-hot encoding**.
- Engineer new features to enhance model performance if necessary.

### 4. Model Building
- Split dataset into training and testing sets.
- Implement regression models:
  - **Linear Regression**
  - **Ridge Regression**
  - **Lasso Regression**
  - **ElasticNet Regression**

---

## 📊 Results

### Ordinary Least Squares (OLS) Regression:
- **R-squared**: 0.952
- **Adjusted R-squared**: 0.951

### Regularized Regression Models:
- **Ridge Regression**: R-squared = 0.756
- **Lasso Regression**: R-squared = 0.730
- **ElasticNet Regression**: R-squared = 0.766

---

## 📝 Conclusion
- The **OLS regression model** performed the best with an **R-squared value of 0.952**.
- Regularized models (Ridge, Lasso, ElasticNet) yielded lower R-squared values, suggesting that **regularization did not enhance performance** on this dataset.
- This indicates that while regularization techniques can help prevent overfitting, they may not outperform OLS for every dataset.

---

---

## 🛠️ Tools & Libraries
- **Python**: <code>pandas</code>, <code>numpy</code>, <code>scikit-learn</code>, <code>matplotlib</code>, <code>seaborn</code>
- **Jupyter Notebook** for interactive analysis


---

## 🤝 Contributing
Feel free to fork this project, make updates, and submit pull requests. Contributions are always welcome!




