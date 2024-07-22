
#Car Price Prediction

Objective

In this project, the goal is to analyze and visualize used car prices from a dataset available on Kaggle. The aim is to predict the most probable car prices using basic linear regression models: Linear Regression, Ridge Regression, Lasso Regression, and ElasticNet Regression.

Vehicle dataset from cardekho : https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho/


Dataset Description
The dataset contains information about used cars listed on www.cardekho.com. It will be used to predict car prices with the help of regression models.

Dataset Size: 301 records * 9 features

Features:
Car_Name: The name of the car.
Year: The year in which the car was bought.
Selling_Price: The price at which the owner wants to sell the car.
Present_Price: The current ex-showroom price of the car.
Kms_Driven: The distance completed by the car in kilometers.
Fuel_Type: The type of fuel the car uses (e.g., Diesel, Petrol, CNG).
Seller_Type: Defines whether the seller is a dealer or an individual.
Transmission: Defines whether the car is manual or automatic.
Owner: The number of previous owners the car has had.
Steps to Achieve the Objective:
Data Loading:

Load the dataset from Kaggle.
Inspect the dataset for any missing values and perform necessary cleaning.

Data Exploration:
Perform exploratory data analysis (EDA) to understand the distribution of data and the relationships between different features.
Visualize the data using plots to identify trends and patterns.

Feature Engineering:
Convert categorical features into numerical values using techniques like one-hot encoding.
Create new features if necessary to improve the predictive power of the models.

Model Building:
Split the dataset into training and testing sets.

Implement basic linear regression models including:
Linear Regression
Ridge Regression
Lasso Regression
ElasticNet Regression

Results
Ordinary Least Squares (OLS) Regression:

R-squared: 0.952

Adjusted R-squared: 0.951

Regularized Regression Results:

Ridge Regression: R-squared = 0.756

Lasso Regression: R-squared = 0.730

ElasticNet Regression: R-squared = 0.766

Conclusion:-

The OLS regression model performed the best with an R-squared value of 0.952.

Regularized regression models (Ridge, Lasso, ElasticNet) showed lower R-squared values, indicating they were less effective in this particular case.

These results suggest that while regularization techniques are useful to prevent overfitting, they may not always provide better performance compared to OLS in all datasets.
