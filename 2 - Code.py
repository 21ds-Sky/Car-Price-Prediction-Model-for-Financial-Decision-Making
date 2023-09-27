# Importing Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Section 1: Data Loading and Exploration
# Loading Data from CSV
car_data = pd.read_csv('car_data.csv')

# Printing first 5 rows
print(car_data.head())

# Checking Shape of Data
print(f'Shape of data: {car_data.shape}')

# Checking Null Values in Data
print(f'Null Values in data: \n{car_data.isnull().sum()}')

# Checking Data Types
print(f'Data Types of columns: \n{car_data.dtypes}')

# Section 2: Data Cleaning
# Removing Duplicate Rows
car_data.drop_duplicates(inplace=True)

# Removing rows with missing values
car_data.dropna(inplace=True)

# Removing rows with price = 0
car_data = car_data[car_data['Price'] != 0]

# Converting Kilometers_Driven to int
car_data['Kilometers_Driven'] = car_data['Kilometers_Driven'].astype(int)

# Converting Mileage to float
car_data['Mileage'] = car_data['Mileage'].str.replace('kmpl','')
car_data['Mileage'] = car_data['Mileage'].str.replace('km/kg','')
car_data['Mileage'] = car_data['Mileage'].astype(float)

# Converting Engine to int
car_data['Engine'] = car_data['Engine'].str.replace('CC','')
car_data['Engine'] = car_data['Engine'].astype(int)

# Converting Power to float
car_data['Power'] = car_data['Power'].str.replace('bhp','')
car_data = car_data[car_data['Power'] != 'null']
car_data['Power'] = car_data['Power'].astype(float)

# Extracting Brand from Name
car_data['Brand'] = car_data['Name'].str.split().str[0]
car_data.drop('Name', axis=1, inplace=True)

# Section 3: Data Visualization
# Creating Pairplot for Visualizing relationships between features
sns.pairplot(car_data)
plt.show()

# Creating Correlation Matrix and Visualizing with Heatmap
corr = car_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Creating Boxplots for Categorical Features
sns.boxplot(x='Fuel_Type', y='Price', data=car_data)
plt.title('Fuel Type vs Price')
plt.show()

sns.boxplot(x='Transmission', y='Price', data=car_data)
plt.title('Transmission vs Price')
plt.show()

sns.boxplot(x='Owner_Type', y='Price', data=car_data)
plt.title('Owner Type vs Price')
plt.show()

# Section 4: Model Training and Tuning

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv("cleaned_car_data.csv")

# Split the dataset into training and testing sets
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)
print("Linear Regression")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))

# Ridge Regression
ridge = Ridge()
params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid=params, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_ridge = grid_search.best_estimator_
y_train_pred = best_ridge.predict(X_train_scaled)
y_test_pred = best_ridge.predict(X_test_scaled)
print("\nRidge Regression")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))

# Lasso Regression
lasso = Lasso()
params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(lasso, param_grid=params, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_lasso = grid_search.best_estimator_
y_train_pred = best_lasso.predict(X_train_scaled)
y_test_pred = best_lasso.predict(X_test_scaled)
print("\nLasso Regression")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))

# Random Forest Regressor
rf = RandomForestRegressor()

# Define hyperparameters grid
params = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [3, 5, 7, 9],
    "min_samples_split": [2, 5, 10]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=params, 
                           cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Fit the model on training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

#Section 5: Model Evaluation
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the performance metrics on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Metrics:\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}")

#Section 6: Model Refinement

# Define the hyperparameters to tune
hyperparameters = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"]
}

# Create the Grid Search object
grid_search = GridSearchCV(rf_model, hyperparameters, cv=5, n_jobs=-1)

# Fit the Grid Search object to the training set
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_hyperparameters = grid_search.best_params_

print("Best Hyperparameters:")
for param, value in best_hyperparameters.items():
    print(f"{param}: {value}")

# Create a new Random Forest Regressor with the best hyperparameters
best_rf_model = RandomForestRegressor(**best_hyperparameters)

# Train the new model on the training set
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate the performance metrics on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Metrics (after hyperparameter tuning):\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}")

#In this project, we used a Random Forest Regressor to predict the prices of used cars based on various features such as mileage, year, brand, etc. We followed a six-step approach which involved data cleaning and preprocessing, exploratory data analysis, feature engineering, model training and tuning, model evaluation, and model refinement. We used various performance metrics such as mean squared error, root mean squared error, and R-squared score to evaluate the performance of our model. Finally, we performed hyperparameter tuning using Grid Search to find the best combination of hyperparameters for our model, and retrained the model with those hyperparameters to get better performance.