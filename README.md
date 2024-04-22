# Used Car Price Prediction Model for Financial Decision-Making
Problem Description
The objective of this project is to predict the price of used cars based on various features such as car’s 
model, year of manufacturing, kilometers driven, fuel type, transmission type, etc. The dataset for this 
project is obtained from Kaggle and it contains details of used cars sold in India.
Dataset Description
The dataset contains 12 columns and 6019 rows. The columns in the dataset are:
• Name: Name of the car
• Location: Location where the car is being sold or is available for purchase
• Year: Manufacturing year of the car
• Kilometers_Driven: The total kilometers driven by the car
• Fuel_Type: The type of fuel used by the car
• Transmission: The type of transmission used by the car
• Owner_Type: Whether the car has been previously owned or not
• Mileage: The standard mileage offered by the car company in kmpl or km/kg
• Engine: The displacement volume of the engine in cc
• Power: The maximum power of the engine in bhp
• Seats: The number of seats in the car
• Price: The price of the used car in lakhs

Code Explanation : 
Here is the simple explanation for the code you can find at code.py file. 
Section 1: Importing Libraries and Data 
• In this section, the necessary libraries are imported, including pandas, numpy, seaborn, and 
matplotlib. 
• The data is loaded into a pandas dataframe using the read_csv function from pandas. 
Section 2: Data Cleaning and Preprocessing 
• In this section, the data is cleaned and preprocessed. 
• First, irrelevant columns are dropped using the drop function. 
• Then, missing values are handled using the fillna function. 
• Next, categorical variables are encoded using the get_dummies function. 
• Finally, the data is split into training and testing sets using the train_test_split function from 
sklearn. 
Section 3: Feature Scaling 
• In this section, feature scaling is performed on the data. 
• The StandardScaler function from sklearn is used to scale the numerical columns. 
Section 4: Model Training and Tuning 
• In this section, various regression models are trained and tuned. 
• First, a linear regression model is trained using the LinearRegression function from sklearn. 
• Next, a decision tree regressor model is trained using the DecisionTreeRegressor function from 
sklearn. 
• Then, a random forest regressor model is trained using the RandomForestRegressor function 
from sklearn. 
• Hyperparameter tuning is performed on the random forest regressor model using the 
GridSearchCV function from sklearn. 
• Finally, the best model is selected and its performance is evaluated using the mean absolute 
error and mean squared error. 
Section 5: Generating Summary 
• In this section, a summary of the best model’s performance is generated. 


