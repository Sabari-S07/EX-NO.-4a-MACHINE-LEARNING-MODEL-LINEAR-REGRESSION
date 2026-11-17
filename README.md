# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
## AIM
To predict house prices using regression models and compare the performance of different machine learning regression models based on RMSE, MAE, and R².
1.Machine Learning:Machine Learning is used to learn patterns from existing data and make predictions. 
•	Regression is a supervised learning technique used to predict continuous numerical values. 
•	In this experiment, regression models are used to predict the price of a house. 
•	The dataset contains house-related features such as: 
o	square_feet 
o	num_rooms 
o	age 
o	distance_to_city(km) 
•	The target variable is: 
o	price 
## DATASET DESCRIPTION
•	Dataset: House Price Dataset 
•	Problem: Predict house price. 
•	Features (X): 
o	square_feet – size of the house. 
o	num_rooms – number of rooms. 
o	age – age of the house in years. 
o	distance_to_city(km) – distance from the city centre. 
•	Target (y): 
o	price – continuous house price. 
## PROBLEM STATEMENT
•	Develop a machine learning model to predict house prices. 
•	Use house characteristics as input. 
•	Train different regression models. 
•	Compare their prediction performance. 
•	Select the better-performing model based on evaluation metrics. 
## REGRESSION MODELS USED
The uploaded notebook compares the following models:
1.	Linear Regression 
2.	Ridge Regression 
3.	Lasso Regression 
4.	ElasticNet Regression 
5.	Polynomial Regression 
6.	Decision Tree Regressor 
7.	Random Forest Regressor 
8.	Gradient Boosting Regressor 
9.	Support Vector Regressor (SVR) 
10.	K-Nearest Neighbors (KNN) Regressor 
### LIBRARIES USED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
### LOAD THE DATASET
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

df.head()
### DATA OVERVIEW
Display the dataset
df
Display information
df.info()
Display shape
df.shape
Summary statistics
df.describe()
Check missing values
df.isnull().sum()
## EXPLORATORY DATA ANALYSIS
Distribution of Features
The notebook examines the distribution of:
•	square_feet 
•	num_rooms 
•	age 
•	distance_to_city(km) 
•	price 
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
9. CORRELATION ANALYSIS
•	Correlation shows the relationship between numerical variables. 
•	A correlation heatmap is used to visualize these relationships. 
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()
10. SCATTER PLOTS
Scatter plots are used to study the relationship between individual features and house price.
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.show()
## OUTLIER DETECTION
•	Boxplots are used to identify extreme values. 
•	Outliers may negatively affect regression models. 
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
## OUTLIER TREATMENT
The notebook removes extremely low and extremely high house prices using the 1st and 99th percentiles.
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]
•	This reduces the effect of extreme house prices. 
•	It helps the models learn from more typical observations. 
13. DEFINE FEATURES AND TARGET
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']
•	X → Input features. 
•	y → Target house price. 
14. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
•	80% → Training data 
•	20% → Testing data 
## FEATURE SCALING
The notebook uses StandardScaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
•	Scaling puts the features on a comparable scale. 
•	The scaler is fitted only on training data. 
•	The same transformation is applied to test data. 

## BASELINE MODEL
The baseline predicts the mean house price for every test observation.
y_pred_baseline = (
    np.mean(y_train) *
    np.ones_like(y_test)
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline RMSE: {rmse_baseline:.2f}, "
    f"MAE: {mae_baseline:.2f}"
)

## LINEAR REGRESSION
•	Linear Regression finds a linear relationship between input features and house price. 
•	It is used as the main baseline regression model. 
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

## RIDGE REGRESSION
•	Ridge Regression is a regularized version of Linear Regression. 
•	It helps control large model coefficients. 
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

## LASSO REGRESSION
•	Lasso Regression uses L1 regularization. 
•	It can reduce some feature coefficients toward zero. 
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

##  ELASTIC NET REGRESSION
•	ElasticNet combines L1 and L2 regularization. 
•	It is useful when several features may contribute to the prediction. 
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

##  POLYNOMIAL REGRESSION
•	Polynomial Regression extends Linear Regression by creating polynomial features. 
•	The notebook uses degree 2. 
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)

##  DECISION TREE REGRESSOR
•	A Decision Tree divides the data into different regions based on feature values. 
•	It can model nonlinear relationships. 
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)

##  RANDOM FOREST REGRESSOR
•	Random Forest combines multiple decision trees. 
•	It generally provides more robust predictions than a single tree. 
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

##  GRADIENT BOOSTING REGRESSOR
•	Gradient Boosting builds models sequentially. 
•	Each new model attempts to improve the errors of previous models. 
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)

## SUPPORT VECTOR REGRESSOR
•	SVR uses Support Vector Machine principles for regression. 
•	The notebook uses an RBF kernel. 
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

##  KNN REGRESSOR
•	KNN predicts a value based on nearby observations. 
•	The notebook uses 5 neighbors. 
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

## MODEL EVALUATION
The notebook uses three metrics:
RMSE
•	Root Mean Squared Error. 
•	Lower value indicates better performance. 
MAE
•	Mean Absolute Error. 
•	Lower value indicates better performance. 
R²
•	Measures how well the model explains variation in house prices. 
•	Higher value generally indicates better performance. 
Code
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.sort_values(
    by="RMSE"
)

##  MODEL COMPARISON
<img width="601" height="250" alt="image" src="https://github.com/user-attachments/assets/bca8dfef-1052-4e92-90e1-0f6356888be5" />
		

### Comparison Criteria
•	Lower RMSE → Better model. 
•	Lower MAE → Better model. 
•	Higher R² → Better model. 
The notebook sorts the models according to RMSE to compare their performance.
## ACTUAL VS PREDICTED PRICE
plt.figure(figsize=(15,12))

for i, (name, y_pred) in enumerate(models.items()):

    plt.subplot(5,2,i+1)

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")

plt.tight_layout()
plt.show()
•	The plot compares actual house prices with predicted prices. 
•	Points closer to the diagonal line indicate better predictions. 
30. RESIDUAL ANALYSIS
•	A residual is the difference between actual and predicted values. 
Residual = Actual Price − Predicted Price
for name, y_pred in models.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color='r',
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.show()
•	Residuals close to zero indicate smaller prediction errors. 
•	Residual plots help identify unusual prediction patterns. 
##  RANDOM FOREST FEATURE IMPORTANCE
The notebook calculates the importance of each feature using Random Forest.
importances = rf.feature_importances_

feat_names = X.columns

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances,
    y=feat_names
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()
This helps identify which house features contribute more to the Random Forest prediction.
##  GRADIENT BOOSTING FEATURE IMPORTANCE
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances_gbr,
    y=feat_names
)

plt.title(  "Gradient Boosting Feature Importance")

plt.show()
This shows the relative contribution of the house features to the Gradient Boosting model.

## RMSE COMPARISON GRAPH
plt.figure(figsize=(10,6))

sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE")
)

plt.title(
    "RMSE Comparison Across Regression Models"
)

plt.show()
•	The graph provides a visual comparison of model errors. 
•	The model with the lowest RMSE performs best according to this metric.

## OUTPUT
<img width="541" height="393" alt="929cc98a-5f07-439e-8e1b-4508ebf44050" src="https://github.com/user-attachments/assets/d1099d61-80b5-4734-8b23-eed862589cfa" />
<img width="549" height="393" alt="964805a6-8feb-425b-b1de-3e47bfdc1d8f" src="https://github.com/user-attachments/assets/1c88673c-4f21-4aae-ad1c-f64e3ed3a63b" />
<img width="540" height="393" alt="b363a143-3c4b-4636-8c10-e8972a14a7bf" src="https://github.com/user-attachments/assets/48470ae8-710d-4ad2-b65d-597d23cb0e46" />
<img width="540" height="393" alt="29afb055-9913-4dcf-a9d1-6036deba7ce1" src="https://github.com/user-attachments/assets/ec2b61fc-72cb-4d2c-a6cd-23210beb8779" />
<img width="549" height="393" alt="73315f43-08ab-4208-a90e-22d8c7badd59" src="https://github.com/user-attachments/assets/de45ace1-8d53-4b9d-bdb5-b9201f386dd7" />
<img width="637" height="528" alt="efad57d5-e7bb-4ea1-980a-ef3f95f4e20d" src="https://github.com/user-attachments/assets/47b66d08-ca9c-4696-a020-266919be1fc3" />
<img width="579" height="393" alt="1f2d6d0c-5a73-4b00-9f68-403e4deb2592" src="https://github.com/user-attachments/assets/a6fa8964-cc09-4710-a5eb-1944e12becda" />
<img width="578" height="393" alt="2efd3ca3-c935-4e0e-8bdc-356284e01600" src="https://github.com/user-attachments/assets/02368b9b-a3d9-4198-97e8-3e9c2a0feac6" />
<img width="578" height="393" alt="2d0dc34a-3734-4bb1-9113-dad6cbe29862" src="https://github.com/user-attachments/assets/a7a50751-0a0d-48a9-aa80-aa57307ad0c4" />
<img width="578" height="393" alt="5b494c0a-162b-45a9-b24f-1dd72f67e92c" src="https://github.com/user-attachments/assets/06f86c41-26a0-4776-933e-b3c7bd7f180b" />
<img width="549" height="355" alt="c47df92d-6996-4cc0-a215-784961094293" src="https://github.com/user-attachments/assets/115978b9-835b-4a93-acd3-0981b35a6f78" />
<img width="523" height="355" alt="3a808c0f-4b53-41b1-80cd-f662e998cdb8" src="https://github.com/user-attachments/assets/e70e5372-ed1d-43dc-8dd5-44d0ec72f409" />
<img width="540" height="355" alt="50a1cecd-56a4-4a1c-a51d-2923c2cfe440" src="https://github.com/user-attachments/assets/7a94635a-d73c-4a9d-a6d1-5fdfc9778ece" />
<img width="532" height="355" alt="0f6a3dae-a51e-48eb-b550-e4df1ca092cb" src="https://github.com/user-attachments/assets/2ad3c95f-d657-4899-9e6b-7cd3aa97260c" />
<img width="578" height="355" alt="b02cec7a-c9fd-405e-9223-599320e52a58" src="https://github.com/user-attachments/assets/7557f381-6ed5-442b-a43c-536dcb32646c" />
<img width="1488" height="1190" alt="b74d0bf4-6624-41d9-8cc0-04c3017c4085" src="https://github.com/user-attachments/assets/a30566f3-7866-4f85-8839-66beae41e933" />
<img width="576" height="393" alt="ad35ea8f-173f-4b0a-ac08-dc27cd6cf323" src="https://github.com/user-attachments/assets/cee5de70-5727-4963-8d6e-2c35fb97f2ab" />
<img width="576" height="393" alt="99f91f98-98d3-4312-af09-21ed6aed14ce" src="https://github.com/user-attachments/assets/5871df40-11e1-4730-8ec1-4395c7bd04a3" />
<img width="576" height="393" alt="09de21a5-39ab-42b7-9f57-65f408264424" src="https://github.com/user-attachments/assets/54573cdc-b8fb-4671-983e-ffc940067c9e" />
<img width="587" height="393" alt="53963de5-c32f-4c42-9140-b3067ef9f8a3" src="https://github.com/user-attachments/assets/ea8b3bf2-8ea9-4acd-8f68-d02e05e74d29" />
<img width="576" height="393" alt="009a809c-7186-4fd5-972b-71e40fdc67fd" src="https://github.com/user-attachments/assets/65b38249-f9c2-408a-8881-7d3438f89244" />
<img width="597" height="393" alt="cbe0aa4c-7fd6-4837-b4da-184499a5bca7" src="https://github.com/user-attachments/assets/5f3e4df9-c08f-4050-8c68-27b7ccfd44d3" />
<img width="569" height="393" alt="c7f514ed-d56a-45c4-8a1e-063a283c4114" src="https://github.com/user-attachments/assets/8b66ce63-8758-45bf-8b3a-af3989eb9c74" />
<img width="590" height="393" alt="97bd034e-fb92-4769-9cbc-4219e101804c" src="https://github.com/user-attachments/assets/95fe35c0-ef79-4455-bf7e-0bd4723c81d5" />
<img width="578" height="393" alt="27276ae3-1471-46d0-b83d-3bbde34ec1f4" src="https://github.com/user-attachments/assets/695a16ff-d0bd-4adb-a998-9c5c7685c483" />
<img width="569" height="393" alt="8fe7f0a7-2767-424c-948b-13586579d562" src="https://github.com/user-attachments/assets/2cf9396a-e9d1-4eac-850b-b40c04d9cf53" />
<img width="663" height="374" alt="c6cdac90-0744-4491-8ca2-4664daa69a7a" src="https://github.com/user-attachments/assets/518f513b-7396-4b6a-8c77-52fb408f9f38" />
<img width="658" height="374" alt="8120b392-a4af-4d38-b435-5e3a6b2470bd" src="https://github.com/user-attachments/assets/ab1ea1f8-0abb-4c06-be4e-cb1ba526c4e4" />
<img width="980" height="547" alt="8a6928d9-4e48-4238-a6a9-186fa340fc30" src="https://github.com/user-attachments/assets/53143105-cccf-4066-80c1-b4000628c8a3" />

## CONCLUSION
Thus, Linear Regression and other regression models were successfully applied for house price prediction, and their performance was compared using standard regression evaluation metrics.

