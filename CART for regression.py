#CART for regression
import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_regression(X, y, name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nDataset:", name)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)


# -------------------------
# Dataset 1: Diabetes
# -------------------------
diabetes = load_diabetes()
run_regression(diabetes.data, diabetes.target, "Diabetes Dataset")

# -------------------------
# Dataset 2: California Housing
# -------------------------
housing = fetch_california_housing()
run_regression(housing.data, housing.target, "California Housing Dataset")
