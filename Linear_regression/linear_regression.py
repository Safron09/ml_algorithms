# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Data from Your Excel File
df = pd.read_csv("data/test.csv")


print("Revie of Data:")
print(df.head())

# The data has two columns: 'x' and 'y'
# 'x' → Independent variable (input)
# 'y' → Dependent variable (output / target)

# Step 3: Prepare the Data for the Model
X = df[['x']]
y = df['y']


# Step 4: Create and Train Linear regression model
model = LinearRegression()
model.fit(X, y)

theta_1 = model.coef_[0]  # Slope (w)
theta_0 = model.intercept_ # Intercept (b)

print("\n Model Parameter Learned:")
print(f" slope (01/ weight): {theta_1:.4f}")
print(f" Inetrcept (00 / bias): {theta_0:.4f}")

# Step 5 Make Prediction
df['y_pred'] = model.predict(X)    #New column wiht pred values

print("\n Sample Prediction")
print(df.head(5))

# Stpe 6 Evaluate Performance
mse = mean_squared_error(y, df['y_pred'])
r2 = r2_score(y, df['y_pred'])

print("\n model Evaluation:")
print(f" Mean square Error (MSE): {mse:.3f}")
print(f" R2 Score: {r2:.3f}")

# Step 7 Visualize Regression LINE
plt.figure(figsize=(8,5))
plt.scatter(X, y, color = 'blue', label='Actual Data Point')
plt.plot(X, df['y_pred'], color='red', label = 'Predicted Line')
plt.xlabel('x (Input Features)')
plt.ylabel('y (Target oupput)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()