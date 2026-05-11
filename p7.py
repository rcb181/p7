import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(Xtr, ytr)
yp = lr.predict(Xte)

plt.scatter(yte, yp)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r')
plt.title("Linear Regression")
plt.grid()
plt.show()

print("MSE:", mean_squared_error(yte, yp))
print("R2:", r2_score(yte, yp))


a = fetch_openml(name="autoMpg", version=1, as_frame=True)
X = a.data[['horsepower']].dropna().astype(float)
y = a.target.loc[X.index].astype(float)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(3)
model = LinearRegression().fit(poly.fit_transform(Xtr), ytr)
yp = model.predict(poly.transform(Xte))

s = np.argsort(Xte.values.flatten())

plt.scatter(Xte, yte)
plt.plot(Xte.values.flatten()[s], yp[s], 'r')
plt.title("Polynomial Regression")
plt.grid()
plt.show()

print("MSE:", mean_squared_error(yte, yp))
print("R2:", r2_score(yte, yp))
