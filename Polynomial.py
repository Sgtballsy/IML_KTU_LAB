import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('.csv')
print(dataset)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
pr=PolynomialFeatures(degree=)
x_poly=pr.fit_transform(x)
pr=linearRegression()
pr.fit(x_poly,y)
y_pred=pr.predict(x_poly)
print(y_pred)

plt.scatter(x,y,color='red')
plt.plot(x,y_pred,color='blue')
plt.title('Polynomial Regression')
plt.xlabel('indepenent variable')
plt.ylabel('dependent variable')
plt.show()

print("Accuracy of the model is:",pr.score(x_poly,y))
