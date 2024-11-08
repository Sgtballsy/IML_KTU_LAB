from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('.csv')
print('dataset')
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values
lr=LinearRegression
x_train,x_test,y_test,y_train=train_test_split(x,y)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)


plt.scatter(x,y,color='red')
plt.plot(x_test,y_pred)
plt.show()

print('accuracy',lr.score(x_test,y_test))
print('MSE:',mean_squared_error(y_test,y_pred))
print('MAE:',mean_absolute_error(y_test,y_pred))
