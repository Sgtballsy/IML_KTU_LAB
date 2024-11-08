import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error



dataset = pd.read_csv('./a_e_sal.csv')



x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

xt,xts,yt,yts = train_test_split(x,y)



lr = LinearRegression()

lr.fit(xt,yt)



x1s,x2s = np.meshgrid(np.linspace(xt[:,0].min(),xt[:,0].max(),10)

,np.linspace(xt[:,1].min(),xt[:,1].max(),10))



ox = pd.DataFrame({'x1':x1s.ravel(),'x2':x2s.ravel()})



fy = lr.predict(ox)

fy = np.array(fy)



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111,projection='3d')

ax.scatter(xt[:,0],xt[:,1],yt,color='black')

ax.plot_surface(x1s,x2s,fy.reshape(x1s.shape),color='black',alpha=0.5)

ax.set_xlabel('age')

ax.set_ylabel('exp')

ax.set_zlabel('salary')



print("Mean Squared Error: ",mean_squared_error(yts,lr.predict(xts)))

print("Mean Absolute Error: ",mean_absolute_error(yts,lr.predict(xts)))

print("R2 Score: ",lr.score(xts,yts))



plt.savefig("MVLR.png")
