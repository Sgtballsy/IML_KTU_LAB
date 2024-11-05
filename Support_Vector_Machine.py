import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target

print("target_names: setosa->0.0,\nversicolor->1.0,\nvirginica->2.0")

xt,xts,yt,yts = train_test_split(x,y,test_size=0.2)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(xt,yt)
y_pred = svm_classifier.predict(xts)
print("Accuracy: ",accuracy_score(yts,y_pred))
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))
z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM')
plt.show()
