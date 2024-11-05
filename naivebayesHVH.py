import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score,recall_score

dataset = pd.read_csv('./spam.csv')
x=dataset.iloc[:,1].values
y=dataset.iloc[:,0].values

le = LabelEncoder()
y_labels = le.fit_transform(y)

vect = CountVectorizer()

x_fit = vect.fit_transform(x)


model=GaussianNB()
xt,xts,yt,yts = train_test_split(x_fit.toarray(),y_labels)
model.fit(xt,yt)
y_pred = model.predict(xts)

p = model.score(x_fit.toarray(),y_labels)
print("Accuracy: ",p)

m = precision_score(yts,y_pred,average=None)
n = recall_score(yts,y_pred,average=None)
print("Precision: ",m)
print("Recall: ",n)