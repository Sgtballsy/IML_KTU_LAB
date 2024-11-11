import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

y_onehot = np.zeros((y.size, y.max() + 1))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_layer_size = X_train.shape[1]  
hidden_layer_size = 5  
output_layer_size = y_train.shape[1]

np.random.seed(42)
W1 = np.random.randn(input_layer_size, hidden_layer_size)  
b1 = np.zeros((1, hidden_layer_size))  
W2 = np.random.randn(hidden_layer_size, output_layer_size)  
b2 = np.zeros((1, output_layer_size))  

training_losses = []
accuracyx = []
excc = []
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)  
    loss = np.mean(np.square(y_train - A2))
    training_losses.append(loss)
    dA2 = 2 * (A2 - y_train) / y_train.shape[0]  
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

    if epoch % 100 ==0:
        Z1_test = np.dot(X_test, W1) + b1
        A1_test = sigmoid(Z1_test)
        Z2_test = np.dot(A1_test, W2) + b2
        A2_test = sigmoid(Z2_test)
        predictions = np.argmax(A2_test, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracyx.append(accuracy_score(true_labels, predictions))
        excc.append(epoch)

Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

predictions = np.argmax(A2_test, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

def plot_neural_network(input_size, hidden_size, output_size):
    ip_labels=["Sepal Length","Sepal Width","Petal Length","Petal Width"]
    ip_labels = ip_labels[::-1]
    op_labels=["Setosa","Versicolor","Virginica"]
    op_labels = op_labels[::-1]
    plt.figure(figsize=(10, 5))
    plt.axis('off')  
    for i in range(input_size):
        plt.scatter([0], [i],color="black",marker=">",s=120)
        plt.text(0, i+0.15, f'{ip_labels[i]}', fontsize=12)
    for i in range(hidden_size):
        plt.scatter([0.5], [i],color="black",marker="o")
        plt.text(0.5, i+0.15, f'h{i+1}', fontsize=12)
    for i in range(output_size):
        plt.scatter([1], [i],color="black",marker=">")
        plt.text(1, i+0.15, f'{op_labels[i]}', fontsize=12)
    for i in range(input_size):
        for j in range(hidden_size):
            plt.plot([0, 0.5], [i, j], 'k-', lw=0.5)
    for i in range(hidden_size):
        for j in range(output_size):
            plt.plot([0.5, 1], [i, j], 'k-', lw=0.5)


def plot_accuracy_curve(training_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses,color="black",label="training loss")
    plt.plot(excc,accuracyx,color="black",linestyle="--" , label="accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    
plot_neural_network(input_layer_size, hidden_layer_size, output_layer_size)
plot_accuracy_curve(training_losses)

plt.show()