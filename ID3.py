import numpy as np
import pandas as pd
import math

df = pd.read_csv('./tennis.csv', names=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Tennis'])
features =[feat for feat in df]
features.remove('Play Tennis')
print(features)


class Node:
    def __init__(self):
        self.children = []
        self.value = "" #empty  str
        self.isLeaf = False
        self.pred = ""  # empty str

def Entropy(data):
    yes = 0
    no = 0
    for _, i in data.iterrows():
        if i['Play Tennis'] == 'Yes':
            yes += 1
        else:
            no += 1
    if yes == 0 or no == 0:
        return 0
    else:
        p = yes / (yes + no)
        n = no / (yes + no)
        return ((-p * math.log(p, 2)) - (n * math.log(n, 2)))

def info_gain(data, attr):
    uniq = np.unique(data[attr])
    gain = Entropy(data)
    for i in uniq:
        subda = data[data[attr] == i]
        suben = Entropy(subda)
        gain -= (float(len(subda)) / len(data)) * suben
    return gain

def ID3(data, attr):
    root = Node()
    max_gain = float('-inf')
    feat = ""
    for ft in attr:
        gain = info_gain(data, ft)
        if gain > max_gain:
            max_gain = gain
            feat = ft
    root.value = feat
    unique = np.unique(data[feat])
    for u in unique:
        subdata = data[data[feat] == u]
        if Entropy(subdata) == 0.0:
            child = Node()
            child.isLeaf = True
            child.value = u
            child.pred = np.unique(subdata['Play Tennis'])[0]
            root.children.append(child)
        else:
            hold = Node()
            hold.value = u
            newattr = attr.copy()
            newattr.remove(feat)
            child = ID3(subdata, newattr)
            hold.children.append(child)
            root.children.append(hold)
    return root

def Display(root: Node, depth=0):
    for i in range(depth):
        print('\t', end="")
    print(root.value, end=" ")
    if root.isLeaf:
        print("> ", root.pred)
    print()
    for child in root.children:
        Display(child, depth + 1)

def classify(root: Node, new):
    for child in root.children:
        if child.value == new[root.value]:
            if child.isLeaf:
                print("Prediction: ", child.pred)
                return child.pred
            else:
                return classify(child.children[0], new)

root = ID3(df, features)
print("Decision Tree: ")
Display(root)
print("\n\n")
new = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
print("Input: ", new)
classify(root, new)
print("Root Attribute: ", root.value)