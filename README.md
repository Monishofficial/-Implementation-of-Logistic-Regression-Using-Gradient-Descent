# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MONISH N
RegisterNumber:  212223240097
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")

dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])

y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)

print(y_prednew)
```
## Output:
#### Read the file and display:
![image](https://github.com/user-attachments/assets/c299fc9d-30be-4764-97a9-ffd8908edc1e)
#### Categorizing columns:
![image](https://github.com/user-attachments/assets/9f00a201-6fc1-4d79-9973-9a27d3d32143)
#### Labelling columns and displaying dataset:
![image](https://github.com/user-attachments/assets/fceba170-d2d5-4a74-9e72-a0052d7302e7)
#### Display dependent variable:
![image](https://github.com/user-attachments/assets/4946a67d-9b34-47d8-9d7f-d73a46f0dbcd)
#### Printing accuracy:
![image](https://github.com/user-attachments/assets/508a168f-eff9-42bb-a743-25a172e5b985)
#### Printing Y:
![image](https://github.com/user-attachments/assets/f0291f56-e85a-492e-8ad2-2e4ffe9b3332)
#### Printing y_prednew:
![image](https://github.com/user-attachments/assets/4af07ffd-4e00-48b4-bc86-a5bf8688ffd2)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

