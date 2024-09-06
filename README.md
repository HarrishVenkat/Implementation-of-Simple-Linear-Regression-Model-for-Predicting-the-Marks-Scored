# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1:start the program

step 2: Import the standard Libraries.

stap 3:Set variables for assigning dataset values.

step 4:Import linear regression from sklearn.

step 5:Assign the points for representing in the graph.

step 6:Predict the regression for marks by using the representation of the graph.

step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

step 8:End the program

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harrish Venkat V 
RegisterNumber:  212223240049
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
## Plot:
![image](https://github.com/user-attachments/assets/f44ceaef-d202-4fb6-a665-486215509187)
![image](https://github.com/user-attachments/assets/b89f052a-82e1-48d8-aaf7-c257b9280aa0)
## Values of MSE,MAE, and RMSE:
![image](https://github.com/user-attachments/assets/0ee2a621-c45a-46cf-b1f7-2c28abbd55b0)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
