# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shobika P
RegisterNumber:  212221230096

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![Screenshot (135)](https://user-images.githubusercontent.com/94508142/196183164-c17a387f-3de0-4b70-a713-bebebef24d83.png)
![Screenshot (136)](https://user-images.githubusercontent.com/94508142/196183195-7bdd416a-12e0-4d05-8ca0-044c3d3dea1d.png)
![Screenshot (137)](https://user-images.githubusercontent.com/94508142/196183238-516e540c-dcd1-4092-a769-a24904c8e4b1.png)
![Screenshot (138)](https://user-images.githubusercontent.com/94508142/196183273-cc5b6ec0-1b25-4665-9a40-a87cbf876bd1.png)
![Screenshot (139)](https://user-images.githubusercontent.com/94508142/196183323-44f41777-e29e-4c45-893c-31b2f633213d.png)
![Screenshot (140)](https://user-images.githubusercontent.com/94508142/196183370-dd88ff14-c529-46e2-9cf5-8f8253e073c0.png)
![Screenshot (141)](https://user-images.githubusercontent.com/94508142/196183398-354a5c55-5fb4-456e-b86e-19415c9a29b2.png)
![Screenshot (142)](https://user-images.githubusercontent.com/94508142/196183447-e904df1c-a78d-4499-9cb3-9a6395a6b1ee.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
