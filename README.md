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

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Shobika P
RegisterNumber:  212221230096


import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
## Original Data:
![image](https://user-images.githubusercontent.com/94508142/204136255-8cb10f6a-6c74-440f-add7-a7be6008664b.png)
## After removing:
![image](https://user-images.githubusercontent.com/94508142/204136278-512a878a-d155-48b9-9569-b8f11edb494e.png)
## Null Data:
![image](https://user-images.githubusercontent.com/94508142/204136293-5a15afff-05bd-4845-b2b3-66642d8f8a5f.png)
## Label Encoder:
![image](https://user-images.githubusercontent.com/94508142/204136330-1d33be6b-8b16-4ad7-b24d-6157af5aeeb6.png)

## Y
![image](https://user-images.githubusercontent.com/94508142/204136347-d8dee653-d011-4866-a2ab-d8e3f9aeeeff.png)
## Y_prediction:
![image](https://user-images.githubusercontent.com/94508142/204136360-7af0b407-7529-4be9-9fcf-24ca0ebc9400.png)
## Accuracy
![image](https://user-images.githubusercontent.com/94508142/204136366-0769bc13-f1d0-4388-a72d-9f4d30469c01.png)
## Cofusion:
![image](https://user-images.githubusercontent.com/94508142/204136373-349bf1c1-052c-4255-83c7-557dd0dcc0b8.png)
## Classification:
![image](https://user-images.githubusercontent.com/94508142/204136392-5c88781d-6104-449c-a5ce-3ae6689cc1d4.png)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
