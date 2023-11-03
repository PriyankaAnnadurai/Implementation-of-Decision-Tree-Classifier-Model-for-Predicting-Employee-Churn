# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module 
   from sklearn.

   

## Program:
```py
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PRIYANKA.A
RegisterNumber:  212222230113
*/

import pandas as pd
data=pd.read_csv('/content/Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### Data Head:




![276257614-8850104f-05c5-44ac-af22-ad93fb88abe7](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/0946f977-5e7b-4b84-b89c-622e5551bdb3)




### Dataset Info:




![276257804-9b2b5660-3d11-4747-8244-35f4ff1b94a8](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/0428b5d7-fd26-430e-9326-da4784e15454)





### Null dataset:




![276258036-40aea425-2092-4bed-852f-e618627784bc](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/edd0cba8-44be-4731-894e-97e92db5b453)





### Values Count in Left Column:




![276258214-f9b639ea-ba36-4e3f-841c-16b4dce87b10](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/9b6db5b9-e7a5-478c-b5e8-8e26cc94c4ca)





### Dataset transformed head:





![276258309-970953ab-f37c-4739-83e2-1c0d5cb0d8d9](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/00ad6187-c10c-4892-8287-6c3865bb2827)





### x.head():




![276258426-34760264-f038-4f2b-83ee-a56128efada6](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/38996680-2cc4-4b8d-a8e9-cb3240a8b2be)





### Accuracy:





![276258721-1d1a55c7-8c3e-46eb-a5a1-2b26eaeb5caa](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/d264437f-050b-41bf-af2d-08c7d097738b)





### Data Prediction:





![276258763-0443eb09-e704-4d1d-bbdb-99dcdd722377](https://github.com/PriyankaAnnadurai/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118351569/616709ce-ff25-4add-82e4-2f640d29209a)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
