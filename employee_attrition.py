#Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#Importing data
emp_file = pd.ExcelFile('employeeAttrition.xlsx')
emp_data = emp_file.parse('Existing employees')
ex_emp_data = emp_file.parse('Employees who have left')

#Cleaning data
#Employee data
emp_data.drop(['dept'],axis=1,inplace=True)
emp_data.replace({"salary":{"low":1,"medium":2,"high":3}},inplace=True)
#Ex-employee data
ex_emp_data.drop(['dept'],axis=1,inplace=True)
ex_emp_data.replace({"salary":{"low":1,"medium":2,"high":3}},inplace=True)

#Adding target column - 'employed'
emp_data['employed'] = 1
ex_emp_data['employed'] = 0

#Viewing data
emp_data.head(5)
ex_emp_data.head(5)

#Merging data
data = emp_data.append(ex_emp_data)

#Cleaning merged data
data.isnull().sum() #1045003
data.dropna(inplace=True)
data.isnull().sum() #0
data.drop(['Emp ID'],axis=1,inplace=True)

#Splitting data for training and testing
features = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']
target = ['employed']

x = data[features] #Feature data
y = data[target]  #Target data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

#Fitting model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#Accuracy test for the model
print("Accuracy ",metrics.accuracy_score(y_test,y_pred))

#Predicting with new data
model.predict(np.array([[0.7,0.2,8,100,3,0,2,3]]))