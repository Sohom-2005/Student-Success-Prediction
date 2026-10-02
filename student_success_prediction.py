import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('student_success_dataset.csv')

print(df.head())

print(df.shape)

print(df.info())

#Missing Values

print(df.isnull().sum())

df['Attendance']=df['Attendance'].fillna(df['Attendance'].mean())
df['SleepHours']=df['SleepHours'].fillna(df['SleepHours'].mean())

print(df.isnull().sum().sum())

#Encoding

le=LabelEncoder()

df['Internet']=le.fit_transform(df['Internet'])
df['Passed']=le.fit_transform(df['Passed'])

print(df.head())
print(df.dtypes)

#Feature Scaling

feat=['StudyHours','Attendance','PastScore','SleepHours']

scaler=StandardScaler()

df_scaled=df.copy()

df_scaled[feat]=scaler.fit_transform(df[feat])

#Train-Test Split

X=df_scaled[feat]
y=df_scaled['Passed']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Training

model=LogisticRegression()

model.fit(X_train,y_train)

#Prediction

y_pred=model.predict(X_test)

#Classification Report

print(classification_report(y_test,y_pred))

#Confusion Matrix

conf_mat=confusion_matrix(y_test,y_pred)

#Plot (confusion Matrix)

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,fmt='d',cmap='Blues',xticklabels=['Fail','Pass'],yticklabels=['Fail','Pass'])
plt.xlabel('Predicted',fontsize=12)
plt.ylabel('Actual',fontsize=12)
plt.title('Confusion Matrix',fontsize=20)
plt.tight_layout()

plt.show()


