import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
cf.go_offline()

train= pd.read_csv('titanic_train.csv')
print(train.head())
train.isnull()
sns.set_style('whitegrid')
def impute_age(cols):
    Age=cols.iloc[0]
    Pclass=cols.iloc[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex=pd.get_dummies(train['Sex'],drop_first=True,dtype='int')
embark=pd.get_dummies(train['Embarked'],drop_first=True,dtype='int')
train=pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
X= train.drop('Survived',axis=1)
y= train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))