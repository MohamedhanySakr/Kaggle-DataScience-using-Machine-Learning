#STEP1 importing the libraries
import numpy as np
import pandas as pd
#STEP2 importing our data(train,test)using pandas
df = pd.read_csv(r'train.csv')
df_test = pd.read_csv(r'test.csv')
df.shape
df_test.shape
#STEP3 see if there are any Missing Values(NAN)
df.isnull().sum()
df_test.isnull().sum()
#Drop unneeded columns like(id)
df=df.drop(['id'],axis=1)
df_test=df_test.drop(['id'],axis=1)
df
df.thalium_scan.dtypes
df.vessels.dtypes
#STEP4 FILL NAN values
df['thalium_scan']=df['thalium_scan'].fillna(df['thalium_scan'].mean())
df['vessels']=df['vessels'].fillna(df['vessels'].mean())
#STEP5 concatenatr all the data in one DataFrame
final_df=pd.concat([df,df_test],axis=0)
final_df.shape
final_df['heart_disease']
# Remove duplicated values if it exists
final_df =final_df.loc[:,~final_df.columns.duplicated()]
final_df
#split our data
df_Train=final_df.iloc[:182,:]
df_Test=final_df.iloc[182:,:]
df_Test.drop(['heart_disease'],axis=1,inplace=True)
x_train = df_Train.drop(['heart_disease'],axis=1)
y_train = df_Train['heart_disease']
#STEP6 You can import any Machine Learning Classification Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#i will use Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train , y_train)
#Make the prediction
y_pred = lr.predict(df_Test)
y_pred
#STEP 7 Use the cross validation score to see if our model is doing well or not
from sklearn.model_selection import cross_val_score
scoring= 'accuracy'
score=cross_val_score(lr , x_train,y_train ,cv=8, n_jobs=1 ,scoring=scoring)
print(score)
# To get the average mean
round(np.mean(score)*100,2)