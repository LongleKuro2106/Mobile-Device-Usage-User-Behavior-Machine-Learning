import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras
from tensorflow.keras.optimizers import Adam


df = pd.read_csv('user_behavior_dataset.csv')
df.head()
df.columns
df.describe()
df.isnull().sum()

df.info()


df['Device Model'].value_counts()
df['Device Model']=df['Device Model'].replace('Xiaomi Mi 11',0)
df['Device Model']=df['Device Model'].replace('iPhone 12',1)
df['Device Model']=df['Device Model'].replace('Google Pixel 5',2)
df['Device Model']=df['Device Model'].replace('OnePlus 9',3)
df['Device Model']=df['Device Model'].replace('Samsung Galaxy S21',4)

df['Operating System'].value_counts()
df['Operating System']=df['Operating System'].replace('Android',1)
df['Operating System']=df['Operating System'].replace('iOS',0)

df['Gender'].value_counts()
df['Gender']=df['Gender'].replace('Male',1)
df['Gender']=df['Gender'].replace('Female',0)

X=df.drop(columns=['User Behavior Class'],axis=1)
y=df['User Behavior Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

