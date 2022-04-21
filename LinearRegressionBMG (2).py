#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model


# In[116]:


data = pd.read_csv(r'C:\Users\salsa\Downloads\student_scores_test.csv')
df=pd.DataFrame(data)


# In[99]:


plt.scatter(x,y)
plt.xlabel("Hours")
plt.ylabel("Scores")


# In[37]:


plt.figure(figsize=(10,5))
df['Scores'].plot(kind="hist")


# In[107]:


sns.pairplot(data)


# In[43]:


print(df[["Scores","Hours"]].corr())


# In[115]:


y =  df['Scores']
x =  df[['Hours']]
reg =  linear_model.LinearRegression()
reg.fit(x, y)
print("intersep:",reg.intercept_)
print("koefisien:",reg.coef_)
y_pred=reg.predict(x)
frame=pd.DataFrame({'Data Asli':y,'Data Prediksi':y_pred})
frame


# In[114]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:"+str(mean_squared_error(y_pred,y)))
print("RMSE:"+str(np.sqrt(mean_squared_error(y_pred,y))))
print("MAE:"+str(mean_absolute_error(y_pred,y)))

