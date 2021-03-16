#!/usr/bin/env python
# coding: utf-8

# # AAYUSH KUMAR (TASK 1 Data Science and Business Analytics)

# # Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[160]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing given url for the datasets.

# In[161]:


url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


# This is the Data set provided to us having Duration of study in hours and Scores respectively.
# 
# 

# In[162]:


df=pd.read_csv(url)
df


# Lets create a 2d plot to have manual look whether we can predict manually or not

# In[163]:


#plotting x and y
df.plot(x='Hours',y='Scores',style='o',title='prediction perecntage')
#plt.title(prediction perecntage)
plt.ylabel('percentage scored')
plt.xlabel('hours')
plt.show()


# # Here,we are going to have a look on data set

# In[164]:


df.columns


# In[165]:


df.describe()


# In[166]:


df.info()


# In[167]:


X=df.iloc[:, :-1].values 
Y=df.iloc[:,1].values
sns.distplot(df["Hours"])


# In[168]:


sns.distplot(df["Scores"])


# # Importing sickit learn and its module to perform training

# In[169]:


from sklearn.model_selection import train_test_split


# In[170]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,random_state=101)


# In[171]:


from sklearn.linear_model import LinearRegression


# In[172]:


reg=LinearRegression()


# In[173]:


reg.fit(X_train,Y_train)


# In[174]:


print("Training Successful")


# In[175]:


print(reg.intercept_)


# In[176]:


coeff_df=pd.DataFrame(reg.coef_)
coeff_df


# # performing Linear Regression 

# In[177]:


line=reg.coef_*X+reg.intercept_
plt.title('regression line')
plt.scatter(X,Y)
plt.plot(X,line)
plt.show()


# In[178]:


print(X_test)


# Showing Real vs Predicted scores after training data

# In[179]:


Y_pred=reg.predict(X_test)


# In[180]:


df=pd.DataFrame({'Real':Y_test,'predicted':Y_pred})
df


# In[181]:


plt.scatter(Y_test,Y_pred)
plt.show()


# # Prediction for our given duaration i.e 9.25 hrs

# In[182]:


hours=[[9.25]]
my_pred=reg.predict(hours)
print("duration of study={}".format(hours))
print('my prediction={}'.format(my_pred[0]))


# # Thank you,I have learnt data exploration,linear regression,prediction in this task 1.
