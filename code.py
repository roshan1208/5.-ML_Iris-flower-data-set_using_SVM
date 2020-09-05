#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris = load_iris()


# In[5]:


iris.keys()


# In[6]:


iris['data']


# In[39]:


iris_feat = pd.DataFrame(iris['data'],columns=iris['feature_names'])


# In[40]:


iris_feat.head()


# In[37]:


sns.set_palette('Dark2')
sns.set_style('whitegrid')
sns.pairplot(iris_feat)


# In[8]:


iris_feat.head()


# In[11]:


iris_target = pd.DataFrame(iris['target'],columns = ['Iris'])


# In[15]:


X= iris_feat
y = iris_target


# In[16]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[19]:


from sklearn.svm import SVC


# In[20]:


model = SVC()


# In[21]:


model.fit(X_train,y_train)


# In[22]:


predd = model.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report


# In[41]:


print(confusion_matrix(y_test,predd))
print('\n')
print(classification_report(y_test,predd))


# In[ ]:





# # Let's see if we can get result more accurate using Gridsearch 

# In[25]:


from sklearn.model_selection import GridSearchCV


# In[26]:


param_grid = {'C' : [0.1,1,10,100,1000], 'gamma' : [1,0.1,0.01,0.001,0.0001], 'kernel' : ['rbf']}


# In[27]:


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)


# In[28]:


grid.fit(X_train,y_train)


# In[29]:


grid.best_estimator_


# In[30]:


grid.best_params_


# In[31]:


grid_predd = grid.predict(X_test)


# In[32]:


print(confusion_matrix(y_test,grid_predd))
print('\n')
print(classification_report(y_test,grid_predd))


# In[ ]:




