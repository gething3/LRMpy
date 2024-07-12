#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Linear Regression assignment - Python


# In[2]:


from matplotlib import pyplot as plt


# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


get_ipython().system('pip install scikit-learn')


# In[6]:


from sklearn.linear_model import LinearRegression


# In[8]:


data = pd.read_csv('regrex1-3.csv')


# In[9]:


print(data)


# In[10]:


data_y = np.array(data.y)
data_x = np.array(data.x)


# In[18]:


plt.scatter(data_x, data_y)
plt.savefig("pyLM.png")
plt.show()


# In[19]:


x = np.array(data['x']).reshape((-1,1))
y = np.array(data['y'])
model = LinearRegression()
model.fit(x, y)


# In[13]:


pred = model.predict(x)


# In[20]:


plt.plot(data.x, pred, label = 'Linear Regression', color = 'green')
plt.scatter(data_x, data_y, label = 'Data Points')
plt.title('Python Linear Regression Model')
plt.legend()
plt.savefig("pyLRM.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




