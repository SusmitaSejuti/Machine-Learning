#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


m=1.2
c=2

#Line Plot
X = np.array([i for i in range(-1,150)])
Y_old = m * X + c
Y = np.random.normal(0, 3, X.shape[0]) + Y_old


# In[7]:


plt.plot(X, Y_old, label = "actual")
plt.legend()
plt.show()
plt.scatter(X,Y, label="dataset")
plt.legend()
plt.show()


# In[10]:


def model(X,w,b):
    return X*w +b


# In[11]:


model(X, 1 ,2)


# In[13]:


def loss(w, b): # model Parameter 
    mse = (1/Y.shape[0]) * np.sum((Y - model(X, w, b)) ** 2)
    return mse


# In[50]:


loss( 2 ,1)


# In[51]:


def grad_desc(epoch = 10, lr = 1): # hyper parameter
    #Initialization
    w = 0 
    b = 0
    for i in range(epoch):
        
        #calculate derivatives against parameters
        
        dw =  (1/Y.shape[0]) * np.sum(-2* X* (Y - model(X, w, b)))
        
        db = (1/Y.shape[0]) * np.sum(-2* (Y - model(X, w, b)))
        
        #Update Parameters
        w = w - lr * dw
        b = b - lr * db
        
        print(i+1,". Loss: ", loss(w,b), ", w:", w ,",b:" ,b)
        
        #plot the lines
        Y_hat = w * X + b
        plt.plot(X, Y_hat, label = "Epoch: " +str(i+1))
        
    plt.scatter(X, Y , label = "dataset")
    plt.legend()
    plt.show()
    return w, b
        


# In[54]:


w, b = grad_desc(10, 0.00001)
# epoch barate thakle loss komte tahkbe...But ekta somoi pore komar rate fix hoye jabe/ onk kome jabe
#


# In[53]:


for i in range(90, 150):
    w = i / 100
    b = 2
    
    l = loss(w ,b)
    plt.scatter(w, l)
    #plt.plot(w, l)
    
plt.plot([i/100 for i in range(90,150)], [10 for i in range(90,150) ])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




