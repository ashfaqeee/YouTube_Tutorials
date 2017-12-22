
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def shiftVector(rowV,n):
    t=[]
    for i in range(0,n,1):
        pre_zeros = np.zeros([1,i])
        post_zeros = np.zeros([1,n-i-1])
        t_row = np.append(pre_zeros,rowV)
        t_row = [np.append(t_row,post_zeros)]
        if i == 0:
            t = t_row
        else:
            t = np.append(t,t_row,axis=0)
    return t


# In[3]:


tx = np.arange(0,10*3.14159,0.01)
ty = np.arange(1,10*3.14159+1,0.01)
trainX = np.sin(tx)
trainY = np.sin(ty)


# In[4]:


plt.plot(trainX)
plt.plot(trainY)
plt.show()
print(trainX)


# In[5]:


trainX = shiftVector(trainX,5)
print(trainX)


# In[6]:


trainX = trainX.T
print(trainX[0:10])


# In[7]:


trainY = np.append(trainY,np.zeros([1,5-1]))
print(trainY)


# In[8]:


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=5))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=20, batch_size=20)


# In[10]:


Y_hat = model.predict(trainX)

plt.plot(Y_hat)
plt.plot(trainY)
plt.show()

