
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


n_hn = 2
N_epoch = 4200

xt = np.linspace(-1,1,num=21)
yt = np.transpose([np.zeros(xt.shape)])

k=0
for x1 in xt:
    yt[k][0]=2*x1*x1+1
    k+=1

x_s = np.concatenate(([np.ones(len(xt))],[xt]),axis=0)
print(x_s)


# In[3]:


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([n_hn,2]))
v = tf.Variable(tf.random_uniform([n_hn,1]))
w = 4*(w-.5)-1
v = 4*(v-.5)-1

a = tf.matmul(w,x)
d = tf.sigmoid(a)
output = tf.matmul(d,v,transpose_a=True)

cost = tf.reduce_mean((y-output)*(y-output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


# In[4]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_epoch):
        _,c = sess.run([optimizer,cost],feed_dict = {x:x_s,y:yt})
        print("epoch:",epoch,"cost:",c)
    
    print(sess.run(w))
    print(sess.run(v))

