
# coding: utf-8

# In[1]:


import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt


# In[6]:


np.random.seed(101) 
tf.set_random_seed(101) 

x = np.linspace(0, 50, 50) 
x += np.random.uniform(-4, 4, 50)

y = np.linspace(-2, 2, 50)
y += np.random.uniform(-2,2,50)

n = len(x) # Number of data points 


# In[7]:


plt.scatter(x, y) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title("Training Data") 
plt.show() 


# In[8]:


X = tf.placeholder("float") 
Y = tf.placeholder("float") 
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b")

learning_rate = 0.01
training_epochs = 1000

y_pred = tf.add(tf.multiply(X, W), b) 

cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
 
init = tf.global_variables_initializer()


# In[9]:


with tf.Session() as sess: 
    sess.run(init) 
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
        # Feeding each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b) 


# In[10]:


predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 


# In[11]:


plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show()

