#!/usr/bin/env python
# coding: utf-8

# #### AML Assignment 1 - Neural Networks
# #### Kent ID - 811292363

# In[1]:


from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)


# In[2]:


train_labels[0]


# In[3]:


max([max(sequence) for sequence in train_data])


# In[4]:


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])


# In[5]:


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# In[6]:


x_train[0]


# In[7]:


y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


# In[8]:


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(32, activation="tanh"),
    layers.Dense(32, activation="tanh"),
    layers.Dense(32, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])


# In[9]:


model.compile(optimizer="adam", 
              loss="mean_squared_error",
              metrics=["accuracy"])


# #### Model Validation

# In[10]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[11]:


# Model trained with 20 epoch with batch size of 256

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=256,
                    validation_data=(x_val, y_val))


# In[12]:


history_dict = history.history
history_dict.keys()


# #### Training & Validation loss Plot

# In[13]:


import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# #### Training and validation accuracy Plot

# In[14]:


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[15]:


results = model.evaluate(x_test, y_test)


# In[16]:


results


# #### Model Structure & Results

# In[17]:


## Libraries required for setting up an environment


from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras import regularizers


# Neural network implementation using 3 layered approach with a single dropout layer

model = keras.Sequential()
model.add(Dense(32,activation='tanh')) 
model.add(Dropout(0.5))

model.add(Dense(32,activation='tanh',kernel_regularizer=regularizers.L1(0.01), activity_regularizer=regularizers.L2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


# Optimizer "adagrad" for squared error loss and accuracy metrics

model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["accuracy"])


## Splitting the train data

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# Training a neural network

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=256,
                    validation_data=(x_val, y_val))


# Training and Validation accuracy Plot

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



# Results

results = model.evaluate(x_test, y_test)
results


# ###### References :
# https://keras.io/about, https://keras.io/api/optimizers/, https://keras.io/api/losses/
# 

# In[ ]:





# #### Conclusion
# 
# As intructed, I have achieved the following in my code by building below:
# The neural network is designed with three layers, using the tanh activation function instead of relu, the Adam optimizer instead of rmsprop, applying L1 and L2 regularizers, and incorporating a dropout layer with a 50 percent dropout rate to enhance its performance and generalization capabilities.
# 
# By modeling complicated non-linear correlations in the data, the network may be able to learn deeper patterns thanks to the tanh activation function. Compared to rmsprop, the Adam optimizer is renowned for its effective memory management and flexibility to handle various data types, which can result in faster convergence and higher overall performance. By punishing excessive weights, the L1 and L2 regularizers assist minimize overfitting and encourage the network to learn more straightforward and broadly applicable patterns. Furthermore, the dropout layer forces the network to acquire more resilient features by arbitrarily removing inputs during training, which reduces overfitting. All in all, these design decisions are meant to enhance the network's capacity to efficiently learn from the data and make good generalizations
# 
# #### Final achieved results after incorporating all the changes:
# 
# Accuracy = 99.37 
# Validation accuracy = 87.20
# 
# 

# In[ ]:





# In[ ]:




