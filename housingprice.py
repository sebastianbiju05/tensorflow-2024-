#A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1 bedroom house cost 100k, a 2 bedroom house cost 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

def house_model():
    xs = tf.constant([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=tf.float32)
    ys = tf.constant([[0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]], dtype=tf.float32)
    
    model = tf.keras.Sequential([
        layers.Dense(1, input_shape=(1,))
    ])
   
    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(xs, ys, epochs=1000)
    
    return model

model = house_model()
new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)
