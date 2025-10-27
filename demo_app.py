import pandas as pd
import numpy as np
import tensorflow as tf

height_weight_ds = pd.read_csv("data.csv")

x = height_weight_ds["height"]
y = height_weight_ds["weight"]

max_value = x.max()
min_value = x.min()


x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_norm = (x - x.min() ) / (x.max() - x.min())
y_norm = (y - y.min() ) / (y.max() - y.min())

model = tf.keras.Sequential( [ tf.keras.layers.Dense(1, input_shape=[1])] ) 
model.compile(optimizer='adam', loss="mse")
model.fit(x_norm,y_norm, epochs=10)


input = int(input(" Enter the Height ? "))    

value_x_normalised = (input -  min_value ) / (max_value-min_value)

print(value_x_normalised)


result= model.predict( [[value_x_normalised]] )

print(result)

# print("Weight of the Person who has height  ",input," is" , result)
