import pandas as pd
import numpy as np
import tensorflow as tf

height_weight_ds = pd.read_csv("data.csv")
x = height_weight_ds["height"]
y = height_weight_ds["weight"]

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  # preferred over input_shape in Dense
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(x_norm, y_norm, epochs=200, verbose=0)

input_height = float(input("Enter the Height? "))
input_norm = (input_height - x_min) / (x_max - x_min)

predicted_norm = model.predict(np.array([[input_norm]]))
predicted_weight = predicted_norm[0][0] * (y_max - y_min) + y_min

print(f"Weight of the person with height {input_height} is {predicted_weight:.2f}")
