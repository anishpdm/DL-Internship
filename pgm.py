import pandas as pd
import numpy as np

height_weight_ds = pd.read_csv("data.csv")

x = height_weight_ds["height"]
y = height_weight_ds["weight"]

max_value = x.max()
min_value = x.min()

x = (x - x.min() ) / (x.max() - x.min())

w = 0
b= 0
lr=0.01

for epoch in range(1000):
    y_pred = w * x + b
    loss = np.mean( (y_pred -y)**2 )

    dw= np.mean( 2* x * (y_pred-y))
    db= np.mean( 2* (y_pred-y) )

    w = w - lr * dw
    b = b - lr * db

    print(" epoch : ", epoch ,"w : " ,w ,"b : ", b , "loss :", loss  )

input = int(input(" Enter the Height ? "))    

value_x_normalised = (input -  min_value ) / (max_value-min_value)

print( "Normalised ", value_x_normalised )
predicted_x = w * value_x_normalised + b

print("Weight of the Person who has height  ",input," is" , predicted_x)
