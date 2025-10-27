import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_path = "demo.jpg"  
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=1)  # convert to grayscale
img = tf.image.resize(img, [1000, 1000]) 
img = tf.image.convert_image_dtype(img, tf.float32) 

img_np = np.array(tf.squeeze(img) * 255, dtype=np.uint8)

print("🖼️ Grayscale Image Pixel Values (5×5):")
print(img_np)

plt.imshow(img_np, cmap="gray")
plt.title("5x5 Grayscale Image")
plt.axis("off")
plt.show()
