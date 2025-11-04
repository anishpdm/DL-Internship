import tensorflow as tf
import matplotlib.pyplot as plt

img = tf.io.read_file("demo.jpg")
img = tf.image.decode_jpeg(img, channels=1)  # or channels=3 for color
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.expand_dims(img, axis=0) 

sobel = tf.image.sobel_edges(img) 

# Compute edge magnitude
edges_magnitude = tf.sqrt(sobel[...,0]**2 + sobel[...,1]**2)

# Display
plt.imshow(tf.squeeze(edges_magnitude), cmap='gray')
plt.title("Automatic Edge Detection (Sobel)")
plt.axis('off')
plt.show()
