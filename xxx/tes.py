import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ✅ Step 1: Load the image
image_path = "demo.jpg"  # replace with your image
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=1)  # convert to grayscale
img = tf.image.resize(img, [50, 50])  # make it smaller for easier printing
img = tf.image.convert_image_dtype(img, tf.float32)  # scale [0,1]

# ✅ Step 2: Convert to NumPy array and scale to [0–255] for better readability
img_np = np.array(tf.squeeze(img) * 255, dtype=np.uint8)

print("🖼️ Grayscale Image Pixel Values (5×5):")
print(img_np)

# ✅ Step 3: (Optional) visualize it
plt.imshow(img_np, cmap="gray")
plt.title("5x5 Grayscale Image")
plt.axis("off")
plt.show()
