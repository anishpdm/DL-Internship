import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image and convert to grayscale matrix
image_path = "demo.jpg"
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=1)  # grayscale
img = tf.image.convert_image_dtype(img, tf.float32)  # normalize [0,1]
img_np = img.numpy().squeeze()  # convert to NumPy (H, W)

print("🖼️ Image shape:", img_np.shape)
print("🔹 Sample 5x5 pixel matrix:\n", img_np[:5, :5])

# Step 2: Define a 3x3 Sobel-X filter
filter_sobel_x = np.array([
    [-1.0,  0.0,  1.0],
    [-2.0,  0.0,  2.0],
    [-1.0,  0.0,  1.0]
])

print("\n🧮 Filter matrix:\n", filter_sobel_x)

# Step 3: Manual convolution on a small 5x5 patch
region = img_np[:5, :5]
print("\n🎯 5x5 region for manual convolution:\n", region)

output_manual = np.zeros((3, 3))  # because 5x5 - 3x3 + 1 = 3x3

for i in range(3):  # slide vertically
    for j in range(3):  # slide horizontally
        patch = region[i:i+3, j:j+3]           # extract 3x3 region
        result = np.sum(patch * filter_sobel_x) # dot product
        output_manual[i, j] = result
        print(f"\n➡️ Step ({i},{j}):\nPatch:\n{patch}\nConvolved value: {result:.4f}")

print("\n✅ Final 3x3 convolution result (manual):\n", output_manual)

# Step 4: Compare with TensorFlow’s conv2d result on same area
# Convert image to proper shape: [1, H, W, 1]
img_tf = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(img_np), 0), -1)
filter_tf = tf.reshape(tf.convert_to_tensor(filter_sobel_x, dtype=tf.float32), [3,3,1,1])
filtered_tf = tf.nn.conv2d(img_tf, filters=filter_tf, strides=[1,1,1,1], padding="VALID")
filtered_tf_np = filtered_tf.numpy().squeeze()

print("\n🧩 TensorFlow 3x3 convolution output:\n", filtered_tf_np[:3,:3])
