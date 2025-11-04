import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load image and convert to tensor
image_path = "demo.jpg"
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=1)  # grayscale
img = tf.image.convert_image_dtype(img, tf.float32)  # normalize [0,1]
img_matrix = tf.expand_dims(img, axis=0)  # add batch dimension: [1, H, W, 1]

print("Original image shape:", img_matrix.shape)

# Step 2: Define a Sobel-X filter for vertical edge detection (3×3)
filter_sobel_x = tf.constant([
    [[[-1.0]], [[ 0.0]], [[ 1.0]]],
    [[[-2.0]], [[ 0.0]], [[ 2.0]]],
    [[[-1.0]], [[ 0.0]], [[ 1.0]]]
], dtype=tf.float32)

print("Filter shape:", filter_sobel_x.shape)

# Step 3: Apply convolution
filtered_img = tf.nn.conv2d(
    img_matrix,
    filters=filter_sobel_x,
    strides=[1, 1, 1, 1],
    padding="SAME"
)

# ✅ Apply ReLU activation (replace negatives with 0)

# Step 4: Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(filtered_img), cmap='gray')
plt.title("After Sobel-X Filter (Raw)")
plt.axis('off')
plt.show()
