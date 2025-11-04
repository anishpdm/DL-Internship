import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load image and convert to tensor
image_path = "mmsuper.jpg"
img = tf.io.read_file(image_path)
img = tf.image.decode_jpeg(img, channels=1)  # grayscale
img = tf.image.convert_image_dtype(img, tf.float32)  # normalize [0,1]
img_matrix = tf.expand_dims(img, axis=0)  # add batch dimension: [1, H, W, 1]

print(img_matrix)



# Step 2: Define Sobel-X and Sobel-Y filters
filter_sobel_x = tf.constant([
    [[[-1.0]], [[ 0.0]], [[ 1.0]]],
    [[[-2.0]], [[ 0.0]], [[ 2.0]]],
    [[[-1.0]], [[ 0.0]], [[ 1.0]]]
], dtype=tf.float32)

filter_sobel_y = tf.constant([
    [[[-1.0]], [[-2.0]], [[-1.0]]],
    [[[ 0.0]], [[ 0.0]], [[ 0.0]]],
    [[[ 1.0]], [[ 2.0]], [[ 1.0]]]
], dtype=tf.float32)

# Step 3: Apply convolutions
edges_x = tf.nn.conv2d(img_matrix, filters=filter_sobel_x, strides=[1, 1, 1, 1], padding="SAME")
edges_y = tf.nn.conv2d(img_matrix, filters=filter_sobel_y, strides=[1, 1, 1, 1], padding="SAME")

# Step 4: Compute edge magnitude
edges_magnitude = tf.sqrt(tf.square(edges_x) + tf.square(edges_y))

# Step 5: Display results
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(img_matrix), cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(edges_x), cmap='gray')
plt.title("Vertical Edges (Sobel-X)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(edges_magnitude), cmap='gray')
plt.title("Edge Magnitude (Combined X & Y)")
plt.axis('off')

plt.show()
