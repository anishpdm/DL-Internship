import tensorflow as tf
from tensorflow.keras import layers, models


train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# 2️⃣ Create Image Datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(150, 150),
    batch_size=32)

# Normalize pixel values (0–255 → 0–1)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# 3️⃣ Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary output
])

# 4️⃣ Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5️⃣ Train Model
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 6️⃣ Optional: Save Model
model.save('cat_dog_model.h5')



# 7️⃣ Predict on new image
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('cat.0.jpg', target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = tf.expand_dims(img_array, 0)  # batch dimension

prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")