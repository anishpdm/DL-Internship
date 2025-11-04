
import tensorflow as tf
from tensorflow.keras import layers, models as model
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('cat_dog_model.h5')



# 7️⃣ Predict on new image
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('tts.jpg', target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = tf.expand_dims(img_array, 0)  # batch dimension

print(img_array)


prediction = model.predict(img_array)

print(prediction)

if prediction[0] > 0.5:
    print("🐶 Dog")
else:
    print("🐱 Cat")