import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image
img_path = '/Users/khader/Desktop/brainTumors/test/test/all/230.JPG'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Load the trained model
model = tf.keras.models.load_model('path_to_save_model.h5')

# Perform inference
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence_percent = np.max(predictions[0]) * 100

# Interpret the predictions
class_names = ['No Tumor', 'Tumor']
predicted_label = class_names[predicted_class]

print('Predicted Label:', predicted_label)
print('Confidence Percent:', confidence_percent)

