import os
import tensorflow as tf
import numpy as np
counter=0
counterTum=0
# Load the trained model
model = tf.keras.models.load_model('path_to_save_model.h5')

# Define the class labels
class_labels = ['No Tumor', 'Tumor']

# Set the path to the directory containing the images
directory_path = "/Users/khader/Desktop/brainTumors/test/test/tumor"  # Replace with the path to your image directory

# Iterate over the files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(image)

        # Get the predicted class and confidence
        class_index = np.argmax(predictions[0])
        class_confidence = predictions[0][class_index]

        # Print the prediction result
        predicted_class = class_labels[class_index]
        confidence_percent = class_confidence * 100
        if predicted_class=='Tumor':
            counterTum+=1
            
        counter +=1
        
        print(f"The image {filename} is classified as '{predicted_class}' with a confidence of {confidence_percent:.2f}%")
accuracy=((counter-counterTum)/counter)*100
print(100-accuracy,'%')
# '/Users/khader/Desktop/brainTumors/test/test'
