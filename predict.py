import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

#load the model
model = keras.models.load_model('model')

#galaxy names
features = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
            'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

#load the input image
image_path = Path("./imgs/test_img_000.jpg")

#resize the image
img = keras.utils.load_img(image_path, target_size=(69,69))
image_array = keras.utils.img_to_array(img)
image_array= tf.expand_dims(image_array, axis=0)

#predict the result
predict = model.predict(image_array).argmax(axis=1)
print("Prediction:", features[np.argmax(predict)])