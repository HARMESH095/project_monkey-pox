import numpy as np

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('monkeypox_model.h5')

test_gen = ImageDataGenerator(rescale=1./255)


test_data = test_gen.flow_from_directory('./Dataset/Original Images', target_size=(256, 256), shuffle=False, class_mode='binary')


test_loss, test_accuracy = model.evaluate(test_data)



print("Test Loss:", test_loss)


print("Test Accuracy:", test_accuracy)
