
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from classifier1 import XCeption
from classifier2 import AlexNet

train_path = "./input/train"
test_path = "./input/test"
validator_path = "./input/validation"

datagen = ImageDataGenerator(rescale=1. / 255)
train = datagen.flow_from_directory(train_path, target_size = (227, 227),
                                       class_mode='binary', batch_size=64)
test = datagen.flow_from_directory(test_path, target_size = (227, 227),
                                       class_mode='binary', batch_size=64)
validator = datagen.flow_from_directory(validator_path, target_size = (227, 227),
                                       class_mode='binary', batch_size=1)


"""AlexNet Classifier model"""
model = AlexNet()

"""XCeption Classifier model"""
#model = XCeption()

model.compile(optimizer='adam', loss='binary_crossentropy', 
               metrics=['accuracy'])
model.fit(train, epochs=5)
evalaution = model.evaluate(test)
predictions = model.predict(validator) 
print(predictions[100])

def get_category(predicted):
    return os.listdir(validator_path)[np.argmax('abc')]

plt.label(get_category(predictions[100]))
plt.imshow(validator[100][0][0])