
from keras.layers import Input, Dense, Activation, \
    Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
import os

train_path = "./input/train"
test_path = "./input/test"
validator_path = "./input/validation"

datagen = ImageDataGenerator(rescale=1. / 255)
train = datagen.flow_from_directory(train_path, target_size=(227, 227),
                                    class_mode='binary', batch_size=64)
test = datagen.flow_from_directory(test_path, target_size=(227, 227),
                                   class_mode='binary', batch_size=64)
validator = datagen.flow_from_directory(validator_path, target_size=(227, 227),
                                        class_mode='binary', batch_size=1)


def AlexNet():
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(11, 11),
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(
        5, 5), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(
        3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Flatten())

    model.add(Dense(4096, input_shape=(227*227*3,)))
    model.add(Activation('relu'))

    model.add(Dense(4096))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def get_category(predicted):
    return os.listdir(test_path)[np.argmax(predicted)]


if __name__ == '__main__':
    model = AlexNet()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train, epochs=10, validation_data=validator)
    loss, acc = model.evaluate(test)

    image_me = load_img(test_path + 'cat_100.jpg', target_size=(227, 227))

    img = (np.array(image_me) / 255.0).reshape(1, 227, 227, 3)
    pred = model.predict(img)
    label = get_category(pred)
    plt.xlabel(label)
    plt.imshow(image_me)
