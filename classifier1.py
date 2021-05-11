
import keras
from keras import layers
from keras.models import Model
   
def XCeption():
    img_input = layers.Input(shape=(227, 227, 3))
   
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2), use_bias=False,)(img_input)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2), padding='same',use_bias=False)(x)
    residual = layers.BatchNormalization(axis=1)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=1)(residual)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=1)(residual)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x       

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=1)(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=1)(residual)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2, activation='softmax')(x)
    
    model = Model(img_input, x, name='xception')
    
    return model

