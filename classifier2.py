
from keras.layers import Input, Dense, Activation, \
                        Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential


def AlexNet():
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),\
                     strides=(4,4), padding='valid'))
    model.add(Activation('swish'))    
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('swish'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('swish'))
   
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('swish'))
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('swish'))
    
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    model.add(Flatten())
    
    model.add(Dense(4096, input_shape=(227*227*3,)))
    model.add(Activation('swish'))    
    
    model.add(Dense(4096))
    model.add(Activation('swish'))   
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    return model    
