import logging
import numpy as np
import keras, os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Dropout


def cnn_model(class_unit=3):
    chanDim = -1
    model = Sequential()
    model.add(Conv2D(input_shape=(400, 700, 3), filters=16, kernel_size=(9, 9), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(7, 7), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=8, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=class_unit, activation="softmax"))

    opt = Adam(learning_rate=0.0005,
               beta_1=0.8,
               beta_2=0.9, )

    model.compile(optimizer=opt,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model
    
def log_info(train_generator):
    # Label mapping
    classes = train_generator.class_indices

    label_map = np.array(list(classes.items()))

    label = label_map[:, 0].tolist()
    label_id = label_map[:, 1].tolist()

    logging.info(label_map)
    logging.info('\n')
    logging.info(label)
    logging.info('\n')
    logging.info(label_id)
    
def load_data(path, split_ratio=0.25, shuffle=True):
    train_datagen = ImageDataGenerator(validation_split=split_ratio)

    # data flow
    data_path = 'fish_dataset'  # input dataset
    train_generator = train_datagen.flow_from_directory(path,
                                                        target_size=(400, 700),
                                                        subset='training',
                                                        shuffle=shuffle
                                                        )

    val_generator = train_datagen.flow_from_directory(path,  # same directory as training data
                                                      target_size=(400, 700),
                                                      subset='validation',  # set as validation data
                                                      shuffle=shuffle
                                                      )
    return train_generator, val_generator
