import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# data split & augmentation
train_datagen = ImageDataGenerator(validation_split=0.2)

# data flow
train_path = 'fish_dataset' # input dataset
train_generator = train_datagen.flow_from_directory(train_path, 
                                                    target_size=(224,224),
                                                    subset='training'
                                                    )

val_generator = train_datagen.flow_from_directory(train_path, # same directory as training data
                                                  target_size=(224,224),
                                                  subset='validation' # set as validation data
                                                  )
                                                  
# Label mapping
classes = train_generator.class_indices

label_map = np.array(list(classes.items()))

label = label_map[:,0].tolist()
label_id = label_map[:,1].tolist()

print(label_map)
print('\n')
print(label)
print('\n')
print(label_id)

# Global variable for training
num_of_train_samples = train_generator.n
num_of_val_samples = val_generator.n
n_classes = val_generator.num_classes
batch_size = 32
epochs = 50

checkpoint_filepath = 'model/cnn_model.h5'

# reset model
model = None
hist = None
opt = None

def cnn_model():
  chanDim = -1
  model = Sequential()
  model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(9,9),padding="same", activation="relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=128, kernel_size=(7,7), padding="same", activation="relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  #model.add(Dropout(rate=0.25))

  model.add(Conv2D(filters=256, kernel_size=(5,5), padding="same", activation="relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Conv2D(filters=512, kernel_size=(5,5), padding="same", activation="relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
  #model.add(Dropout(rate=0.25))

  model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(units=4096,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(rate=0.25))
  model.add(Dense(units=1024,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(rate=0.25))
  model.add(Dense(units=256,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(rate=0.25))
  model.add(Dense(units=64,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(rate=0.25))
  model.add(Dense(units=8,activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(rate=0.25))
  model.add(Dense(units=n_classes, activation="softmax"))

  opt = Adam(learning_rate=0.0005,
             beta_1=0.8,
             beta_2=0.9,)
  
  model.compile(optimizer=opt, 
                loss=keras.losses.categorical_crossentropy, 
                metrics=['accuracy'])  

  return model


model = cnn_model()

model.summary()

# training phase
checkpoint = ModelCheckpoint(checkpoint_filepath, 
                             verbose=1,
                             monitor='val_accuracy',
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto')

hist = model.fit(train_generator,
                 steps_per_epoch=num_of_train_samples // batch_size,
                 epochs=epochs,
                 validation_data=val_generator,
                 validation_steps=num_of_val_samples // batch_size,
                 callbacks=[checkpoint])

# testing phase

load_model = load_model(checkpoint_filepath)

Y_pred = load_model.predict_generator(val_generator, num_of_val_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix \n')
print(confusion_matrix(val_generator.classes, y_pred))
print('\n\n accuracy score \n')
print(accuracy_score(val_generator.classes, y_pred))
print('\n \n Classification Report \n')

target_names = label

print(classification_report(val_generator.classes, y_pred, target_names=target_names))
