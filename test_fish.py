from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lib import *
import logging
import numpy as np


# config 
batch_size = 32
epochs = 25
checkpoint_filepath = 'model/checkpoint.h5'
metrics = "val_accuracy"
path = 'fish_dataset'
retrain_model = True
logging.basicConfig(level=logging.INFO)

# data split & augmentation
train_generator, val_generator = load_data(path, split_ratio=0.25)
log_info(train_generator)

model = cnn_model(val_generator.num_classes)
if not retrain_model:
    model.load_weights(checkpoint_filepath)
model.summary()

# training phase
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             verbose=1,
                             monitor=metrics,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')

if retrain_model:
    hist = model.fit(train_generator,
                 steps_per_epoch=None,
                 epochs=epochs,
                 verbose=1,
                 validation_data=val_generator,
                 validation_steps=None,
                 callbacks=[checkpoint])
                 
# reload the model again with best weights
# model.load_weights(checkpoint_filepath)
result = model.evaluate(val_generator, return_dict=True, batch_size=1, verbose=1)
logging.info("Evaluation", result)

# reload the generator again, quick hack. TODO: reset class position back from original generator
train_generator, val_generator = load_data(path, split_ratio=0.25, shuffle=False)
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
logging.info('Confusion Matrix \n')
logging.info(confusion_matrix(val_generator.classes, y_pred))
logging.info('\n\n accuracy score \n')
logging.info(accuracy_score(val_generator.classes, y_pred))
logging.info('\n \n Classification Report \n')

classes = train_generator.class_indices
label_map = np.array(list(classes.items()))
label = label_map[:, 0].tolist()
target_names = label
logging.info(classification_report(val_generator.classes, y_pred, target_names=target_names))

# test load_img for prediction
img_path = ["fish_dataset/Abudefduf/fish5_final.png", "fish_dataset/Amphiprion/6_bg.png", "fish_dataset/Chaetodon/fish9_final.png"]

s_path = img_path[2]
image = load_img(s_path)
logging.info("Testing image: " + s_path)
input_arr = np.array([img_to_array(image)])
c_pred = np.argmax(model.predict(input_arr))
print("prediction: class " + str(c_pred))

