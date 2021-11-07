import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as TensorFlowKeras
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

base_model = MobileNet(input_shape=(224,224,3), include_top= False)

for layer in base_model.layers:
    layer.trainable = False

model = TensorFlowKeras.models.Model(base_model.input, TensorFlowKeras.layers.Dense(activation='softmax', units=4)(TensorFlowKeras.layers.Flatten()(base_model.output)))
model.compile(optimizer='adagrad', metrics=['accuracy'], loss= categorical_crossentropy)

train_data = ImageDataGenerator(rescale=1.0/255.0,horizontal_flip=True,zoom_range=0.2,shear_range=0.2).flow_from_directory(directory= "dataset/train", target_size=(224,224), shuffle=True, batch_size=32)
train_data.class_indices
val_data = ImageDataGenerator(rescale=1.0/255.0,horizontal_flip=True,zoom_range=0.2,shear_range=0.2).flow_from_directory(directory= "dataset/test", target_size=(224,224),shuffle=True,batch_size=32)

modelH = model.fit(train_data,epochs=20,validation_data=val_data,callbacks=[EarlyStopping(monitor='val_accuracy',min_delta= 0.01,patience= 5,verbose= 1,mode='auto'),ModelCheckpoint(filepath="model1.h5",monitor='val_accuracy',verbose=1,save_best_only=True,mode='auto')])
"""
To plot accuracy and loss as epochs progress. Not required for training. Solely for visualization purposes.
model = load_model("model.h5")

modelHistory=modelH.history
modelHistory.keys()

plt.plot(modelHistory['accuracy'])
plt.plot(modelHistory['val_accuracy'],c = "red")
plt.title("acc vs v-acc")
plt.show()

plt.plot(modelHistory['loss'])
plt.plot(modelHistory['val_loss'],c="red")
plt.title("loss vs v-loss")
plt.show()
"""
