import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                      shear_range=0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

training_set = train_datagen.flow_from_directory("train",
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("test",
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides =2))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides =2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))

model.add(tf.keras.layers.Dense(units = 13, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x = training_set, validation_data = test_set, epochs = 100)

model.save('yoga_multi.h5')

"""
from keras.models import load_model
my_model = load_model('yoga_multi.h5')
"""

import numpy as np
from keras.preprocessing import image
test_img = image.load_img('Single_Prediction/astavakrasana.png', target_size = (64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
result = model.predict(test_img)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'adho mukha svanasana'

elif result[0][1] == 1:
    prediction = 'adho mukha vriksasana'
    
elif result[0][2] == 1:
    prediction = 'agnistambhasana'
    
elif result[0][3] == 1:
    prediction = 'ananda balasana'
    
elif result[0][4] == 1:
    prediction = 'anantasana'

elif result[0][5] == 1:
    prediction = 'anjaneyasana'
    
elif result[0][6] == 1:
    prediction = 'ardha bhekasana'

elif result[0][7] == 1:
    prediction = 'ardha chandrasana'
    
elif result[0][8] == 1:
    prediction = 'ardha matsyendrasana'
    
elif result[0][9] == 1:
    prediction = 'ardha pincha mayurasana'
    
elif result[0][10] == 1:
    prediction = 'ardha utanasana'
    
elif result[0][11] == 1:
    prediction = 'ashtanga namaskara'
    
elif result[0][12] == 1:
    prediction = 'astavakrasana'
    
print(prediction)