from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
import keras
import timeit


# dimensions of our images.
img_width, img_height = 178, 218

train_data_dir = '/Data/train'
validation_data_dir = '/Data/validation'
nb_train_samples = 206
nb_validation_samples = 206
epochs = 50
batch_size = 15

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(0.0009),
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


model.summary()
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    classes=['men', 'women'],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    classes=['men', 'women'],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle = True)


scores = model.evaluate(validation_generator)
print('validation loss:', scores[0])
print('validation accuracy:', scores[1])

result = timeit.timeit("'-'.join([str(i) for i in range(100)])", number=20000)
print(result)
   
model.save_weights('first_try.h5')

#after adding   rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2, to the image generator we got a lower accuracy
#1 batchnormalization :   validation loss: 0.566597580909729 validation accuracy: 0.8058252334594727
#deleting second layer: validation loss: 0.6727764010429382 validation accuracy: 0.5873786211013794
# adding 1 more section: validation loss: 0.6243094801902771 validation accuracy: 0.8252426981925964
#higher batch size = worst accuracy and higher loss
#batch size = 8 validation loss: 0.41485774517059326 validation accuracy: 0.8883495330810547
# when we added the dropout to each layer the accuracy got lower

# Plotting our loss charts
import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Plotting our accuracy charts
import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()