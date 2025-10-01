import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

'''for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()'''



model2 = models.Sequential()
model2.add(layers.Input(shape=(32, 32, 3)))

model2.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model2.add(layers.MaxPooling2D((2,2)))

model2.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model2.add(layers.MaxPooling2D((2,2)))
model2.add(layers.Dropout(0.30))


#Flatten
model2.add(layers.Flatten())

model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.BatchNormalization())
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(10, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 Define EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True # rollback to the best weights
)



# Train model with early stopping
history1 = model2.fit(
    x_train, y_train_categorical,
    epochs=50,
    batch_size=64,
    validation_data=(x_test, y_test_categorical),
    callbacks=[early_stop],
    verbose=2
)

'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model ANN accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model ANN loss')
plt.ylabel('loss')
plt.xlabel('epoch')'''

loss, accuracy = model2.evaluate(x_test, y_test_categorical)
print(f"Loss:{loss}")
print(f"Accuracy:{accuracy}")

model2.save("image_classifier.keras")

