import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

#splitting data
# images are 28x28  , 60000 images.
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#label names.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#normalizing the images in range 0 - 1
train_images = train_images/255.0
test_images = test_images/255.0


#creating a Sequential model.
#flattening , input layer.
# 128 neuron layer (Dense means fully connected layer)
# 80 neuron layer
# output layer
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(80, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])




model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# number of epochs means how many times we are gonna see the same image.
model.fit(train_images, train_labels, epochs=5)




test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)


#to predict,

predictions = model.predict(test_images)
# print(class_names[np.argmax(predictions[0])])


plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
