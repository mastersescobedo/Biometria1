import os

import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from natsort import natsorted
from sklearn import model_selection
from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import layers, models

categories = 'faces/faces/train'
data_path = 'faces/faces'
n_classes = 151

names_categories = natsorted(os.listdir(categories))


def get_train_data(names_categories, data_path):
    X = []
    y = []
    # Se listan las categorias
    for i, c in enumerate(names_categories):
        # Sobre cada categoria se buscan lso archivos que pertenecesn
        for f in os.listdir(os.path.join(data_path, 'train', c)):
            img_path = os.path.join(data_path, 'train', c, f)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            # Se almacena la imagen como datos de entrada y al categoria como etiqueta
            X += [x]
            y += [i]

    y = np.array(y)

    return X, y


def get_test_data(names_categories, data_path):
    X_test = []
    y_test = []
    # se ordenan los datos de igual manera que las categorias
    for f in natsorted(os.listdir(os.path.join(data_path, 'test'))):
        # A partir del nombre de la imagen se toma la categoria y se le asigna el mismo numero que en los datos de train
        img_path = os.path.join(data_path, 'test', f)
        name_category = f.split('.')[0]
        label = names_categories.index(name_category)
        # Se almacena la etiqueta en el vector de etiquetas
        y_test.append(label)
        # Se habre la imagen y se guarda en el vector de muestras
        img = image.load_img(img_path, target_size=(299, 299))
        x_test = image.img_to_array(img)

        X_test += [x_test]

    y_test = np.array(y_test)

    return X_test, y_test


#####################################################################
#####################################################################
# Se cargan los datos
X, y = get_train_data(names_categories, data_path)
X_test, y_test = get_test_data(names_categories, data_path)

# Se preprocesan los datos para la que se encuentren acordes a la red ya preentrenada
X = applications.inception_v3.preprocess_input(np.array(X))
X_test = applications.inception_v3.preprocess_input(np.array(X_test))

# Se carga la red
base_model = applications.inception_v3.InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = layers.GlobalAveragePooling2D()(base_model.output)
# Add the prediction layer of size n_classes
predictions = layers.Dense(n_classes, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

# Se define que capas son entrenables
for layer in base_model.layers:
    layer.trainable = False

# Se realiza la separacion entre los datos de train y validación
x_train, x_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Se define un optimizador y se realiza la compilacion del modelo
opt = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

############### DAta augmentation #####################

datagen_train = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True
                )

datagen_train.fit(x_train)

model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=20),
                steps_per_epoch=len(x_train) / 10, epochs=10,
                validation_data=(x_val, y_val), validation_steps=20)

# # Se entrena el modelo
# model.fit(x=x_train, y=y_train, batch_size=20, epochs=1, validation_data=(x_val, y_val), verbose=2)

# Se evaluan los datos de test
print(model.evaluate(X_test, y_test))
