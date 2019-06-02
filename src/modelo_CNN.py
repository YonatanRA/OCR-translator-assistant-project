# modelo_CNN.py 

import pandas as pd                                                   # dataframe
import keras                                                          # libreria redes neuronales 
from keras.models import Sequential                                   # construccion secuencial
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # tipos de capas neuronales
from keras.layers.normalization import BatchNormalization             # normalizacion
from keras.callbacks import ReduceLROnPlateau                         # reduccion de la tasa de aprendizaje                      
from keras.preprocessing.image import ImageDataGenerator              # procesamiento de imagenes           


 
datos=pd.read_csv('emnist-letters-train-fixed.csv')        # carga los datos


X=(datos.iloc[:,1:].values).astype('float32')   # todos los valores de los pixeles
y=datos.iloc[:,0].values.astype('int32')        # etiquetas 

X=X/255.0                           # normaliza los datos
X=X.reshape(X.shape[0], 28, 28,1)   # redimensiona para keras


batch=64                    # lote de datos
n_clases=max(y)+1           # numero de clases (etiquetas)
epocas=20                   # numero de epocas de entrenamiento
dim_entrada=(28, 28, 1)     # dimension de entrada de la red

# convierte la clase vector a clase matrices binarias (onehot)
Y=keras.utils.to_categorical(y, n_clases)


# modelo convolucional 2D
modelo=Sequential()
modelo.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=dim_entrada))
modelo.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
modelo.add(MaxPool2D((2, 2)))
modelo.add(Dropout(0.20))
modelo.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
modelo.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
modelo.add(MaxPool2D(pool_size=(2, 2)))
modelo.add(Dropout(0.25))
modelo.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
modelo.add(Dropout(0.25))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.25))
modelo.add(Dense(n_clases, activation='softmax'))

modelo.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy']) 

reduccion_tasa_apr=ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.0001)  # reduccion tasa aprendizaje


img_gen=ImageDataGenerator(
        featurewise_center=False,               # se pone la media a 0 en todo el dataset
        samplewise_center=False,                # se pone la media a 0 de la muestra
        featurewise_std_normalization=False,    # se divide entre la std del dataset
        samplewise_std_normalization=False,     # se divide cada entrada entre std
        zca_whitening=False,                    # ZCA whitening, reduccion de dimensiones (similar a PCA)
        rotation_range=15,                      # se rotan las imagenes aleatoriamente (de 0 a 180 grados)
        zoom_range=0.1,                         # zoom aleatorio 
        width_shift_range=0.1,                  # cambio aleatorio horizontal (fraccion del ancho total)
        height_shift_range=0.1,                 # cambio aleatorio vertical (fraccion de la altura total)
        horizontal_flip=False,                  # giro aleatorio horizantal
        vertical_flip=False)                    # giro aleatorio vertical
             
        
print (modelo.summary())   # resumen de la topologia de red

img_gen.fit(X)             # optimiza la imagen

opt=modelo.fit_generator(img_gen.flow(X, Y, batch_size=batch), epochs=epocas, validation_data=None,
                      verbose=1, steps_per_epoch=X.shape[0]//batch, callbacks=[reduccion_tasa_apr])  # optimiza el modelo
                            
modelo.save('modelo_cnn_letras.h5')     # guarda los pesos y estructura del modelo


