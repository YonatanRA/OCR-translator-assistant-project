![IronHack Inverse](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/ironhack.png)

# Mamba  (OCR-Translator-Assistant)

**Yonatan Rodriguez**

**Data Analytics Bootcamp**



## Overview

The first idea for this project was the OCR translator, using a custom logistic regression (softmax). The model reads handwritten characters, changes them to string format and then translates. This approach achieves an accuracy of 70%, with 26 labels, the uppercases, and 69.6% with 62 labels, uppercases, lowercases and numbers.

The next step has been the implementation of a convolutional neural network (CNN) as the predict model. The accuracy grows up until 85% for 62 labels and 94% for 26 labels.

With this model, the challenge was build a voice assistant, called Mamba.



##
## Data

* [EMNIST Dataset](https://www.kaggle.com/crawford/emnist)

The dataset is the extended MNIST, with numbers, uppercases and lowercases. For some reason, the images are rotated and fliped, so first is needed fix the data. The script fix_data.py has this purpose.

Once the data is fixed, it's possible to train the models.



##
## Softmax model

The softmax model is the logistic regression multilabel, using the sigmoid function. The minimize from scipy is used to optimize the cost function. The model and the training are in clasificador_letras_softmax.py script. The plots of confussion matrix are showing the results. First with 26 labels:

![Confussion M Soft 26](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20Softmax%20(letters).png)


Now, 62 labels:

![Confussion M Soft 62](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20Softmax.png)




##
## CNN model

The model and the training are in modelo_CNN.py script.The convolutional network has the following architecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 12, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 6, 6, 128)         73856     
_________________________________________________________________
dropout_3 (Dropout)          (None, 6, 6, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               589952    
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 27)                3483      
=================================================================
Total params: 732,795
Trainable params: 732,539
Non-trainable params: 256

The input shape is (28,28,1) and the output layer is equal to the number of labels. The results, like above, are:

![Confussion M CNN 26](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20CNN%20(letters).png)

![Confussion M CNN 62](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20CNN.png)
























