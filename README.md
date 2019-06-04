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

The softmax model is the logistic regression multilabel, using the sigmoid function. The minimize from scipy is used to optimize the cost function. The plots of confussion matrix are showing the results. First with 26 labels:

![Confussion M Soft 26](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20Softmax%20(letters).png)


Now, 62 labels:

![Confussion M Soft 62](https://github.com/YonatanRA/OCR-translator-assistant-project/blob/master/images/Confusion%20Matrix%20Softmax.png)




##
## CNN model














