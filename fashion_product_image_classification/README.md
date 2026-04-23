# Fashion Product Image Classification
**Brittany D'Erasmo**

## Introduction

The purpose of this project was to train a Convolutional Neural Network (CNN) that can be used to classify images of fashion products. Using Python libraries, a Sequential CNN was trained, tested and evaluated on existing labeled images, then optimized for improvement. The result provides insight to which version of the model would perform best in real-world fashion and e-commerce applications such as visual search, product recommendation systems, trend analysis and more.

## Dataset and Preprocessing

The dataset used for this project is the Fashion Product Images (Small) dataset found on [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), which contains 44,419 images and product style data. Style attributes include things like gender, season, baseColor, masterCategory and subCategory. For this project only masterCategory is considered and set as the target variable for classification. Any malformed rows, rows with missing images or missing masterCategory were dropped. masterCategory includes seven possible values as classes: Apparel, Accessories, Footwear, Personal Care, Free Items, Sporting Goods, and Home. Due to too few samples of the latter four classes, only Apparel, Accessories and Footwear were used.

The dataset was balanced so that 6,600 samples of each class were kept. Prior to balancing, Apparel accounted for more than half the dataset which would have biased the model toward predicting the majority class. Balancing ensured each class contributed equally to training. The resulting dataset of 19,800 samples was then split into 70% training (13,860 samples), 15% validation (2,970 samples) and 15% test (2,970 samples). Images were resized to 128×128 pixels and pixel values were normalized to a range of 0 to 1 by dividing by 255.

## Model Design and Training

A sequential CNN baseline model was designed with three convolutional blocks, each consisting of Conv2D, BatchNormalization and MaxPooling2D layers, followed by a Flatten layer, a Dense layer containing 256 units, and a Dropout layer with a rate of 0.5 for regularization. Conv2D layers used ReLU activation and the output layer used Softmax activation with one node per class. It was compiled with sparse categorical crossentropy loss, the Adam optimizer and accuracy as the evaluation metric. Training ran for up to 30 epochs with early stopping based on validation loss with a patience of 5 epochs and a batch size of 32.

## Model Improvement

In a first attempt to improve the accuracy of the classifier, data augmentation was added to the baseline model. This included a horizontal flip, a rotation factor of 0.1, and a zoom factor of 0.1. The second attempt consisted of adding spatial dropout after each block of the baseline model, with increasing rates of 0.1, 0.2, and 0.3 respectively. The third model combined both data augmentation and spatial dropout. All three improvement models used the same compile and training parameters as the baseline.

## Observations

All models achieved excellent accuracy, with the improvement models producing only slight differences. The baseline model achieved 97.21% validation accuracy with a small gap of 0.79% between training and validation accuracy, indicating very slight overfitting. The augmentation model achieved slightly higher accuracy than the baseline (97.44%) and a train-val gap of -0.70% indicating no overfitting. The dropout model achieved 97.71% validation accuracy — the highest of all four models — and a train-val gap of -0.34%, indicating no overfitting. The combined augmentation and dropout model achieved 96.53% validation accuracy and the largest negative train-val gap of -1.50%, indicating the strongest generalization across all models and no overfitting. Although the dropout model achieved the highest raw accuracy, the combined model is the best overall performer as it is most balanced for accuracy and generalization, making it the most suitable model for real-world fashion product classification.

### Results Summary

| Model | Train Acc | Val Acc | Gap |
|---|---|---|---|
| Baseline | ~97.21% | 97.21% | +0.79% |
| Augmentation | ~96.74% | 97.44% | -0.70% |
| Dropout | ~97.37% | 97.71% | -0.34% |
| Combined | ~95.03% | 96.53% | -1.50% |

## Confusion Matrix

*Add confusion matrix image here — save from Colab using `plt.savefig('confusion_matrix.png')` and upload to this repo.*

![Confusion Matrix](confusion_matrix.png)

## Combined Model Actual and Predicted Classes

*Add predicted vs actual labels image here.*

![Predicted vs Actual](predicted_vs_actual.png)

## Conclusion

This project demonstrates how Convolutional Neural Networks can be used to classify images. A large dataset of close to 20,000 images were resized and normalized then used to train, test and evaluate a sequential 3-block CNN. Three improved models were evaluated, with the combined augmentation and dropout model demonstrating the strongest generalization.

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Dataset

[Fashion Product Images (Small) — Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
