# Yoga Pose Classifier
## Table of Contents:
1. [Data Preprocessing](https://github.com/leahnagy/yoga_pose_classifier/blob/6497ccf1f0a9ee57b87440900f49bb27d54f1c85/code/preprocess.ipynb)
2. [Logistic Regression Baseline Model](https://github.com/leahnagy/yoga_pose_classifier/blob/71dc45ae218e3d98554dd47832a693a818d6de02/code/baseline_logistic_regression.ipynb)
3. [Convolutional Neural Network Baseline Model](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)
4. [Transfer Learning with VGG16](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)

## Abstract
Since the beginning of the Covid-19 pandemic, the popularity of AI-enabled fitness applications has exploded. Using computer vision and deep neural networks, yoga streaming applications can offer real-time feedback to correct users' alignment. The goal of this project was to develop the first phase of research to develop such a tool. This phase classified five yoga poses using neural networks and obtained optimal results utilizing transfer learning.  

## Data
The dataset contains 1,549 images of single poses split between 5 classes:
1. Warrior2
2. Downward Dog
3. Tree
4. Goddess
5. Plank

During the preprocessing stage, the images were resized to (224, 224, 3). The original dataset is located on [Github](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset).

## Alogorithms
*Logistic Regression Baseline Model*
1. Transformed images into an array of vectors with uniform image sizes of (224x224) with three color channels.
2. Split data into train/validation/test sets.
3. Reduced dimensions using PCA.
4. Trained & fit a logistic regression model to the training data.
5. Evaluated accuracy results on the validation set.
<br><br>
*Convolutional Neural Network Baseline Model*
