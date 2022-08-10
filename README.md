# Yoga Pose Classifier
## Table of Contents:

1. [Presentation Slides](https://github.com/leahnagy/yoga_pose_classifier/blob/38cf9e1d1d380eef35831c28ef6c1f0c85fa667c/slides_yoga.pdf)
2. [Data Preprocessing](https://github.com/leahnagy/yoga_pose_classifier/blob/6497ccf1f0a9ee57b87440900f49bb27d54f1c85/code/preprocess.ipynb)
3. [Logistic Regression Baseline Model](https://github.com/leahnagy/yoga_pose_classifier/blob/71dc45ae218e3d98554dd47832a693a818d6de02/code/baseline_logistic_regression.ipynb)
4. [Convolutional Neural Network Baseline Model](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)
5. [Transfer Learning with VGG16](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)
6. [Transfer Learning with TensorFlow's MoveNet Model](https://github.com/leahnagy/yoga_pose_classifier/blob/c75af93766dafaee5ba4949ad6ce10d773a0d5d7/code/MoveNet.ipynb)


## Abstract
Since the beginning of the Covid-19 pandemic, the popularity of AI-enabled fitness applications has exploded. Using computer vision and deep neural networks, yoga streaming applications can offer real-time feedback to correct users' alignment. The goal of this project was to develop the first phase of research to develop such a tool. This phase classified five yoga poses using neural networks and obtained optimal results utilizing transfer learning.  

## Data
The dataset contains 1,549 images of single poses split between 5 classes:
1. Warrior2
2. Downward Dog
3. Tree
4. Goddess
5. Plank

During the preprocessing stage, the images were resized to (224, 224, 3). The original dataset is located on [Kaggle](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset).

## Alogorithms<br>
*Logistic Regression Baseline Model*
<ol>
<li> Transformed images into an array of vectors with uniform image sizes of (224x224) with three color channels.</li>
<li> Split data into train/validation/test sets.</li>
<li> Reduced dimensions using PCA. </li>
<li> Trained & fit a logistic regression model to the training data.</li>
<li> Evaluated accuracy results on the validation set. </li>
</ol>
<br>

*Convolutional Neural Network*
<ol>
<li> Preprocessed images.</li>
<li> Created baseline convolutional neural network.</li>
<li> Added drop out layers. </li>
<li> Added early stopping.</li>
<li> Applied transfer learning using VGG16 & MoveNet models. </li>
</ol>
<br>

*Results*<br>

The results showed that for image classification, logistic regression does not accurately classify images due to the number of features, or pixels in this case. Even when reducing dimensionality, the results were still poor. A convolutional neural network is better suited for this task. <br><br>
While the baseline CNN performed much better than logistic regression, it would require a much larger dataset to reduce overfitting and provide accurate classifications. Combining transfer learning with convolutional neural networks provided the most accurate results. MoveNet performed the best, as it is trained to find 17 keypoints in the body to classify different body positions. 
<br><br>

**Final Model** <br>

CNN using TensorFlow's MoveNet model produced the following accuracy scores:<br>
- Training Accuracy: 0.9167
- Validation Accuracy: 0.9394
- Testing Accuracy: 0.9680

## Tools
- Numpy and Pandas for data manipulation
- Scikit-learn for modeling
- Keras and TensorFlow for deep learning modeling
- Matplotlib and Seaborn for plotting

## Communication
In addition to the slides and visuals presented, this project will be embedded on my GitHub site along with an article describing the steps of the project in detail on my personal blog. 

