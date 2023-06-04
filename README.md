# Deep Learning & Neural Networks Project
## Yoga Pose Classifier
Created By: Leah Nagy

### Contents:
1. [Presentation Slides](https://github.com/leahnagy/yoga_pose_classifier/blob/38cf9e1d1d380eef35831c28ef6c1f0c85fa667c/slides_yoga.pdf)
2. [Data Preprocessing](https://github.com/leahnagy/yoga_pose_classifier/blob/6497ccf1f0a9ee57b87440900f49bb27d54f1c85/code/preprocess.ipynb)
3. [Baseline Model: Logistic Regression](https://github.com/leahnagy/yoga_pose_classifier/blob/71dc45ae218e3d98554dd47832a693a818d6de02/code/baseline_logistic_regression.ipynb)
4. [Baseline Model: Convolutional Neural Network](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)
5. [Transfer Learning with VGG16](https://github.com/leahnagy/yoga_pose_classifier/blob/62d3720e347140e4f9d29ae8147a2e667388e785/code/DL_baselines.ipynb)
6. [Transfer Learning with TensorFlow's MoveNet Model](https://github.com/leahnagy/yoga_pose_classifier/blob/c75af93766dafaee5ba4949ad6ce10d773a0d5d7/code/MoveNet.ipynb)


### Project Description
In the wake of the Covid-19 pandemic, the demand for AI-integrated fitness applications has surged. These applications, leveraging computer vision and deep learning, offer real-time feedback for users, thereby improving their yoga pose alignment. This project forms the initial phase of research directed at developing such a tool, specifically focusing on classifying five distinct yoga poses using neural networks. The most successful model was developed employing transfer learning.  

### Dataset
The dataset comprises 1,549 images depicting five different yoga poses:
1. Warrior2
2. Downward Dog
3. Tree
4. Goddess
5. Plank

All images were preprocessed to a uniform size of (224, 224, 3) for compatibility with neural networks. The original dataset is hosted on [Kaggle](https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset).

### Methodology

*Logistic Regression Baseline Model:*
1. Converted images into an array of vectors with uniform dimensions (224x224) and three color channels.
2. Divided data into train/validation/test sets.
3. Applied Principal Component Analysis (PCA) for dimensionality reduction.
4. Trained and fitted a logistic regression model using the training data.
5. Evaluated model performance on the validation set.

*Convolutional Neural Network*
1. Preprocessed the images.
2. Constructed a baseline convolutional neural network.
3. Introduced dropout layers to mitigate overfitting.
4. Applied early stopping to prevent overtraining.
5. Implemented transfer learning using VGG16 & MoveNet models.

### Results

Logistic regression proved inadequate for accurate image classification due to the high dimensional feature space (pixels). Despite applying dimensionality reduction techniques, the performance was subpar. Convolutional neural networks demonstrated superior performance for this task.

Although the baseline convolutional neural network outperformed logistic regression, overfitting was still a concern and could be mitigated by employing a larger dataset. Transfer learning combined with convolutional neural networks offered the highest accuracy. TensorFlow's MoveNet model, designed to identify 17 body keypoints, excelled in classifying different body positions, making it the best performer.
<br><br>

**Final Model Performance (CNN using TensorFlow's MoveNet model):** 
- Training Accuracy: 0.9167
- Validation Accuracy: 0.9394
- Testing Accuracy: 0.9680

### Tools Utilized
- Numpy and Pandas for data manipulation
- Scikit-learn for modeling
- Keras and TensorFlow for deep learning modeling
- Matplotlib and Seaborn for plotting
