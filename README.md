
# Plant Disease Detection For Apple Leaves

## Table of Contents

### 1. Introduction

- Background

- Problem Statement

- Objectives

- Scope

### 2. Methodology

#### 2.1 Dataset Description

#### 2 .2 Data Preprocessing

- Loading and Preparing Images
- Image Formatting and Conversion
- Feature Extraction and Selection

#### 2 .3 Machine Learning Algorithms

- Logistic Regression
- Linear Discriminant Analysis
- K Nearest Neighbours
- Decision Trees
- Random Forest
- Naïve Bayes
- Support Vector Machine

#### 2 .4 Model Training and Validation

#### 2. 5 Testing and Performance Evaluation

### 3. Implementation and Results

- Accuracy and Performance Metrics
- Comparison of Machine Learning Models
- Limitations and Challenges

### 4. Conclusion

- Summary of Achievements
- Contributions and Significance
- Future Work and Improvements

## 1. Introduction

##### 1.1 Background

Plant diseases pose significant threats to agricultural productivity, leading to
yield losses and economic hardships for farmers. Timely and accurate detection
of plant diseases is crucial for implementing effective disease management
strategies and minimizing crop damage. Traditional manual methods of disease
diagnosis can be time-consuming, subjective, and error-prone. Therefore, the
integration of technology, such as machine learning and image processing, has
emerged as a promising approach to automate and enhance plant disease
detection

##### 1.2 Problem Statement

 The main objective of this project is to develop a Plant Disease Detection
 System using machine learning algorithms and image processing techniques
 The system aims to accurately classify plant leaves as healthy or diseased by
 analyzing digital images of leaves. By automating the detection process, farmers
 and agricultural experts can promptly identify and address plant diseases
 enabling timely interventions and optimizing crop management practices

##### 1.3 Objectives

 The primary objectives of this project are as follows
 1. Develop a robust and accurate Plant Disease Detection System
 2. Implement machine learning algorithms for automated classification of plant leaves
 3. Utilize image processing techniques to extract relevant features from leaf images
 4. Evaluate the performance and accuracy of different machine learning models
 5. Provide a user-friendly interface for easy and intuitive interaction with the system

##### 1.4 Scope

 This project focuses on the detection of plant diseases specifically in apple
 leaves. The dataset used for training and testing the models is obtained from the
 Plant-Village Dataset, which contains images of healthy apple leaves and leaves
 affected by diseases such as Apple Scab, Black Rot, and Cedar Apple Rust
 The system aims to achieve high accuracy in disease classification and provide
 a practical tool for farmers and agricultural professionals to identify and manage
 plant diseases effectively. The project does not cover real-time disease detection in the field or the integration of hardware devices for image acquisition

## 2. Methodology

#### 2.1 Dataset Description

 The dataset used for this Plant Disease Detection System comprises images of
 apple leaves obtained from the Plant-Village Dataset. The dataset is organized
 into four main categories representing different classes of apple leaf conditions
 Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, and
 Apple___healthy

- Apple___Apple_scab: This category contains 2520 images, with 80%
 images assigned for training and 20% images for testing

- Apple___Black_rot: The dataset includes 2484 images in this category,
 with 80% images allocated for training and 20% images for testing

- Apple___Cedar_apple_rust: The dataset consists of 2200 images of
 leaves affected by cedar apple rust, with 80% images used for training and
 20% images for testing

- Apple___healthy: This category contains 2510 images of healthy apple
 leaves. Out of these, 80% images are designated for training, and 20% images are reserved for testing.

 The training images are utilized to teach the machine learning models to
 recognize patterns and distinguish between healthy and diseased leaves. The
 testing images are used to evaluate the performance and accuracy of the trained
 models on unseen data
 By leveraging this diverse dataset, the Plant Disease Detection System aims to
 accurately classify apple leaves as healthy or affected by diseases such as apple
 scab, black rot, or cedar apple rust. The dataset's composition enables the
 system to learn from a wide range of leaf conditions and improve its ability to
 generalize and identify plant diseases accurately

#### 2.2 Data Preprocessing

- Loading and Preparing Images
  
 In the context of the apple leaf disease detection project, the first step is to
 acquire a dataset consisting of images of apple leaves affected by different
 diseases. These images are then loaded into the system to make them accessible
 for further processing. Additionally, the images are prepared by performing
 necessary adjustments such as resizing them to a consistent resolution, cropping
 out unnecessary portions, or normalizing the color distribution
 Image Formatting and Conversion
 Once the apple leaf images are loaded, they need to be formatted and converted
 to ensure compatibility with the subsequent stages of the project. This involves
 standardizing the image format by converting them to a specific file type like
 JPEG or PNG. Furthermore, adjustments may be made to the color space
 resolution, or other image attributes to ensure consistency and facilitate accurate analysis

- # Feature Extraction and Selection
  
Feature extraction is a crucial step in detecting diseases in apple leaves. In this project, features were extracted using **EfficientNet**, a state-of-the-art deep convolutional neural network architecture known for its performance and efficiency.

While various libraries were explored, including Maustic, the final implementation relied on **EfficientNetB0** to extract high-level image features from the apple leaf images. EfficientNet automatically learns rich feature representations related to color, texture, and shape, which are vital for distinguishing between healthy and diseased leaves.

By using EfficientNet for feature extraction, we leverage its ability to capture subtle patterns and morphological cues in the leaf images. These extracted features are then used as input for traditional machine learning algorithms to perform classification.

- # Feature Selection
  
 This step involves choosing a subset of the extracted features based on their
 relevance and discriminatory power. Feature selection helps reduce the
 dimensionality of the dataset by eliminating noise or redundant information. By
 selecting the most informative features, the efficiency and accuracy of the
 disease detection model can be improved

### 2.3 Machine Learning Algorithms

 The apple leaf disease detection project utilizes a range of machine learning
 algorithms to develop an effective disease classification model. The following
 algorithms are employed

 1. Logistic Regression: Logistic Regression is used to predict the probability
 of an apple leaf being healthy or diseased based on the extracted features
 2.  XGBoost (Extreme Gradient Boosting): A powerful and scalable ensemble learning method based on gradient boosting that improves accuracy by correcting the errors of previous trees in the model sequence.
 3. K Nearest Neighbors (KNN): K Nearest Neighbors classifies apple leaves
 by comparing their features to those of the nearest neighbors in the
 feature space
 4. Decision Trees: Decision Trees use a series of if-else conditions to
 classify samples based on their features and their hierarchical
 relationships
 5. Random Forest: Random Forest is an ensemble learning method that
 combines multiple decision trees to enhance classification accuracy
 6. Naïve Bayes: Naïve Bayes is a probabilistic algorithm that calculates the
 probability of an apple leaf belonging to a particular disease class
 7. Support Vector Machine (SVM): Support Vector Machine constructs
 hyperplanes in a high-dimensional feature space to classify apple leaves

#### 2.4 Model Training and Validation

 After selecting the machine learning algorithms, the models are trained using a
 labeled dataset consisting of apple leaf images with corresponding disease
 labels. The models learn to recognize patterns and relationships between
 features and disease classes during this training phase
 To ensure the reliability and generalization of the models, a validation process
 is carried out. The trained models are evaluated using a separate validation
 dataset that was not used during the training. This helps assess the models'
 ability to accurately classify unseen apple leaf samples

#### 2.5 Testing and Performance Evaluation

 Once the models are trained and validated, they are tested on a separate testing
 dataset that contains new, unseen apple leaf images. The models predict the
 disease class for each sample, and performance evaluation metrics such as
 accuracy, precision, recall, and F1 score are calculated to measure the
 effectiveness of the models in disease detection

## Additional Evaluation: Overfitting, Dimensionality Impact
To further evaluate model generalization, the models were trained and validated under two settings:

Using all extracted features.

Using reduced features obtained via dimensionality reduction techniques such as PCA and SelectKBest.

This comparative analysis helps identify:

Whether the model is overfitting on high-dimensional data.

If reduced features provide better generalization and efficiency.

Findings:

Some models, particularly Random Forest and Logistic Regression, maintained or improved accuracy on reduced features.

This indicates that the feature reduction step not only simplifies the model but also helps mitigate overfitting.

KNN and SVM, however, were more sensitive to dimensionality changes, showing varied performance.
#### Result and Conclusion



Logistic Regression Accuracy: 0.9954
Support Vector Machine Accuracy: 0.9928
Decision Tree Accuracy: 0.7952
Random Forest Accuracy: 0.9578
XGBClassifier Accuracy: 0.9763
K-Nearest Neighbors Accuracy: 0.9578
Gaussian Naive Bayes Accuracy: 0.8806 

##### Based on the accuracy results, the Logistic Regression achieves the highest mean accuracy of 0.99.54, making it the best model among the ones

#### System Implementation
## Frontend (User Interface)
Developed using Streamlit for an intuitive, browser-based interface.

# Key features:

Image upload support (.jpg, .jpeg, .png)

Model selection dropdown for choosing different ML classifiers

Real-time display of:

Original RGB image

Converted HSV image

Segmented output showing healthy/diseased regions

Display of predicted disease class label

Integrated OpenCV for:

Image reading and format conversion (RGB, BGR, HSV)

Visualization of processed images

Backend (Processing & Prediction)
#  Image Segmentation using HSV Masks
Applied after converting to HSV and before feature extraction.

Separates leaf into:

Healthy regions (green)

Diseased regions (brown)

Uses defined HSV color ranges to create binary masks.

Produces a segmented image that removes background and highlights only relevant regions.

# Feature Extraction with EfficientNetB0
The segmented image is resized to 224×224 and preprocessed.

Passed to EfficientNetB0, a pre-trained CNN model from Keras.

Outputs a fixed 1280-dimensional feature vector using global average pooling.

Ensures meaningful and compact representation of the image.

# Prediction with Pre-trained ML Models
Extracted features are input into one of several pre-trained models, based on user selection:

Logistic Regression

K-Nearest Neighbors

Random Forest

Support Vector Machine (SVM)

Decision Tree

Naive Bayes

Models are stored as .pkl files and dynamically loaded.

# Final prediction is one of the following classes:

Apple___Apple_scab

Apple___Black_rot

Apple___Cedar_apple_rust

Apple___healthy

### Limitation and Challenges

#### Limitations

- Dataset Size: The performance of the models heavily relies on the size
 and diversity of the dataset. If the dataset used for training is small or
 lacks representation of certain classes or variations, it may limit the
 generalizability of the models

- Class Imbalance: If the dataset has imbalanced class distributions, where
 some classes have significantly fewer samples than others, it can affect
 the model's ability to accurately classify the minority classes

- Feature Extraction: . Deep learning techniques
 like convolutional neural networks (CNNs) can often perform better by
 automatically learning features directly from the images

- Model Selection: The code evaluates a set of machine learning models,
 but it may not include the best-performing model for this specific task
 Trying a wider range of models or exploring deep learning architectures
 could potentially yield better results
 Challenges

- Overfitting: Overfitting occurs when a model learns to perform well on
 the training data but fails to generalize to unseen data. It is a common
 challenge in machine learning, especially with complex models
 Regularization techniques and careful model evaluation are essential to
 mitigate overfitting

- Hyperparameter Tuning: The performance of machine learning models
 can be highly sensitive to the choice of hyperparameters. Finding the
 optimal set of hyperparameters for each model can be a time-consuming
 and computationally intensive task, requiring thorough experimentation
 and validation

- Interpretability: Some machine learning models, particularly deep
 learning models, are often considered black boxes, meaning it can be
 challenging to understand and interpret their decision-making process
 Interpretability is crucial in domains where understanding the reasoning
 behind predictions is necessary

- Scalability: The code provided may not scale well to larger datasets or
 real-world scenarios. Processing and training on large-scale datasets can
 require significant computational resources and efficient algorithms to
 handle the increased complexity and computational demands

### Summary of Achievements

The implemented image classification system has achieved the following accomplishments

- Successfully trained and evaluated multiple machine learning models for
 image classification

- Utilized feature extraction techniques and trained classifiers to classify
 images into predefined classes

- Tested the models on a given dataset and evaluated their performance
 using appropriate metrics such as accuracy, precision, and recall

### Contributions and Significance

 This project has contributed to the understanding and implementation of image
 classification using machine learning techniques. The code provides a
 framework for feature extraction and classification that can be extended and
 customized for various image classification tasks. The evaluation of different
 models helps in identifying the most suitable algorithms for the given dataset
 Overall, the project contributes to the field of image processing and pattern
 recognition

### Future Work and Improvements

While the implemented system has achieved notable results, there are several areas for future work and improvements

- Dataset Expansion: Obtaining a larger and more diverse dataset can
 improve the generalization and accuracy of the models. Increasing the
 dataset size and including more samples for each class can help address
 class imbalance issues and provide a more representative dataset

- Deep Learning Approaches: Exploring deep learning architectures, such
 as convolutional neural networks (CNNs), can potentially yield better
 performance. CNNs are capable of automatically learning relevant
 features directly from the images, removing the need for handcrafted
 feature extraction

- Hyperparameter Tuning: Further experimentation with hyperparameter
 tuning can help optimize the models' performance. Conducting a
 systematic search for optimal hyperparameters can lead to improved
 accuracy and robustness

- Ensemble Methods: Investigating ensemble methods, such as combining
 predictions from multiple models, can potentially enhance the
 classification accuracy. Techniques like bagging, boosting, or stacking
 can be explored to improve overall performance

- Real-World Deployment: Considering the deployment of the image
 classification system in real-world scenarios can present additional
 challenges, such as handling large-scale datasets, efficient inference, and
 integrating the system into existing frameworks or applications
 By addressing these future work areas and incorporating improvements, the
 image classification system can be enhanced to achieve even better
 performance, broader applicability, and increased practical value
