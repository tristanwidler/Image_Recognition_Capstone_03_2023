Introduction¶

This is an concolutional neural network (CNN) based image recognition model created as part of my Capstone Project at Western Governors University that was trained on a subset of the Caltech 256 dataset.
Model/Project Purpose

This models purpose is to identify objects from a computer vision stream on a robotic system to aid in enhanced situational awareness. More specifically the model was created with emergency services in mind and as such the training dataset includes categories such as fire trucks and people.
Project Goal

The goal for this project is as follows:

    Acheive a validation and training accuracy of 85% or greater.

Assumptions

    A dataset exists at "./Caltech_256_Subset"
    The aforementioned dataset is of the following file structure:
        Dataset Folder
            Category_1
                Image_1
                Image_x
            Category_x
                Image_1
                Image_x

Model Attributes

    Sequential Keras Model
    Model Layer Structure (In Order):
        Rescalling()
        Conv2D()
        BatchNormalization()
        Conv2D()
        MaxPooling2D()
        BatchNormalization()
        SeperableConv2D()
        Flatten()
        Dense()
        BatchNormalization()
        AlphaDropout()
        Dense()
        BatchNormalization()
        Dense()
    Uses Selu activations with Softmax output activation
    Compiliation
        Optimizer
            SGD Optimizer
                Exponential Learning Rate Decay
            Loss
                SparseCetegoricalCrossentropy()
            Metric
                Accuracy

Results

After training on the Caltech 256 Subset with a validation split of 20% for 30 Epochs, the results are as follows:

    End Validation Accuracy:
        66.41%
    End Training Accuracy:
        78.32%

Upon analysis of the accuracy and loss histories, the model shows signs of overfitting. The project did not meet it's goal of >85% validation and training accuracies. Some possible improvements are as follows:

    Lower the initial learning rate
    Create a larger dataset to train on
    Combine the current CNN with a Support Vector Machine for a potential small dataset performance increase
    Incorporate Spacial Pyramid Pooling for potentially better feature extraction
    Incorporate automated hyperparameter tuning through Keras


