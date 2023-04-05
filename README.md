# Project Introduction¶

This project uses a Jupyter Notebook to create a Keras convolutional neural network (CNN) based image recognition model with the TensorFlow framework. There is also a Python file that uses the Streamlit library to locally host a web application so a user may interract with the model. The project was initially created as part of my Capstone Project at Western Governors University and trained on a subset of the Caltech 256 dataset.
## Model/Project Purpose

This model's purpose is to identify objects from a computer vision stream on a robotic system to aid in enhanced situational awareness. More specifically the model was created with emergency services in mind and as such the training subset includes categories such as fire trucks and people.

There are currently no provisions to integrate the model with such a robotic system or computer vision stream as it was beyond the project scope.
### Project Goal

The goal of this project was simple:

    - Achieve a validation and training accuracy of 85% or greater.
## How to Use

### Fresh Installation and Startup

To start streamlit web application, assuming a fresh operating system is installed, one may do the following: (Note: This is just one method. So long as the file structure is intact, any valid use of the "streamlit run Main.py" command in the correct directory should work.)

    - Download the Repository files (The dataset and "Main.ipynb" files are optional)
    - Download the JetBrains Toolbox
        - https://www.jetbrains.com/lp/toolbox/
    - Install and open said Toolbox to install PyCharm Community
    - Install Python 3.9+ (Newer versions should work though 3.9 was used during development)
        - https://www.python.org/downloads/
    - Open PyCharm, click "open project", and navigate to the folder which contains the repository files.
    - Once the project is opened, Click the drop-down next to the run button on the top right and edit the configuration. In the "Script path:" field navigate to the Main.py file.
    - In the interpreter field select the interpreter of the installed Python version and apply. If the interpreter is not populating, apply and navigate to the File->Settings->Project-Python Interpreter setting and locate and apply your interpreter.
    - After the interpreter is applied a virtual environment which includes Pip should be created.
    - Navigate to the terminal tab at the bottom of the main project page and expand it. At the top of the tab click the dropdown arrow and open a new terminal.
    - This terminal should be using the newly created virtual environment which is denoted by a "(<venv-name>)" appearing next to the current command line. If this is not the case ensure you are in a newly created terminal, not a powershell.
    - Next, run the "pip install <package_name>" command for each of the packages below:
        - Steamlit
        - Numpy
        - Tensorflow
        - PIL
    - Once all dependancies are installed confirm your terminal is in the same directory as the Main.py file and run the command "streamlit run Main.py" to start the web application. A browser should automatically open, but a web address will be provided should you need to manually open one.
### Application Usage

The web application consists of two pages. These pages can be navigated to by opening the sidebar and selecting the desired radio button.

    - Page 1:
        - Select a sample image from the dropdown or upload your own ".jpg" file. This image will be used as input to the model on the next page.
    - Page 2:
        - Review the model structure and that the selected image is correct. Then click the "Process" button to receive the prediction output from the model.
## Project Results

After training on the Caltech 256 Subset with a validation split of 20% for 30 Epochs, the results are as follows:

    - End Validation Accuracy:
        - 66.41%
    - End Training Accuracy:
        - 78.32%

Upon analysis of the accuracy and loss histories, the model shows signs of overfitting. The project did not meet its goal of >85% validation and training accuracies. Some possible improvements are as follows:

    - Lower the initial learning rate
    - Create a larger dataset to train on
    - Combine the current CNN with a Support Vector Machine for a potential small dataset performance increase
    - Incorporate Spacial Pyramid Pooling for potentially better feature extraction
    - Incorporate automated hyperparameter tuning through Keras
