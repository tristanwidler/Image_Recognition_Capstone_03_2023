import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image

"""
Author: Tristan Widler
App Version: 1.0
Used IDE and Version: PyCharm 2021.3.1
Python Version: 3.9

Overview:
    The purpose of this file is to create an interactive web application where a user can interact with the image
    recognition model stores in the "Model" folder. The user may select an image from the "Example_Images" folder or 
    upload their own ".jpg" file to be processed by said model. Additionally, the application gives the user some 
    architectural and training accuracy graphs stored in the "Static_Visual_Aid" folder. 
    
    Once an image is processed, the user will be presented with some graphs of the prediction output from the model
    generated using streamlit utilities. 

Dependencies:
    streamlit
    numpy
    tensorflow
    PIL

How to run:
    Open a terminal in your IDE or OS. Navigate to the directory with this file.
    Ensure you are using a virtual environment with the above dependencies installed.
    Then run:
        "streamlit run Main.py"
    A new browser tab/window should appear with the application. If not, the address to the locally hosted web app will 
    be given in the terminal.
"""

# Constant variables
SELECTED_IMAGE_NAME = "Selected_Image.jpg"
EXAMPLE_IMAGE_NAME_DICT = {"Door Knob": "Door_Knob_Example.jpg",
                           "Fire Extinguisher": "Fire_Extinguisher_Example.jpg",
                           "Fire Hydrant": "Fire_Hydrant_Example.jpg",
                           "Fire Truck": "Fire_Truck_Example.jpg",
                           "Ladder": "Ladder_Example.jpg",
                           "People": "People_Example.jpg",
                           "Faces": "Faces_Example.jpg"}
CLASS_NAMES = ['058.doorknob', '070.fire-extinguisher', '071.fire-hydrant', '072.fire-truck',
               '126.ladder', '159.people', '253.faces-easy-101']
STATIC_VISUAL_AID = ["Static_Visual_Aid/Train_and_Val_Accuracy_and_Loss_3_22_2023_Capstone_Model.png",
                     "Static_Visual_Aid/Model_Structure.png"]

"""
Initializes the streamlit page layout.
Layout:
    Page One (Image Upload)
    Page Two (Image Processing)
    Sidebar (Navigation)
"""
def main():
    # Sidebar setup
    sidebar = st.sidebar.container()
    starting_drop_dn_index = 7

    sidebar.header("Navigation :world_map:")
    navigation_radio_btn = sidebar.radio("Menus", ("Image Upload", "Image Processing"))

    st.title("Image Recognition Application")

    # Page selection radio button conditional statements
    """ 
    Overview:
        Handles the setup and functionality of the "Image Upload" page. 
        Creates prompts and a dropdown full of options from the "drop_dn_options" list.
        Displays selected user image or selected example image.
    """
    if navigation_radio_btn == "Image Upload":
        drop_dn_options = ["Door Knob", "Fire Extinguisher", "Fire Hydrant", "Fire Truck",
                           "Ladder", "People", "Faces", "User-Defined"]
        st.header("Image Upload Utility :camera:")
        st.info("Please upload an image to be guessed by the machine learning model.\n\n\n"
                "Once uploaded, use the sidebar to navigate to the \"Image Processing Utility.\"")

        drop_dn_result = st.selectbox("Image selection options:", drop_dn_options, starting_drop_dn_index)

        if drop_dn_result == "User-Defined":
            file = st.file_uploader("Upload a '.jpg' image file here:")

            if not file:
                st.write("Please select a file for processing.")
            if file:
                saveSelectedUserImage(file)
        else:
            saveSelectedExampleImage(drop_dn_result)

    """ 
    Overview:
        Handles the setup and functionality of the "Image Processing" page. 
        Displays image recognition model architecture as well as accuracy and loss training metrics from the 
        "Static_Visual_Aid" folder with related text.
        Displays selected user image or selected example image.
        Loads the model from the "Model" folder.
        Creates a "Process" button which feeds the "Selected_Image.jpg" to the model and displays the prediction 
        information gathered from said model via the "displayPredictionInfo(predictions)" function.
    
    Exceptions/Errors:
        OSError (Built-in):
            Thrown if the "Selected_Image.jpg" file cannot be read.
        JPGWarn (Custom):
            "Thrown" if the "Selected_Image.jpg" file is not a "JPEG" file.
    """
    if navigation_radio_btn == "Image Processing":
        st.header("Image Processing Utility :computer:")
        try:
            image = Image.open(SELECTED_IMAGE_NAME)
            if isAcceptableFormat(image):
                st.info("These are the loss and accuracy graphs for both the training and validation datasets used to "
                        "train this model. As can be seen below the model is currently suboptimal as it shows signs "
                        "of overfitting and the validation accuracy never reached more than 69%.")
                st.image(STATIC_VISUAL_AID[0])
                st.info("This image shows the architecture of the model used in this program which was created through "
                        "the TensorFlow framework.")
                st.image(STATIC_VISUAL_AID[1])

                displaySelectedImg()
                process_btn_clicked = st.button("Process Image", "process_btn")
                if process_btn_clicked:
                    with st.spinner():
                        loaded_model = tf.keras.models.load_model('Model/CapstoneModel.h5')
                        predictions = loaded_model.predict(np.expand_dims(tf.image.resize(image, (200, 200)), 0))
                    displayPredictionInfo(predictions)
            else:
                throwJPGWarn()
        except OSError:
            throwOSErrorWarn()


"""
Overview:
    Formats, and parses data from the "predictions" parameter and displays said data using the "st.bar_chart()" method.
Parameter:
    predictions:
        Contains the raw prediction output from the model used by the "Image Processing" page. 
"""
def displayPredictionInfo(predictions):
    score = tf.nn.softmax(predictions[0])
    test_dict = {}

    st.success(
        "This image most likely belongs to '{}' with a {:.2f} percent confidence."
            .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))
    )

    prediction_ls = str(predictions) \
        .replace("[[", "") \
        .replace("]]", "") \
        .replace("   ", " ") \
        .replace("  ", " ") \
        .replace("  ", " ") \
        .replace("\n", "") \
        .strip() \
        .split(" ")

    temp_str = "Here are the raw prediction values for each class of the classification model:\n\n"
    for i in range(len(CLASS_NAMES)):
        temp_str = temp_str + "{} = {}\n\n".format(CLASS_NAMES[i], prediction_ls[i])
        test_dict[CLASS_NAMES[i]] = score.numpy()[i]

    st.bar_chart(test_dict)
    st.info(temp_str)


"""
Overview:
    Opens and saves the image from the "file" parameter.
    
Parameter:
    file:
         Contains the output of the "st.file_uploader()" method.
         
Exceptions/Errors:
    OSError (Built-in):
        Thrown if the "Selected_Image.jpg" file cannot be read.
    JPGWarn (Custom):
        "Thrown" if the "Selected_Image.jpg" file is not a "JPEG" file.
"""
def saveSelectedUserImage(file):
    try:
        image = Image.open(file)
        if image.format == "JPEG":
            image.save(SELECTED_IMAGE_NAME)
            image.close()
            displaySelectedImg()
        else:
            throwJPGWarn()
    except OSError:
        throwOSErrorWarn()


"""
Overview:
    Opens and saves the example image from the "Example_Images" folder as indicated by the "drop_dn_result" parameter.

Parameter:
    drop_dn_result:
         Contains the output of the drop-down input created on the "Image Upload" page.

Exceptions/Errors:
    OSError (Built-in):
        Thrown if the "Selected_Image.jpg" file cannot be read.
    JPGWarn (Custom):
        "Thrown" if the "Selected_Image.jpg" file is not a "JPEG" file.
"""
def saveSelectedExampleImage(drop_dn_result):
    try:
        image = Image.open("Example_Images/" + EXAMPLE_IMAGE_NAME_DICT[drop_dn_result])
        if image.format == "JPEG":
            image.save(SELECTED_IMAGE_NAME)
            image.close()
            displaySelectedImg()
        else:
            throwJPGWarn()
    except OSError:
        throwOSErrorWarn()


"""
Overview:
    Displays the image at "Selected_Image.jpg" if the file exists.
"""
def displaySelectedImg():
    if os.path.exists(SELECTED_IMAGE_NAME):
        st.info("Img from uploaded file:")
        st.image(SELECTED_IMAGE_NAME)

"""
Overview:
    Displays an "st.error()" message.
"""
def throwJPGWarn():
    st.error("You must upload a \".jpg\" file. Please try again.")

"""
Overview:
    Displays an "st.error()" message.
"""
def throwOSErrorWarn():
    st.error("An OSError has been encountered. The uploaded file may not be an acceptable image format.")

"""
Overview:
    Checks if the "image" parameter is of the "JPEG" format.

Parameter:
    image:
        Contains an image gathered via the "Image.open()" method.

Returns:
    True or False
"""
def isAcceptableFormat(image):
    if image.format == "JPEG":
        return True
    return False


if __name__ == '__main__':
    main()
