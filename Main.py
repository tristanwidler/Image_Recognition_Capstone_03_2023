import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image

SELECTED_IMAGE_NAME = "Selected_Image.jpg"
EXAMPLE_IMAGE_NAME_DICT = {"Door Knob": "Door_Knob_Example.jpg",  # Accurate Prediction Achieved
                           "Fire Extinguisher": "Fire_Extinguisher_Example.jpg",  # Accurate Prediction Achieved
                           "Fire Hydrant": "Fire_Hydrant_Example.jpg",  # Accurate Prediction Achieved
                           "Fire Truck": "Fire_Truck_Example.jpg",  # Accurate Prediction Achieved
                           "Ladder": "Ladder_Example.jpg",  # Accurate Prediction Achieved
                           "People": "People_Example.jpg",  # Accurate Prediction Achieved
                           "Faces": "Faces_Example.jpg"}  # Accurate Prediction Achieved
CLASS_NAMES = ['058.doorknob', '070.fire-extinguisher', '071.fire-hydrant', '072.fire-truck',
               '126.ladder', '159.people', '253.faces-easy-101']
STATIC_VISUAL_AID = ["Static_Visual_Aid/Train_and_Val_Accuracy_and_Loss_3_22_2023_Capstone_Model.png",
                     "Static_Visual_Aid/Model_Structure.png"]


def main():
    # Page Setup
    sidebar = st.sidebar.container()

    drop_dn_options = ["Door Knob", "Fire Extinguisher", "Fire Hydrant", "Fire Truck",
                       "Ladder", "People", "Faces", "User-Defined"]

    starting_drop_dn_index = 7

    sidebar.header("Navigation :world_map:")
    navigation_radio_btn = sidebar.radio("Menus", ("Image Upload", "Image Processing"))
    st.title("Image Recognition Application")

    # Page selection conditional statements
    if navigation_radio_btn == "Image Upload":
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


def saveSelectedExampleImage(drop_dn_res):
    try:
        image = Image.open("Example_Images/" + EXAMPLE_IMAGE_NAME_DICT[drop_dn_res])
        if image.format == "JPEG":
            image.save(SELECTED_IMAGE_NAME)
            image.close()
            displaySelectedImg()
        else:
            throwJPGWarn()
    except OSError:
        throwOSErrorWarn()


def displaySelectedImg():
    if os.path.exists(SELECTED_IMAGE_NAME):
        st.info("Img from uploaded file:")
        st.image(SELECTED_IMAGE_NAME)


def throwJPGWarn():
    st.error("You must upload a \".jpg\" file. Please try again.")


def throwOSErrorWarn():
    st.error("An OSError has been encountered. The uploaded file may not be an acceptable image format.")


def isAcceptableFormat(image):
    if image.format == "JPEG":
        return True
    return False


if __name__ == '__main__':
    main()
