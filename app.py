import os
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.utils import img_to_array
from keras import backend as K

# Set seed for reproducibility
import random
seed = 9
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Clear Keras session
K.clear_session()


# Models dictionary - maps display names to model filenames
models = {
    "Logistic Regression": "logistic_regression_model",
    "K Neighbors Classifier": "k-nearest_neighbors_model",
    "Random Forest Classifier": "random_forest_model",
    "SVM": "support_vector_machine_model",
    "Decision Tree Classifier": "decision_tree_model",
    "Naive Bayes": "gaussian_naive_bayes_model"
}

# Disease labels mapping
labels_mapping = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy"
}

def rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr_hsv(image):
    """Convert BGR image to HSV format"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def img_segmentation(rgb_img, hsv_img):
    """Segment the image to identify healthy and diseased regions"""
    try:
        # Define color ranges for healthy (green) and diseased (brown) regions
        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        lower_brown = np.array([10, 0, 10])
        upper_brown = np.array([30, 255, 255])
        
        # Create masks for healthy and diseased regions
        healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
        
        # Combine masks
        final_mask = healthy_mask + disease_mask
        final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
        
        return final_result
    except Exception as e:
        st.error(f"Error in image segmentation: {str(e)}")
        return None

def extract_features(image, base_model):
    """Extract features from image using EfficientNetB0"""
    try:
        # Preprocess image
        cnn_input = cv2.resize(image, (224, 224))
        cnn_input = img_to_array(cnn_input)
        cnn_input = np.expand_dims(cnn_input, axis=0)
        cnn_input = preprocess_input(cnn_input)
        
        # Extract features
        features = base_model.predict(cnn_input, verbose=0)[0]
        st.write(f"Number of features extracted: {len(features)}")
        return features
    except Exception as e:
        st.error(f"Error in feature extraction: {str(e)}")
        return None

def load_model(model_name):
    """Load the selected model"""
    try:
        model_file = models[model_name]
        model_path = os.path.join('ml_project_models', 'ml_project', f"{model_file}.pkl")
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title(" Disease Detection for Apple Leaves")
    st.subheader("Upload an image and select a model to detect the disease")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image of an apple leaf", type=["png", "jpg", "jpeg"])

    try:
        # Load EfficientNetB0 model
        base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling='avg')
    except Exception as e:
        st.error(f"Error loading EfficientNetB0 model: {str(e)}")
        return

    if uploaded_file is not None:
        try:
            # Read and display uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            if image is None:
                st.error("Error: Could not read the uploaded image")
                return
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Model selection
            model_name = st.selectbox("Select a model for disease detection", list(models.keys()))

            if st.button("Detect"):
                # Process image
                rgb_image = rgb_bgr(image)
                hsl_image = bgr_hsv(rgb_image)
                segmented_image = img_segmentation(rgb_image, hsl_image)
                
                if segmented_image is None:
                    return

                # Display processed images
                st.subheader("Processed Images")
                fig = plt.figure(figsize=(15, 5))
                
                # RGB Image
                plt.subplot(1, 3, 1)
                plt.imshow(rgb_image)
                plt.axis('off')
                plt.title("RGB Image")

                # HSL Image
                plt.subplot(1, 3, 2)
                plt.imshow(hsl_image)
                plt.axis('off')
                plt.title("HSL Image")

                # Segmented Image
                plt.subplot(1, 3, 3)
                plt.imshow(segmented_image)
                plt.axis('off')
                plt.title("Segmented Image")

                st.pyplot(fig)

                # Extract features
                features = extract_features(segmented_image, base_model)
                if features is None:
                    return

                # Load model
                loaded_model = load_model(model_name)
                if loaded_model is None:
                    return

                # Debug information
                st.write(f"Model type: {type(loaded_model).__name__}")
                
                # Make prediction
                try:
                    prediction = loaded_model.predict([features])
                    predicted_label = labels_mapping[prediction[0]]
                    
                    # Display result
                    st.subheader("Disease Prediction")
                    st.write("Predicted Label: ", predicted_label)
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.error("Please make sure you're using the correct model files trained on 1280 features.")
                    return

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return

if __name__ == '__main__':
    main()
