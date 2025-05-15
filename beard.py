import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import os
import time

class BeardDetector:
    """
    BeardDetector is a class for detecting beards in images using a pre-trained model.
    It uses MTCNN for face detection and a custom model for beard classification.
    Initializes the model and MTCNN detector.: 
    - model_path: Path to the pre-trained beard detection model.
    Input: image (BGR )
    Output: Dictionary with keys 'Result' and 'Confidence'.
    - 'Result': 'Beard' or 'Non Beard'
    """
    def __init__(self, model_path="beard detection.h5"):

        self.model = load_model(model_path)

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
        
    def predict(self, image):
        
        height, width, _ = image.shape
        # print(f"Image shape: {height}, {width}")

        # Convert BGR (OpenCV) to RGB for face detection
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Detect faces in the image
        boxes, _ = self.mtcnn.detect(pil_image)
        
        # Convert to grayscale for beard detection
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        
        result = {}
        if boxes is not None:
            
            x, y, x2, y2 = map(int, boxes[0])
            
            padding = int(0.2 * (y2 - y))
            padding = 0
            
            # print(f"Coordinates: {x}, {y}, {x2}, {y2}")
            
            # Extract the region of interest (ROI) for beard detection
            roi_gray = frame_gray[y:min(height, y2+padding), x:x2]
            roi_gray = cv2.resize(roi_gray, (64,64))

            # Extract lower part of face (beard region)
            roi_beard = roi_gray[35:64, 1:64]
            roi_beard = cv2.resize(roi_beard, (28,28))

            # Prepare image for prediction
            roi_beard_array = img_to_array(roi_beard)
            roi_beard_array = roi_beard_array / 255.0
            roi_beard_array = np.expand_dims(roi_beard_array, 0)

            # Predict whether the region contains a beard
            prediction = self.model.predict(roi_beard_array, verbose=0)

            # Classify as 'Beard' or 'Non Beard'
            
            res = 'Beard' if prediction[0][0] < 0.5 else 'Non Beard'
            result["Result"] = res
            result["Confidence"] = prediction[0][0]
            
            del roi_beard_array, roi_gray, roi_beard
            torch.cuda.empty_cache()

        return result

def visualize_results(image, result, output_path):
    
    label, confidence = result['Result'], result['Confidence']
    text = f"{label}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    # Initialize the BeardDetector
    detector = BeardDetector(model_path="weights/beard detection.h5")
    
    image_path = 'data/beard/4.png'  
    image = cv2.imread(image_path)
    result = detector.predict(image)
    if result:
        print(f"Results: {result}")
        visualize_results(image, result, "result.jpg")
    
    # input_folder = 'beard/in'
    # output_folder = 'beard/out'
    # log_path = 'log_beard.txt'
    # process_folder(detector, input_folder, output_folder, log_path)
