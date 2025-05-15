# All comments are in English

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
from acne import AcneDetector  
from skin import FacialSkinSegmentator
from facenet_pytorch import MTCNN
from cloth_hair_segmentator import ClothHairSegmentator
from eyeBangs import process_image as hair_eye_contact
from hairCollar import process_image as hair_collar_contact
from find_dominant_colors import process_image as find_dominant_colors
from find_dominant_colors import process_image_cloth as find_dominant_cloth_colors
from ultralytics import YOLO
from bald import BaldnessDetector
from groundingDINO import GroundingDINO, draw_boxes, compute_non_overlap_ratio


st.set_page_config(layout="wide")
st.title("AI system")

acne_model_path = "weights/yolo_acne_detection.h5"
tie_model_path = "weights/tie.pt"
skin_model_path = "weights/resnet34.onnx"
shaved_model_path = "weights/weights.best.inc.bald.hdf5"

acne_detector, skin_segmentator, face_detector, cloth_hair_segmentator, tie_detector, shaved_head_classifier = None, None, None, None, None, None

@st.cache_resource
def load_model():
    acne_detector = AcneDetector(acne_model_path)
    skin_segmentator = FacialSkinSegmentator(onnx_model_path=skin_model_path)
    face_detector = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
    cloth_hair_segmentator = ClothHairSegmentator()
    tie_detector = YOLO(tie_model_path)
    shaved_head_classifier = BaldnessDetector(shaved_model_path)
    GroundingDINO_model = GroundingDINO()
    
    return acne_detector, skin_segmentator, face_detector, cloth_hair_segmentator, tie_detector, shaved_head_classifier, GroundingDINO_model

acne_detector, skin_segmentator, face_detector, cloth_hair_segmentator, tie_detector, shaved_head_classifier, GroundingDINO_model = load_model()

TASKS = [
    "Detect face",
    "Segment facial skin",
    "Detect acne",
    "Classify beard",
    "Hair-collar contact",
    "Check hair color",
    "Extract cloth color",
    "Hair-eye contact",
    "Detect tie",
    "Detect bracelet",
    "Classify Shaved head",
    "Earring tasks"
]

# Function to handle tasks
def handle_task(image, task):
    output_image = image.copy()
    if task == "Detect acne":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            acne_condition, result_image, num_acne = acne_detector.process_single_image(
                image=image_np,  
                confidence_threshold=0.25
            )

            if acne_condition:
                result = f"Detect {num_acne} acne on the face."
            else:
                result = "No acne detected on the face."
            return result, result_image
        except Exception as e:
            return f"Error processing in acne detection: {str(e)}", output_image
    elif task == "Segment facial skin":
        try:
            image_np = np.array(image.convert("RGB"))

            result_image = skin_segmentator.predict(
                image # RGB image
            )

            result = "Segmented facial skin region."
            return result, result_image
        except Exception as e:
            return f"Error processing in facial skin segmentation: {str(e)}", output_image
    elif task == "Hair-eye contact":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            eye_bang_result, _, visualize_image = hair_eye_contact(image_np, face_detector, cloth_hair_segmentator)
            
            if eye_bang_result:
                result = "The hair touches the eyes"
            else:
                result = "The hair doesn't touch the eyes"
            return result, visualize_image
        except Exception as e:
            return f"Error processing in hair-eye contact: {str(e)}", output_image
    elif task == "Hair-collar contact":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            hair_collar_result, visualize_image = hair_collar_contact(image_np, cloth_hair_segmentator)
            
            if hair_collar_result:
                result = "Hair behind the ears touches the collar."
            else:
                result = "Hair behind the ears doesn't touch the collar."
            return result, visualize_image
        except Exception as e:
            return f"Error processing in Hair-collar contact: {str(e)}", output_image
    elif task == "Check hair color":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            hair_collar_result, visualize_image = find_dominant_colors(image_np, cloth_hair_segmentator)
            
            if hair_collar_result:
                result = "The hair color is black or brown-black."
            else:
                result = "The hair color is not black or brown-black."
            return result, visualize_image
        except Exception as e:
            return f"Error processing in hair dominant color extraction: {str(e)}", output_image
    elif task == "Extract cloth color":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            visualize_image = find_dominant_cloth_colors(image_np, cloth_hair_segmentator)
            result = "Extracted dominant cloth colors."
            
            return result, visualize_image
        except Exception as e:
            return f"Error processing in cloth dominant color extraction: {str(e)}", output_image
    elif task == "Detect tie":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            results = tie_detector.predict(source=image_np, conf=0.4, iou=0.3)  # iou=0.3 is the NMS threshold
            visualize_image = results[0].plot()
            if len(results[0].boxes) == 0:
                result = "No tie detected."
            else:
                result = f"Detected {len(results[0].boxes)} ties."
            
            return result, visualize_image
        except Exception as e:
            return f"Error processing in tie detection: {str(e)}", output_image
    elif task == "Classify Shaved head":
        try:
            image_np = np.array(image.convert("RGB"))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            pred_class, _, visualize_image = shaved_head_classifier.predict_single_image(image_np)
            if pred_class == "Yes":
                result = "The man has a shaved head."
            else:
                result = "The man doesn't have a shaved head."
            return result, visualize_image
        except Exception as e:
            return f"Error processing in shaved head classification: {str(e)}", output_image
    elif task == "Earring tasks":
        try:
            image_np = np.array(image.convert("RGB"))
            width, height = image.size
            scale = 1400 // max(height, width) + 1
            new_size = (scale * width, scale * height)
            image = image.resize(new_size, Image.BICUBIC)
            text_queries = ["earrings", "ear"]
            
            results = GroundingDINO_model.detect(image, text_queries, box_threshold=0.35, text_threshold=0.25)
            image_draw_box = draw_boxes(image, results)
            
            earring_detections = results.get('earrings', [])
            ear_detections = results.get('ear', [])
            result = ""
            res = ""
            if not earring_detections:
                result = "No earrings detected. "
            if not ear_detections:
                result += " No ears detected. "
            for i, earring_det in enumerate(earring_detections):
                earring_box = earring_det['box']
                for j, ear_det in enumerate(ear_detections):
                    ear_box = ear_det['box']
                    # print(f"\nChecking Earring {i+1} with Ear {j+1}:")
                    res = compute_non_overlap_ratio(earring_box, ear_box)
                    # print(f"Result: {result}")
                    if res == "Earring is outside the ear.":
                        break
                if res == "Earring is outside the ear.":
                        break
            
            # Number of earrings
            num_earrings = 0
            for label, detections in results.items():
                if label == "earrings":
                    num_earrings += len(detections)
            if num_earrings > 0:
                result = f"Detected {num_earrings} earrings. "
            
            result += res

            return result, image_draw_box
        except Exception as e:
            return f"Error processing in earring detection: {str(e)}", output_image
    elif task == "Detect bracelet":
        try:
            image_np = np.array(image.convert("RGB"))
            width, height = image.size
            scale = 1400 // max(height, width) + 1
            new_size = (scale * width, scale * height)
            image = image.resize(new_size, Image.BICUBIC)
            text_queries = ["bracelets"]
            
            results = GroundingDINO_model.detect(image, text_queries, box_threshold=0.3, text_threshold=0.25)
            image_draw_box = draw_boxes(image, results)
            
            num_bracelets = 0
            for label, detections in results.items():
                if label == "bracelets":
                    num_bracelets += len(detections)
                    
            if num_bracelets == 0:
                result = "No bracelet detected. "
            else:
                result = f"Detected {num_bracelets} bracelet. "

            return result, image_draw_box
        except Exception as e:
            return f"Error processing in bracelet detection: {str(e)}", output_image    


# Initialize session state variables if not already set
if "selected_task" not in st.session_state:
    st.session_state.selected_task = TASKS[0]
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "result_display" not in st.session_state:
    st.session_state.result_display = None
if "output_image" not in st.session_state:
    st.session_state.output_image = None

# Task selector
selected_task = st.selectbox("Select task:", TASKS, index=TASKS.index(st.session_state.selected_task), key="task_selector")

# If the task is changed, reset the image and result
if st.session_state.selected_task != selected_task:
    print(f"Change task to {selected_task}")
    st.session_state.selected_task = selected_task
    st.session_state.uploaded_image = None
    st.session_state.result_display = None
    st.session_state.output_image = None

# Upload image
uploaded_image = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"],
                                 key=f"image_uploader_{st.session_state.selected_task}")

# If a new image is uploaded
if uploaded_image is not None and st.session_state.uploaded_image != uploaded_image:
    st.session_state.uploaded_image = uploaded_image
    st.session_state.result_display = None
    st.session_state.output_image = None

# If an image is already uploaded
if st.session_state.uploaded_image is not None:
    try:
        input_image = Image.open(st.session_state.uploaded_image)
        # Convert RGBA to RGB if necessary
        if input_image.mode == "RGBA":
            input_image = input_image.convert("RGB")
    except Exception as e:
        st.error(f"Error when opening image: {e}")
        st.session_state.uploaded_image = None
        st.session_state.result_display = None
        st.session_state.output_image = None
        st.stop()

    st.write(f"Performing task: {st.session_state.selected_task}...")
    
    col1, col2 = st.columns(2)
    # Visualize input image
    with col1:
        if max(input_image.size) > 500:
            st.image(input_image.resize((500, 500)), caption="Input image")
        else:
            st.image(input_image, caption="Input image")

    visualize_image = None
    if st.session_state.result_display is None or st.session_state.output_image is None:
        with st.spinner("Processing......"):
            print(f"Processing task: {st.session_state.selected_task}")
            result, output_image = handle_task(input_image, st.session_state.selected_task)
            st.session_state.result_display = result
            st.session_state.output_image = output_image
            visualize_image = output_image.copy()
            if max(input_image.size) > 500:
                visualize_image = cv2.resize(visualize_image, (500, 500), interpolation=cv2.INTER_AREA)
            visualize_image = cv2.cvtColor(visualize_image, cv2.COLOR_RGB2BGR)
    
    with col2:
        st.image(visualize_image, caption="Output image")

    st.write(st.session_state.result_display)
else:
    st.write("Please upload an image to get started.")