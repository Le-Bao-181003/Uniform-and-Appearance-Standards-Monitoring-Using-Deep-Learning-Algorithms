import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
import mediapipe as mp
import torch
from cloth_hair_segmentator import ClothHairSegmentator
import time
import datetime
import math

def detect_face(image, face_detector):
    face_box, _ = face_detector.detect(image)   
    bounding_box = None
    if face_box is not None:
        bounding_box = face_box[0].tolist()   
        print(f"Face bounding box: {bounding_box}")
    return image, bounding_box

def find_eye_landmarks(image, face_box):
    if face_box is None:
        face_crop = image
        xmin, ymin, xmax, ymax = 0, 0, image.shape[1], image.shape[0]
    else:
        xmin, ymin, xmax, ymax = map(int, face_box)
        face_crop = image[ymin:ymax, xmin:xmax]
    imgRGB = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    # imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # face_crop = image
    
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
    LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
    RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    
    landmarks = {}
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            landmarks["left_eye_landmarks"] = []
            landmarks["right_eye_landmarks"] = []

            for i, lm in enumerate(faceLms.landmark):
                h, w, _ = face_crop.shape
                x, y = int(lm.x * w), int(lm.y * h)

                if i in LEFT_EYE_LANDMARKS:
                    landmarks["left_eye_landmarks"].append((x+xmin, y+ymin))
                    # landmarks["left_eye_landmarks"].append((x, y))
                if i in RIGHT_EYE_LANDMARKS:
                    landmarks["right_eye_landmarks"].append((x+xmin, y+ymin))
                    # landmarks["right_eye_landmarks"].append((x, y))
        all_eye_landmarks = landmarks["left_eye_landmarks"] + landmarks["right_eye_landmarks"]
        
        # Find index of 362 in left_eye and 133 in right_eye
        index_362_in_left = LEFT_EYE_LANDMARKS.index(362)
        index_133_in_right = RIGHT_EYE_LANDMARKS.index(133)

        # I want all_eye_landmarks = left_eye_landmarks + right_eye_landmarks
        point_362 = all_eye_landmarks[index_362_in_left]
        point_133 = all_eye_landmarks[len(LEFT_EYE_LANDMARKS) + index_133_in_right]

        # Calculate the distance using Euclidean algo
        dx = point_133[0] - point_362[0]
        dy = point_133[1] - point_362[1]
        threshold = ((dx ** 2 + dy ** 2) ** 0.5) // 4  
        print(f"Threshold: {threshold}")
        
    return all_eye_landmarks, threshold

def visualize_eye_landmarks_hair(image, eye_landmarks, hair_mask):
    
    height, width, _ = image.shape
    scale = ( max(height, width) + 500 ) // 500

    segmented_hair = cv2.bitwise_and(image, image, mask=hair_mask)
    for landmark in eye_landmarks:
        cv2.circle(segmented_hair, landmark, scale, (0, 255, 0), -1)
    
    black_image = np.zeros_like(image)
    black_image = segmented_hair
    
    return black_image

def check_eyes_bang_collission2(image, eye_landmarks, hair_mask, threshold):
    threshold = int(threshold)
    if eye_landmarks is None:
        threshold = min(image.shape[1] // 50, image.shape[0] // 50) 
    print(f"Threshold: {threshold}")
    for (x, y) in eye_landmarks:
        for i in range(max(0, y - threshold), min(image.shape[0], y + threshold)):
            for j in range(max(0, x - threshold), min(image.shape[1], x + threshold)):
                if hair_mask[i, j] > 0:
                    return True
    return False

def process_image(image, face_detector, cloth_hair_segmentator):
    # image = cv2.imread(image_path)
    start_time = time.time()
    
    face_box = None
    image, face_box = detect_face(image, face_detector)
    eye_landmarks, threshold = find_eye_landmarks(image, face_box)
    
    _, hair_mask, _ = cloth_hair_segmentator.segment_cloth_and_hair(image)
    
    # Save visualize image
    visualize_image = visualize_eye_landmarks_hair(image, eye_landmarks, hair_mask)
    
    eye_bang_result = check_eyes_bang_collission2(image, eye_landmarks, hair_mask, threshold)
    process_time = time.time() - start_time
    
    return eye_bang_result, process_time, visualize_image

if __name__ == "__main__":
    face_detector = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
    cloth_hair_segmentator = ClothHairSegmentator()
    
    # Process single image
    # input_path = "data/eyeBangs/in/43.png"
    # input_path = "test_data/frame_00009.jpg"
    # input_path = "data/eyeBangs/in/21.png"
    input_path = "data/acne/in/30.jpg"
    image = cv2.imread(input_path)
    eye_bang_result, _, visualize_img = process_image(image, face_detector, cloth_hair_segmentator)
    print(eye_bang_result)
    cv2.imwrite("result.jpg", visualize_img)
    
    