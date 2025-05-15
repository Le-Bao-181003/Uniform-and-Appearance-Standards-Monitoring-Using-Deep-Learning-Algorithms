from ultralytics import YOLO
import os
import cv2
import numpy as np

model_path = "weights/tie.pt"
model = YOLO(model_path)

image_path = "data/tie/1.jpg"
output_path = "result.jpg"
img = cv2.imread(image_path)

results = model.predict(source=img, conf=0.4, iou=0.3)  # iou=0.3 is the NMS threshold

detections = results[0].boxes  
cv2.imwrite(output_path, results[0].plot())  

print(f"Processed {image_path} with {len(detections)} detections")