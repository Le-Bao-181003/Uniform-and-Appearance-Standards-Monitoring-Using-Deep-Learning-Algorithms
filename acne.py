import keras_cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch

class AcneDetector:
    def __init__(self, model_path):
        self.input_size = (640, 640)
        self.class_mapping = {0: 'Acne'}
        
        self.face_detector = MTCNN(
            keep_all=True,  # Keep all detected faces
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = self._create_model()
        self.model.load_weights(model_path)
        
    def _create_model(self):
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            "yolo_v8_xs_backbone",
            include_rescaling=True
        )
        
        model = keras_cv.models.YOLOV8Detector(
            num_classes=len(self.class_mapping),
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=5
        )
        
        return model
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.original_size = image.shape[:2]
        
        image_resized = cv2.resize(image, self.input_size)
        
        image_input = np.expand_dims(image_resized, 0).astype(np.float32)
        
        return image, image_input
    

    
    def draw_detections(self, image, acne_boxes, acne_confidences, face_boxes):
        image_with_boxes = image.copy()
        
        for box in face_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(
                image_with_boxes,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),  # Green for face detection
                2
            )
            cv2.putText(
                image_with_boxes,
                'Face',
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw acne boxes
        for box, conf in zip(acne_boxes, acne_confidences):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(
                image_with_boxes,
                (xmin, ymin),
                (xmax, ymax),
                (255, 0, 0),  # Blue for acne detection
                2
            )
            
            label = f'Acne: {conf:.2f}'
            cv2.putText(
                image_with_boxes,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
            
        return image_with_boxes
    
    def display_single_image(self, image_path, confidence_threshold=0.25, figsize=(12, 8)):
        try:
            image, boxes, confidences = self.detect(image_path, confidence_threshold)
            
            result_image = self.draw_detections(image, boxes, confidences)
            
            plt.figure(figsize=figsize)
            plt.imshow(result_image)
            plt.axis('off')
            
            plt.title(f'Detected {len(boxes)} acne spots')
            
            detection_info = [f'Detection {i+1}: {conf:.2f}' 
                            for i, conf in enumerate(confidences)]
            if detection_info:
                plt.figtext(0.02, 0.02, '\n'.join(detection_info), 
                          fontsize=8, color='blue')
            
            plt.show()
            
            return len(boxes)  
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
            return 0
        
    def detect_faces(self, image):
        boxes, probs = self.face_detector.detect(image)
        
        face_boxes = []
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(b) for b in box]
                # Ensure the box is within image bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax = min(image.shape[1], xmax)
                ymax = min(image.shape[0], ymax)
                face_boxes.append([xmin, ymin, xmax, ymax])
                
        return face_boxes

    def process_face_region(self, image, face_box):
        xmin, ymin, xmax, ymax = face_box
        
        face_image = image[ymin:ymax, xmin:xmax]
        
        face_resized = cv2.resize(face_image, self.input_size)
        
        face_input = np.expand_dims(face_resized, 0).astype(np.float32)
        
        predictions = self.model.predict(face_input, verbose=0)
        
        return face_image, predictions

    def scale_boxes_to_face(self, boxes, face_box, original_face_size):
        xmin_face, ymin_face, xmax_face, ymax_face = face_box
        face_width = xmax_face - xmin_face
        face_height = ymax_face - ymin_face
        
        w_scale = face_width / self.input_size[0]
        h_scale = face_height / self.input_size[1]
        
        scaled_boxes = []
        for box in boxes:
            x1 = int(box[0] * w_scale) + xmin_face
            y1 = int(box[1] * h_scale) + ymin_face
            x2 = int(box[2] * w_scale) + xmin_face
            y2 = int(box[3] * h_scale) + ymin_face
            scaled_boxes.append([x1, y1, x2, y2])
            
        return scaled_boxes
    
    def process_single_image(self, image, output_path=None, confidence_threshold=0.25):
        # image = cv2.imread(image_path)
        try:
            # image = cv2.imread(image_path)
            if image is None:
                # raise ValueError(f"Không thể đọc ảnh: {image_path}")
                raise ValueError("Không thể đọc ảnh")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width = image.shape[:2]
            
            face_boxes = self.detect_faces(image)
            
            if not face_boxes:
                return False, image, 0
            
            result_image = image.copy()
            all_acne_boxes = []
            all_acne_confidences = []
            
            for face_box in face_boxes:
                face_image, predictions = self.process_face_region(image, face_box)
                
                confidences = predictions["confidence"][0]
                mask = confidences >= confidence_threshold
                
                filtered_boxes = predictions["boxes"][0][mask]
                filtered_confidences = confidences[mask]
                
                # Scaled boxes to position on the original image
                scaled_boxes = self.scale_boxes_to_face(
                    filtered_boxes, 
                    face_box,
                    face_image.shape[:2]
                )
                
                all_acne_boxes.extend(scaled_boxes)
                all_acne_confidences.extend(filtered_confidences)
                
                
                xmin, ymin, xmax, ymax = face_box
                cv2.rectangle(
                    result_image,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 0),  
                    max(height, width) // 500
                )
                
            for box, conf in zip(all_acne_boxes, all_acne_confidences):
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(
                    result_image,
                    (xmin, ymin),
                    (xmax, ymax),
                    (255, 0, 0),  
                    max(height, width) // 800
                )
                
                # label = f'Acne: {conf:.2f}'
                # cv2.putText(
                #     result_image, label, (xmin, ymin - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5, (255, 0, 0), 2
                # )
            
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(output_path, result_image_bgr)
            
            print(f"Detect {len(all_acne_boxes)} acne on {len(face_boxes)} face")
            return True, result_image_bgr, len(all_acne_boxes)
            
        except Exception as e:
            print(f"Error when processing: {str(e)}")
            return False, image, 0

def main():
    MODEL_PATH = "weights/yolo_acne_detection.h5"
    
    detector = AcneDetector(MODEL_PATH)
    
    condition, result_image = detector.process_single_image(
        image_path="data/acne/in/30.jpg",
        output_path="detected_30.jpg",
        confidence_threshold=0.25
    )
    print(f"Condition: {condition}")

if __name__ == "__main__":
    main()