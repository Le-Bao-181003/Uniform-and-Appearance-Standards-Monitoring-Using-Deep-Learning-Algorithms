import os
from typing import Optional, Dict
import logging
from typing import Optional
import numpy as np
from PIL import Image
import onnxruntime as ort
from facenet_pytorch import MTCNN
import torch
import torchvision.transforms as transforms
from utils.common import vis_parsing_maps
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class FacialSkinSegmentator:
    """Class for facial skin segmentation using MTCNN and ONNX model."""
    
    def __init__(self, onnx_model_path: str = "weights/resnet34.onnx"):
        """
        Initialize the FacialSkinSegmentator with MTCNN and ONNX model.
        
        Args:
            onnx_model_path (str): Path to the ONNX model file
        """
        # Initialize MTCNN detector
        self.detector = MTCNN(
            keep_all=True, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load ONNX model
        self.session = self._load_onnx_model(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Model input name: {self.input_name}")
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def _load_onnx_model(self, onnx_path: str) -> ort.InferenceSession:
        """Load and initialize the ONNX model."""
        if not os.path.exists(onnx_path):
            raise ValueError(f"ONNX model not found at path: {onnx_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        logger.info(f"ONNX model loaded successfully from {onnx_path}")
        logger.info(f"Using execution provider: {session.get_providers()[0]}")
        
        return session
    
    def _detect_and_crop_face(self, image: Image.Image) -> Optional[tuple[Image.Image, tuple[int, int, int, int]]]:
        """Detect and crop the face region from the image using MTCNN, return face image and bounding box."""
        img_np = np.array(image)
        boxes, _ = self.detector.detect(img_np)
        
        if boxes is None or len(boxes) == 0:
            logger.warning("No face detected in the image.")
            return None
        
        x, y, x2, y2 = map(int, boxes[0])
        width = x2 - x
        height = y2 - y
        
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        width = min(img_np.shape[1] - x, width + 2 * margin)
        height = min(img_np.shape[0] - y, height + 2 * margin)
        
        if width <= 0 or height <= 0:
            logger.warning("Invalid face region dimensions after adjustment.")
            return None
        
        face_region = img_np[y:y + height, x:x + width]
        return Image.fromarray(face_region), (x, y, width, height)
    
    def _prepare_image(self, image: Image.Image, input_size: tuple = (512, 512)) -> np.ndarray:
        """Prepare image for ONNX inference by resizing and normalizing."""
        resized_image = image.resize(input_size, resample=Image.BILINEAR)
        image_tensor = self.transform(resized_image)
        image_batch = image_tensor.unsqueeze(0)
        return image_batch.numpy()
    
    def predict(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Process an RGB image for facial skin segmentation and return a skin mask with original image size.
        
        Args:
            image (Image.Image): Input RGB image
        
        Returns:
            image_visualize_result
        """
        try:
            start_time = time.time()         
            # Detect and crop face
            face_result = self._detect_and_crop_face(image)
            if face_result is None:
                # logger.warning(f"Skipping {image_path} due to no face detected.")
                return {'Mask': np.zeros((image.height, image.width), dtype=np.uint8)}
            
            face_image, (x, y, width, height) = face_result
            face_size = face_image.size # Size of cropped face region
            
            # Prepare image for inference
            image_batch = self._prepare_image(face_image)
            
            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: image_batch})
            output = outputs[0]
            
            # Convert to segmentation mask
            predicted_mask = output.squeeze(0).argmax(0)
            
            # Create binary skin mask (assuming 'skin' is class 1)
            skin_mask = (predicted_mask == 1).astype(np.uint8)  # 1 for skin, 0 for others
            
            # Resize mask back to original face region size (all masks)
            # mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
            # restored_mask = mask_pil.resize(face_size, resample=Image.NEAREST)
            # predicted_mask = np.array(restored_mask)
            
            # Visualize and save results
            # vis_parsing_maps(
            #     face_image,
            #     predicted_mask,
            #     save_image=True,
            #     save_path=output_path,
            # )
            
            ## Resize skin mask to face region size
            skin_mask_pil = Image.fromarray(skin_mask * 255)  # Scale to 0-255 for PIL
            restored_face_mask = skin_mask_pil.resize(face_size, resample=Image.NEAREST)
            skin_mask_face = np.array(restored_face_mask) // 255  # Back to binary (0 or 1)
            
            # Create full-size mask matching original image size
            full_skin_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            full_skin_mask[y:y + height, x:x + width] = skin_mask_face
            
            print(f"Time taken for inference: {time.time() - start_time:.2f} seconds")
             
            image_visualize_result = vis_parsing_maps(
                    image,
                    full_skin_mask,
                    save_image=False,
                )
            return image_visualize_result
            
        except Exception as e:
            logger.error(f"Error processing: {e}")
            return image

if __name__ == "__main__":
    segmentator = FacialSkinSegmentator(onnx_model_path="weights/resnet34.onnx")
    
    image_path = "data/acne/in/30.jpg"
    output_path = "result.jpg"
    image = Image.open(image_path).convert("RGB")
    result = segmentator.predict(
        image # RGB image
    )
    skin_mask = result['Mask']
    print(f"Skin mask shape: {skin_mask.shape}")
    print(f"Skin mask values: {np.unique(skin_mask)}")