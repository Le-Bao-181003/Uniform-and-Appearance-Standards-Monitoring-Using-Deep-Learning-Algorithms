import cv2
import os
from transformers import pipeline
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import time 
from pathlib import Path

class ClothHairSegmentator:
    def __init__(self):
        self.segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

    def filter_small_regions(self, mask, area_threshold):
        binary_mask = np.array(mask > 0, dtype=np.uint8)
        labeled_mask = label(binary_mask)
        filtered_mask = np.zeros_like(binary_mask)

        for region in regionprops(labeled_mask):
            if region.area >= area_threshold:
                for coords in region.coords:
                    filtered_mask[coords[0], coords[1]] = 255
        return filtered_mask

    def segment_cloth_and_hair(self, image, bounding_box=None, 
                             clothes=["Hair", "Upper-clothes", "Skirt", "Dress"]):
        if bounding_box is None:
            crop_image = image
        else:
            xmin, ymin, xmax, ymax = map(int, bounding_box)
            crop_image = image[ymin:ymax, xmin:xmax]
        
        crop_im_pil = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        cloth_hair_segment = self.segmenter(crop_im_pil)
        
        mask_cloth = []
        mask_hair = []
        for s in cloth_hair_segment:
            if s['label'] in clothes and s['label'] != "Hair":
                mask_cloth.append(s['mask'])
            if s['label'] == "Hair":
                mask_hair.append(s['mask'])
        
        # if mask_hair == []:
        #     print("No hair mask found")
        #     return None, None
        
        if len(mask_cloth) == 0:
            cloth_mask  = np.zeros(crop_image.shape[:2], dtype=np.uint8)
        else:
            cloth_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
            for mask in mask_cloth:
                cloth_mask += np.array(mask, dtype=np.uint8)
            
            area_percentage = 2
            total_cloth_pixels = np.sum(cloth_mask > 0)
            area_threshold = total_cloth_pixels * area_percentage // 100
            cloth_mask = self.filter_small_regions(cloth_mask, area_threshold)
            
            cloth_mask = np.where(cloth_mask > 0, 255, 0).astype(np.uint8)
            cloth_mask = cv2.resize(cloth_mask, (crop_image.shape[1], crop_image.shape[0]))
            
        if "Hair" not in clothes or len(mask_hair) == 0:
            hair_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
        else:
            hair_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
            for mask in mask_hair:
                hair_mask += np.array(mask, dtype=np.uint8)
            area_percentage = 2
            total_hair_pixels = np.sum(hair_mask > 0)
            area_threshold = total_hair_pixels * area_percentage // 100
            hair_mask = self.filter_small_regions(hair_mask, area_threshold)
            
            hair_mask = np.where(hair_mask > 0, 255, 0).astype(np.uint8)
            hair_mask = cv2.resize(hair_mask, (crop_image.shape[1], crop_image.shape[0]))
        
        cloth_and_hair_mask = cv2.bitwise_or(cloth_mask, hair_mask)
        
        return cloth_mask, hair_mask, cloth_and_hair_mask

def save_mask(image, mask, output_path = None):
    segmented_mask = cv2.bitwise_and(image, image, mask=mask)
    if output_path is None:
        output_path = "data/segment/segmented_mask.jpg"
    cv2.imwrite(output_path, segmented_mask)

if __name__ == "__main__":
    segmentator = ClothHairSegmentator()
    # image_path = "data/eyeBangs/in/10.png"
    image_path = 'test_data/frame_00222.jpg'
    image = cv2.imread(image_path)
    
    # global output_folder
    # output_folder = f"data/segment"
    # os.makedirs(output_folder, exist_ok=True)
    # print(output_folder)
    
    # Get cloth and hair mask from the image (cv2, BGR format)
    cloth_mask, hair_mask, cloth_and_hair_mask = segmentator.segment_cloth_and_hair(image)
    # save_mask(image, cloth_mask, f"{output_folder}/cloth_mask.jpg")
    # save_mask(image, hair_mask, f"{output_folder}/hair_mask.jpg")
    
    # cloth_and_hair_mask = cv2.bitwise_and(image, image, mask=cloth_mask + hair_mask)
    save_mask(image, cloth_and_hair_mask, f"cloth_and_hair_mask.jpg")
    
    save_mask(image, cloth_mask, f"cloth_mask.jpg")
    save_mask(image, hair_mask, f"hair_mask.jpg")
    
    cloth_and_hair_mask = cv2.bitwise_and(image, image, mask=cloth_and_hair_mask)
    if len(cloth_and_hair_mask.shape) == 2:
        combined_mask_rgb = cv2.cvtColor(cloth_and_hair_mask, cv2.COLOR_GRAY2BGR)
    else:
        combined_mask_rgb = cloth_and_hair_mask

    # Resize mask về cùng kích thước nếu cần
    if combined_mask_rgb.shape != image.shape:
        combined_mask_rgb = cv2.resize(combined_mask_rgb, (image.shape[1], image.shape[0]))

    combined_view = np.hstack((image, combined_mask_rgb))

    # Lưu ảnh
    cv2.imwrite("output_combined.jpg", combined_view)
    
    
    # print(f"hair mask: {hair_mask}")
    # print(f"hair mask shape: {hair_mask.shape}")
    # print(f"hair mask dtype: {hair_mask.dtype}")
    
    
    # def mask_to_coords(mask):
    #     coords = np.column_stack(np.where(mask == 255))
    #     return coords.tolist()
    
    # hair_coords = mask_to_coords(hair_mask)
    # print(f"hair coords: {hair_coords}")
    
