import cv2
import os 
import numpy as np
from cloth_hair_segmentator import ClothHairSegmentator
import time

def check_hair_collar(image, bounding_box, hair_mask, cloth_mask):
    
    # image = cv2.imread(image_path)
    # image_name = os.path.basename(image_path)
    
    hair_points = np.argwhere(hair_mask == 255)
    highest_y_point_in_hair_mask = hair_points[np.argmax(hair_points[:, 0])]
    
    cloth_points = np.argwhere(cloth_mask == 255)
    lowest_y_point_in_cloth_mask = cloth_points[np.argmin(cloth_points[:, 0])]
    # print(f"highest_y_point_in_hair_mask: {highest_y_point_in_hair_mask}, lowest_y_point_in_cloth_mask: {lowest_y_point_in_cloth_mask}")
    
    if bounding_box is None:
        bounding_box = (0, 0, image.shape[1], image.shape[0])
    xmin, ymin, xmax, ymax = map(int, bounding_box)
    
    
    final_mask = cv2.bitwise_or(cloth_mask, hair_mask)
    crop_image = image[ymin:ymax, xmin:xmax]
    segmented_image = cv2.bitwise_and(crop_image, crop_image, mask=final_mask)
    
    
    cv2.circle(segmented_image, (highest_y_point_in_hair_mask[1], highest_y_point_in_hair_mask[0]), 5, (0, 0, 255), -1)
    cv2.circle(segmented_image, (lowest_y_point_in_cloth_mask[1], lowest_y_point_in_cloth_mask[0]), 5, (255, 0, 0), -1)  # Màu xanh (BGR)
    # cv2.imwrite(f"data/hairColar/out/{image_name}", segmented_image)
    
    threshold = 10
    if highest_y_point_in_hair_mask[0] < lowest_y_point_in_cloth_mask[0] + threshold:
        return False, segmented_image
    else:
        return True, segmented_image

def process_image(image, segmentator, yolo_box=None):
    
    cloth_mask, hair_mask, _ = segmentator.segment_cloth_and_hair(image, yolo_box)
    if cloth_mask is None or hair_mask is None:
        print("No cloth or hair mask found")
        return False, image
    
    hair_collar_res, visualize_img = check_hair_collar(image, yolo_box, hair_mask, cloth_mask)
    
    return hair_collar_res, visualize_img

def main(segmentator):
    # Process single image
    image_path = "data/hair/collar/1.png"
    image = cv2.imread(image_path)
    hair_collar_res, visualize_img = process_image(image, segmentator)
    print(f"{image_path}: {hair_collar_res}")
    cv2.imwrite("result.jpg", visualize_img)

if __name__ == "__main__":
    segmentator = ClothHairSegmentator()
    main(segmentator)
    
    