import numpy as np
import cv2
from collections import Counter
import os
import time
import numba
from numba import jit, prange
from math import cos, sin, atan2, sqrt, exp, pi, ceil
from cloth_hair_segmentator import ClothHairSegmentator

np.random.seed(42)


@jit(nopython=True)
def degrees_to_radians(degrees):
    return degrees * (pi / 180.0)

@jit(nopython=True)
def radians_to_degrees(radians):
    return radians * (180.0 / pi)

@jit(nopython=True)
def numba_deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    Calculate color difference using CIEDE2000 formula with numba optimization.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    if abs(L1-L2) > 40 or abs(a1-a2) > 30 or abs(b1-b2) > 40: # dis > 20
        return 25.0
    
    if sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2) < 8.5: # dis < 10
        return 9.0
    
    # Calculate C1, C2 (Chroma)
    C1 = sqrt(a1**2 + b1**2)
    C2 = sqrt(a2**2 + b2**2)
    
    # Calculate C_bar (average Chroma)
    C_bar = (C1 + C2) / 2.0
    
    # Calculate G (creates adjustment to a axis)
    G = 0.5 * (1 - sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    # Calculate a' (adjusted a values)
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Calculate C1', C2' (adjusted Chroma values)
    C1_prime = sqrt(a1_prime**2 + b1**2)
    C2_prime = sqrt(a2_prime**2 + b2**2)
    
    # Calculate h1', h2' (adjusted hue angles)
    h1_prime = 0.0
    if not (a1_prime == 0.0 and b1 == 0.0):
        h1_prime = radians_to_degrees(atan2(b1, a1_prime))
        if h1_prime < 0:
            h1_prime += 360.0
    
    h2_prime = 0.0
    if not (a2_prime == 0.0 and b2 == 0.0):
        h2_prime = radians_to_degrees(atan2(b2, a2_prime))
        if h2_prime < 0:
            h2_prime += 360.0
    
    # Calculate ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    delta_h_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h2_prime - h1_prime) <= 180.0:
            delta_h_prime = h2_prime - h1_prime
        elif h2_prime - h1_prime > 180.0:
            delta_h_prime = h2_prime - h1_prime - 360.0
        else:
            delta_h_prime = h2_prime - h1_prime + 360.0
    
    delta_H_prime = 2.0 * sqrt(C1_prime * C2_prime) * sin(degrees_to_radians(delta_h_prime) / 2.0)
    
    # Calculate CIEDE2000 components
    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    h_bar_prime = 0.0
    if C1_prime * C2_prime != 0:
        if abs(h1_prime - h2_prime) <= 180.0:
            h_bar_prime = (h1_prime + h2_prime) / 2.0
        elif abs(h1_prime - h2_prime) > 180.0 and h1_prime + h2_prime < 360.0:
            h_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
        else:
            h_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0
    
    T = 1.0 - 0.17 * cos(degrees_to_radians(h_bar_prime - 30.0)) + \
             0.24 * cos(degrees_to_radians(2.0 * h_bar_prime)) + \
             0.32 * cos(degrees_to_radians(3.0 * h_bar_prime + 6.0)) - \
             0.20 * cos(degrees_to_radians(4.0 * h_bar_prime - 63.0))
    
    SL = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / sqrt(20.0 + (L_bar_prime - 50.0)**2)
    SC = 1.0 + 0.045 * C_bar_prime
    SH = 1.0 + 0.015 * C_bar_prime * T
    
    RT = -2.0 * sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25.0**7)) * \
         sin(degrees_to_radians(60.0 * exp(-((h_bar_prime - 275.0) / 25.0)**2)))
    
    # Calculate the final color difference
    deltaE = sqrt(
        (delta_L_prime / (kL * SL))**2 +
        (delta_C_prime / (kC * SC))**2 +
        (delta_H_prime / (kH * SH))**2 +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    
    return deltaE

@jit(nopython=True, parallel=True)
def batch_color_distance(norm_color, norm_colors):
    """
    Calculate distance between one color and multiple other colors in parallel.
    
    Args:
        norm_color: Single normalized LAB color
        norm_colors: Array of normalized LAB colors
    
    Returns:
        Array of distances
    """
    distances = np.zeros(len(norm_colors), dtype=np.float32)
    for i in prange(len(norm_colors)):
        distances[i] = numba_deltaE_ciede2000(norm_color, norm_colors[i])
    return distances

class DominantColorFinder:
    def __init__(self, threshold=15.0, block_size=3, top_n=5, quantize_factor=4):
        """
        Initialize a class to find dominant colors in an image.
        
        Args:
            threshold (float): Color distance threshold (CIELAB) to group similar colors (default: 15.0).
            block_size (int): Block size for image downsampling (default: 3).
            top_n (int): Maximum number of dominant colors to display (default: 5).
            quantize_factor (int): Factor for color quantization (default: 4).
        """
        self.threshold = threshold
        self.block_size = block_size
        self.top_n = top_n
        self.quantize_factor = quantize_factor
        self.l_lookup = np.array([(i / 255) * 100 for i in range(256)], dtype=np.float32)
        self.ab_lookup = np.array([i - 128 for i in range(256)], dtype=np.float32)  

    def fast_normalize_cielab(self, lab_colors):
        """
        Normalize multiple LAB colors efficiently using lookup tables.
        
        Args:
            lab_colors (np.array): Array of LAB colors with shape (n, 3).
        
        Returns:
            np.array: Normalized LAB values.
        """
        normalized = np.zeros((len(lab_colors), 3), dtype=np.float32)
        normalized[:, 0] = self.l_lookup[lab_colors[:, 0]]
        normalized[:, 1] = self.ab_lookup[lab_colors[:, 1]]
        normalized[:, 2] = self.ab_lookup[lab_colors[:, 2]]
        return normalized

    def find_dominant_colors(self, image, masks):
        """
        Find the dominant colors in the specified region of the image.
        
        Args:
            image_path (str): Path to the image.
            masks (np.array): Binary mask where foreground is 255.
        
        Returns:
            tuple: (clusters, total_downsampled) - List of color clusters and total pixels after downsampling.
        """
        start_time = time.time()
        # image = cv2.imread(image_path)
        
        height, width = image.shape[:2]
        resize_scale = 1
        resize_time = 0
        
        if min(height, width) > 400:
            resize_start = time.time()
            scale = ceil(min(height, width) / 400)
            image = cv2.resize(image, (width // scale, height // scale), interpolation=cv2.INTER_LINEAR)
            masks = cv2.resize(masks, (width // scale, height // scale), interpolation=cv2.INTER_NEAREST)
            # resize_scale = scale
            # resize_time = time.time() - resize_start
            
        coords = mask_to_coords(masks)
        # print(f"Total pixel: {len(coords)}")
        # print(f"time to get coords: {time.time() - start_time:.2f} second")
        
        sizeDownsample = sqrt(len(coords) // 20000)
        if self.block_size < sizeDownsample:
            self.block_size = int(sizeDownsample)
        
        if not coords:
            return [], 0
            
        coords_arr = np.array(coords)
        rows, cols = coords_arr[:, 0], coords_arr[:, 1]
        
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        cropped_bgr = image[min_row:max_row+1, min_col:max_col+1, :]
        cropped_mask = masks[min_row:max_row+1, min_col:max_col+1]
        
        h, w = cropped_bgr.shape[:2]
        h, w = cropped_bgr.shape[:2]
        clothing_mask = cropped_mask == 255 
        
        downsampled_pixels = []
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                i_end = min(i + self.block_size, h)
                j_end = min(j + self.block_size, w)
                block_mask = clothing_mask[i:i_end, j:j_end]
                
                if np.any(block_mask):
                    block_region = cropped_bgr[i:i_end, j:j_end, :]
                    block_pixels = block_region[block_mask]
                    
                    if len(block_pixels) > 0:
                        random_pixel = block_pixels[np.random.randint(len(block_pixels))]
                        downsampled_pixels.append(random_pixel)
        
        print(f"Total pixel downsampled: {len(downsampled_pixels)}")
        total_downsampled = len(downsampled_pixels)
        
        if not downsampled_pixels:
            return [], 0
            
        start_time = time.time()
        downsampled_pixels_array = np.array(downsampled_pixels, dtype=np.uint8).reshape(-1, 1, 3)
        # downsampled_lab_array = cv2.cvtColor(downsampled_pixels_array, cv2.COLOR_BGR2LAB)
        # downsampled_lab = downsampled_lab_array.reshape(-1, 3)
        
        # Quantize the colors to reduce the number of unique colors
        quantized_pixels = ((downsampled_pixels_array // self.quantize_factor) * self.quantize_factor).astype(np.uint8)
        downsampled_lab_array = cv2.cvtColor(quantized_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        downsampled_lab = downsampled_lab_array.reshape(-1, 3)
        print(f"time to convert to LAB: {time.time() - start_time:.2f} second")
        
        start_time = time.time()
        downsampled_lab_tuples = [tuple(lab) for lab in downsampled_lab]
        color_counts = Counter(downsampled_lab_tuples)
        print(f"time to count colors: {time.time() - start_time:.2f} second")
        # df_top = visualize_lab_color_frequencies(
        # color_counts, 
        # image_path=f"{os.path.splitext(image_path)[0]}_lab_color_freq.png",
        # top_n=50
        # )
        
        start_time = time.time()
        clusters = self.nms_color_clustering_optimized(color_counts)
        clustering_time = time.time() - start_time
        print(f"Total clustering time: {clustering_time:.2f} second")
        
        # Visualize the clusters
        # if clusters:
        visualize_image = self.visualize_clusters(clusters, total_downsampled)
        
        return clusters, total_downsampled, visualize_image

    def nms_color_clustering_optimized(self, color_counts):
        """
        Group similar colors into clusters using NMS (Non-Maximum Suppression) with optimized color normalization.
        Args:
            color_counts (Counter): Counter object containing color frequencies.
        Returns:
            list: List of color clusters, each cluster contains color and frequency information.
        """
        if not color_counts:
            return []
            
        start_time = time.time()
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Sorting time: {time.time() - start_time:.2f} second")
        
        color_array = np.array([color for color, _ in sorted_colors], dtype=np.uint8)
        normalized_colors = self.fast_normalize_cielab(color_array)
        
        normalized_colors_tuple = np.array([
            (lab[0], lab[1], lab[2]) for lab in normalized_colors
        ], dtype=np.float32)
        
        start_time = time.time()
        clusters = []
        used_indices = set()
        distance_time = 0
        print(f"Total colors: {len(sorted_colors)}")
        number_iterator = 0
        
        for i in range(len(sorted_colors)):
            if i in used_indices:
                continue
                
            number_iterator += 1
            current_color, current_freq = sorted_colors[i]
            current_norm_color = normalized_colors_tuple[i]
            
            start_distance_time = time.time()
            
            remaining_indices = np.array([j for j in range(len(sorted_colors)) if j > i and j not in used_indices])
            if len(remaining_indices) > 0:
                remaining_colors = normalized_colors_tuple[remaining_indices]
                distances = batch_color_distance(current_norm_color, remaining_colors)
                
                similar_indices = remaining_indices[distances <= self.threshold]
            else:
                similar_indices = []
                
            distance_time += time.time() - start_distance_time
            
            cluster = {
                'color': current_color,
                'total_frequency': current_freq,
                'members': [(current_color, current_freq)]
            }
            used_indices.add(i)
            
            for j in similar_indices:
                compare_color, compare_freq = sorted_colors[j]
                cluster['total_frequency'] += compare_freq
                cluster['members'].append((compare_color, compare_freq))
                used_indices.add(j)
                    
            clusters.append(cluster)
            
        print(f"Number of iterations: {number_iterator}")
        print(f"Distance calculation time: {distance_time:.2f} second")
        print(f"Clustering loop time: {time.time() - start_time:.2f} second")
        
        start_time = time.time()
        clusters.sort(key=lambda x: x['total_frequency'], reverse=True)
        print(f"Sorting clusters time: {time.time() - start_time:.2f} second")
        
        return clusters

    def visualize_clusters(self, clusters, total_downsampled):
        """
        Create an image of horizontally concatenated sub-images, each filled with a dominant color
        and its frequency text in the top-left corner, using only OpenCV.
        Args:
            clusters (list): List of color clusters.
            total_downsampled (int): Total number of downsampled pixels.
        Returns:
            np.ndarray: The generated image in OpenCV BGR format.
        """
        n = min(self.top_n, len(clusters))
        
        # Filter clusters by frequency threshold (3% of total_downsampled)
        for i in range(n):
            cluster_freq = clusters[i]['total_frequency']
            if cluster_freq < ceil(total_downsampled * 0.03):
                n = i
                break
        
        if n == 0:
            return None

        sub_img_size = (200, 200)
        text_color = (255, 255, 255)  # White text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1

        sub_images = []

        for i in range(n):
            cluster = clusters[i]
            lab_color = np.array([[list(cluster['color'])]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]
            # Create a solid color sub-image
            sub_img = np.full((sub_img_size[1], sub_img_size[0], 3), bgr_color, dtype=np.uint8)
            
            text = f"Freq: {cluster['total_frequency']}"
            text_position = (10, 25)  
            cv2.putText(sub_img, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            sub_images.append(sub_img)

        final_img = np.hstack(sub_images)
        
        return final_img

def mask_to_coords(mask):
    coords = np.column_stack(np.where(mask == 255))
    return coords.tolist()

def save_mask(image_path, mask):
    image = cv2.imread(image_path)
    segmented_mask = cv2.bitwise_and(image, image, mask=mask)
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"segmented_{image_name}")
    cv2.imwrite(output_path, segmented_mask)

def process_image(image, segmentator):
    
    start_time = time.time()
    # image_name = os.path.basename(image_path)
    # image = cv2.imread(image_path)
    # print(f"Processing image: {image_name}")
    
    _, hair_mask, _ = segmentator.segment_cloth_and_hair(image)
    # hair_mask, _, _ = segmentator.segment_cloth_and_hair(image)
    # save_mask(image_path, hair_mask)
    
    
    # segment_time = time.time() - start_time
    # print(f"Segment time: {segment_time:.2f} second")
    
    
    if not hair_mask.any():
        print("Doesn't detect any clothing region!")
        return
    
    start_time = time.time()
    finder = DominantColorFinder(threshold=15.0, block_size=3, top_n=5)
    clusters, total_pixel_downsample, visualize_image = finder.find_dominant_colors(image, hair_mask)
    print(f"Total clustering time: {time.time() - start_time:.2f} second")
    # print(f"Total pixel downsampled: {total_pixel_downsample}")
        
    # for cluster in clusters:
    #     print(f"Color: {cluster['color']} - Freq: {cluster['total_frequency']}")
        
    result = check_hair_dominant_color(clusters, total_pixel_downsample)
    # if result:
    #     print("The hair color is black or brown-black.")
    # else:
    #     print("The hair color is not black or brown-black.")
    
    return result, visualize_image

def process_image_cloth(image, segmentator):
    
    start_time = time.time()
    # image_name = os.path.basename(image_path)
    # image = cv2.imread(image_path)
    # print(f"Processing image: {image_name}")
    
    cloth_mask, _, _ = segmentator.segment_cloth_and_hair(image)
    # hair_mask, _, _ = segmentator.segment_cloth_and_hair(image)
    # save_mask(image_path, hair_mask)
    
    
    # segment_time = time.time() - start_time
    # print(f"Segment time: {segment_time:.2f} second")
    
    
    if not cloth_mask.any():
        print("Doesn't detect any clothing region!")
        return
    
    start_time = time.time()
    finder = DominantColorFinder(threshold=15.0, block_size=3, top_n=5)
    clusters, total_pixel_downsample, visualize_image = finder.find_dominant_colors(image, cloth_mask)
    print(f"Total clustering time: {time.time() - start_time:.2f} second")
    # print(f"Total pixel downsampled: {total_pixel_downsample}")
        
    # for cluster in clusters:
    #     print(f"Color: {cluster['color']} - Freq: {cluster['total_frequency']}")
        
    # result = check_hair_dominant_color(clusters, total_pixel_downsample)
    
    return visualize_image

def visualize_lab_color_frequencies(color_counts, image_path=None, top_n=50):
    """
    Visualize the frequency of LAB colors after quantization and downsampling.
    
    Args:
        color_counts (Counter): Counter object with LAB color tuples as keys and counts as values
        image_path (str, optional): Path to save the visualization image
        top_n (int, optional): Number of top colors to display
        
    Returns:
        None: Displays the visualization and optionally saves to file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import pandas as pd
    
    # Convert color_counts to DataFrame
    df = pd.DataFrame(color_counts.items(), columns=["LAB", "Frequency"])
    df = df.sort_values(by="Frequency", ascending=False)
    df_top = df.head(top_n)
    
    # Convert LAB colors to RGB for visualization
    rgb_colors = []
    for lab_color in df_top["LAB"]:
        # Convert single LAB color to BGR
        lab_array = np.array([[list(lab_color)]], dtype=np.uint8)
        bgr_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)
        # Convert BGR to RGB for matplotlib
        rgb_color = (bgr_array[0, 0, 2] / 255.0, 
                    bgr_array[0, 0, 1] / 255.0, 
                    bgr_array[0, 0, 0] / 255.0)
        rgb_colors.append(rgb_color)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(df_top)), df_top["Frequency"], color=rgb_colors)
    
    # Add total percentage information
    total_pixels = sum(color_counts.values())
    total_displayed = sum(df_top["Frequency"])
    percentage = (total_displayed / total_pixels) * 100
    # plt.figtext(0.5, 0.01, f"Showing top {top_n} colors: {total_displayed} pixels ({percentage:.2f}% of total)", 
    #             ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    # Set x-axis ticks and labels
    plt.xticks(range(len(df_top)), [f"({l},{a},{b})" for l,a,b in df_top["LAB"]], 
              rotation=90, ha='center', fontsize=8)
    
    # Add frequency values on top of each bar
    for i, (freq, color) in enumerate(zip(df_top["Frequency"], rgb_colors)):
        text_color = 'black' if sum(color) > 1.5 else 'white'
        percentage = (freq / total_pixels) * 100
        plt.text(i, freq + (max(df_top["Frequency"]) * 0.01), 
                f"{freq}", 
                ha='center', va='bottom', fontsize=8,
                color='black')
    
    # Set labels and title
    plt.xlabel("LAB Color", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Top {top_n} Most Frequent Colors in LAB Color Space", fontsize=14)
    
    # Add a second x-axis with color index numbers
    # ax2 = plt.twiny()
    # ax2.set_xlim(plt.gca().get_xlim())
    # ax2.set_xticks(range(len(df_top)))
    # ax2.set_xticklabels([f"Color {i+1}" for i in range(len(df_top))], fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save to file if path is provided
    if image_path:
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {image_path}")
    
    plt.show()
    
    return df_top

def normalize_cielab(color_lab):
    # print(lab_color)
    L, a, b = color_lab
    return np.array([
        L / 100.0,
        (a - 128.0) / 127.0,
        (b - 128.0) / 127.0
    ], dtype=np.float32)

def check_hair_dominant_color(clusters, total_pixel_downsample):
    
    # black_color_rgb = [0, 0, 0]
    # brown_black_color_rgb = [36, 29, 29]
    # black_color_lab = cv2.cvtColor(np.array([[black_color_rgb]]).astype(np.uint8), cv2.COLOR_BGR2LAB)[0][0]
    # brown_black_color_lab = cv2.cvtColor(np.array([[brown_black_color_rgb]]).astype(np.uint8), cv2.COLOR_BGR2LAB)[0][0]
    # print(brown_black_color_lab)
    
    black_color_lab = [0, 128, 128]
    black_color_lab_normalized = normalize_cielab(black_color_lab)
    brown_black_color_lab = [28, 129, 123]
    brown_black_color_lab_normalized = normalize_cielab(brown_black_color_lab)
    
    count_black_and_brown_black_pixels = 0

    num_clusters = len(clusters)
    
    num_suitable_clusters = 0
    
    for cluster in clusters:
        color_lab = cluster['color']
        freq = cluster['total_frequency']
        
        color_lab_list = list(color_lab)
        # print(color_lab_list)
        color_lab_normalized = normalize_cielab(color_lab_list)
        
        distane_with_black = numba_deltaE_ciede2000(color_lab_normalized, black_color_lab_normalized)
        # print(f"Distance with black: {distane_with_black}")
        distane_with_brown_black = numba_deltaE_ciede2000(color_lab_normalized, brown_black_color_lab_normalized)
        # print(f"Distance with brown-black: {distane_with_brown_black}")
        
        if distane_with_black <= 10.0 or distane_with_brown_black <= 10.0:
            count_black_and_brown_black_pixels += freq
            num_suitable_clusters += 1
        
        # color_rgb = cv2.cvtColor(np.array([[list(color_lab)]]).astype(np.uint8), cv2.COLOR_LAB2BGR)[0][0]
        # color_rgb = tuple(color_rgb)
    
    percentage = (count_black_and_brown_black_pixels / total_pixel_downsample) * 100
    
    if percentage >= 90 or num_clusters - num_suitable_clusters <= 2:
        return True
    
    return False
        

output_folder = 'data/dominant_colors/'
os.makedirs(output_folder, exist_ok=True)

if __name__ == "__main__":
    
    segmentator = ClothHairSegmentator()
    
    
    # image_path = "data/cloth/10.png"
    image_path = 'data/hair/eyeBangs/frame_00009.jpg'
    image = cv2.imread(image_path)
    
    result, visualize_image = process_image(image, segmentator)
    
    cv2.imwrite("result.jpg", visualize_image)
    
    
    
    
    