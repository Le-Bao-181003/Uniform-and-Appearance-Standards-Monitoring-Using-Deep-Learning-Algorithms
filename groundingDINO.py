import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import numpy as np
import cv2

def compute_non_overlap_ratio(earring_box, ear_box):

    if earring_box is None:
        return "No earring detected in the image"
    if ear_box is None:
        return "No ear detected in the image"

    x_min_e, y_min_e, x_max_e, y_max_e = map(int, earring_box)
    print(f"earring box: {x_min_e, y_min_e, x_max_e, y_max_e}")
    x_min_ear, y_min_ear, x_max_ear, y_max_ear = map(int, ear_box)
    print(f"ear box: {x_min_ear, y_min_ear, x_max_ear, y_max_ear}")
    
    earring_area = (x_max_e - x_min_e) * (y_max_e - y_min_e)
    if earring_area == 0:
        return "Earring area is 0"
    
    x_min_overlap = max(x_min_e, x_min_ear)
    y_min_overlap = max(y_min_e, y_min_ear)
    x_max_overlap = min(x_max_e, x_max_ear)
    y_max_overlap = min(y_max_e, y_max_ear)
    
    overlap_width = max(0, x_max_overlap - x_min_overlap)
    overlap_height = max(0, y_max_overlap - y_min_overlap)
    overlap_area = overlap_width * overlap_height
    
    non_overlap_area = earring_area - overlap_area
    non_overlap_ratio = (non_overlap_area / earring_area) * 100

    # Threshold for non-overlap ratio
    threshold = 30
    if non_overlap_ratio < threshold:
        return "Earring is inside the ear"
    else:
        return "Earring is outside the ear"

def draw_boxes(image, detection_results):
    """
    Draw bounding boxes and labels on the image with different colors for each label.

    Args:
        image: Input image (PIL Image or numpy array) (RGB)
        detection_results: Dictionary with detection data
                          Format: {label: [{"box": [x_min, y_min, x_max, y_max], "score": float}, ...]}

    Returns:
        numpy array: Image with boxes and labels drawn
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Define a color palette (BGR format)
    color_palette = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (128, 128, 128), # Gray
        (0, 128, 128),  # Teal
    ]

    # Assign a color to each label
    label_colors = {}
    for i, label in enumerate(detection_results.keys()):
        label_colors[label] = color_palette[i % len(color_palette)]  # Cycle through colors if needed

    num_boxes = 0

    # Draw boxes and labels
    for label, detections in detection_results.items():
        color = label_colors[label]  # Get color for this label
        for detection in detections:
            x_min, y_min, x_max, y_max = detection["box"]
            score = detection["score"]

            # Draw rectangle
            cv2.rectangle(
                image_cv,
                (x_min, y_min),
                (x_max, y_max),
                color=color,
                thickness=2
            )

            # Draw label text
            # label_text = f"{label} ({score:.2f})"
            label_text = f"{label}"
            cv2.putText(
                image_cv,
                label_text,
                (x_min, y_min - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA
            )

            num_boxes += 1

    # Add summary text with total number of boxes
    # summary_text = f"Boxes: {num_boxes}"
    # cv2.putText(
    #     image_cv,
    #     summary_text,
    #     (10, 50),
    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #     fontScale=1.5,
    #     color=(255, 0, 0),  # Blue for summary text
    #     thickness=2,
    #     lineType=cv2.LINE_AA
    # )

    return image_cv

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: List of [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.

    Args:
        boxes: List of boxes [[x_min, y_min, x_max, y_max], ...]
        scores: List of confidence scores
        iou_threshold: IoU threshold to suppress boxes (default: 0.5)

    Returns:
        List of indices to keep
    """
    if not boxes:
        return []

    # Sort boxes by scores in descending order
    indices = np.argsort(scores)[::-1]
    keep = []

    while indices.size > 0:
        # Pick the box with highest score
        i = indices[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        ious = [calculate_iou(boxes[i], boxes[j]) for j in indices[1:]]

        # Keep boxes with IoU below threshold
        indices = indices[1:][np.array(ious) <= iou_threshold]

    return keep

class GroundingDINO:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        """Initialize GroundingDINO with model_id."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def detect(self, image, text, box_threshold=0.4, text_threshold=0.3, iou_threshold=0.5):
        """
        Detect objects in the image based on text queries, with NMS to remove overlapping boxes.

        Args:
            image: PIL Image object (RGB)
            text: List of strings or single string (objects to detect)
            box_threshold: Confidence threshold for boxes (default: 0.4)
            text_threshold: Confidence threshold for text matching (default: 0.3)
            iou_threshold: IoU threshold for NMS (default: 0.5)

        Returns:
            dict: Dictionary with keys as labels and values as lists of boxes (coordinates and scores)
                  Format: {label: [{"box": [x_min, y_min, x_max, y_max], "score": float}, ...]}
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image object")
        if isinstance(text, str):
            text = [text]
        if not text or not isinstance(text, list):
            raise ValueError("Text input must be a non-empty list or string")

        image = image.convert("RGB")

        # Create inputs for model
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        # Infer
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )

        # Result dictionary
        detection_results = {}

        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]

            # Group boxes by label
            label_boxes = {}
            for box, score, label in zip(boxes, scores, labels):
                score_value = score.item()
                if score_value >= box_threshold:
                    if label not in label_boxes:
                        label_boxes[label] = {"boxes": [], "scores": []}
                    label_boxes[label]["boxes"].append([int(coord) for coord in box.tolist()])
                    label_boxes[label]["scores"].append(score_value)

            # Apply NMS per label
            for label, data in label_boxes.items():
                keep_indices = nms(data["boxes"], data["scores"], iou_threshold)
                detection_results[label] = [
                    {
                        "box": data["boxes"][i],
                        "score": data["scores"][i]
                    }
                    for i in keep_indices
                ]

        del inputs, outputs
        torch.cuda.empty_cache()

        return detection_results
    
if __name__ == "__main__":
    # Example usage
    grounding_dino = GroundingDINO()
    input_image_path = "data/earring/33.png"
    image = Image.open(input_image_path)
    width, height = image.size
    scale = 1400 // max(height, width) + 1
    new_size = (scale * width, scale * height)
    image = image.resize(new_size, Image.BICUBIC)
    text_queries = ["earrings", "ear"]
    
    results = grounding_dino.detect(image, text_queries, box_threshold=0.35, text_threshold=0.25)
    image_draw_box = draw_boxes(image, results)
    cv2.imwrite("result.jpg", image_draw_box)
    
    for label, detections in results.items():
        print(f"{label}:")
        for detection in detections:
            print(f"  Box: {detection['box']}, Score: {detection['score']:.2f}")

    earring_detections = results.get('earrings', [])
    ear_detections = results.get('ear', [])
    for i, earring_det in enumerate(earring_detections):
        earring_box = earring_det['box']
        for j, ear_det in enumerate(ear_detections):
            ear_box = ear_det['box']
            print(f"\nChecking Earring {i+1} with Ear {j+1}:")
            result = compute_non_overlap_ratio(earring_box, ear_box)
            print(f"Result: {result}")