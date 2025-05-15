import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import time

class BaldnessDetector:
    def __init__(self, model_weights_path, img_height=218, img_width=178):
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.model_weights_path = model_weights_path
        self.model = self._create_model()
        self.model.load_weights(self.model_weights_path)

    def _create_model(self):
        """Create and return the model architecture"""
        inc_model = InceptionV3(weights=None,
                              include_top=False,
                              input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        
        x = inc_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        
        return Model(inputs=inc_model.input, outputs=predictions)

    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img_pred = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                            (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_pred = img_pred.astype(np.float32) / 255.0
        return np.expand_dims(img_pred, axis=0)

    def add_prediction_overlay(self, img, pred_class, confidence):
        """Add prediction text overlay to image"""
        img_display = img.copy()
        # text = f"{pred_class}: {confidence*100:.2f}%"
        text = f"{pred_class}"
        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(text, font, 
                                                            font_scale, 
                                                            font_thickness)
        cv2.rectangle(img_display, 
                     (position[0], position[1] - text_height - baseline),
                     (position[0] + text_width, position[1] + baseline),
                     (0, 0, 0),
                     -1)
        
        cv2.putText(img_display, text, position, font, font_scale, 
                   font_color, font_thickness)
        
        return img_display

    def predict_single_image(self, img):
        """Process a single image and save the result"""
        
        try:
            # img = cv2.imread(image_path)
            if img is None:
                print(f"Error reading image")
                return None, None, 0

            img_pred = self.preprocess_image(img)
            result = self.model.predict(img_pred)
            pred_class = "Yes" if result[0][1] > 0.5 else "No"
            confidence = result[0][1] if result[0][1] > 0.5 else result[0][0]

            img_result = self.add_prediction_overlay(img, pred_class, confidence)
            # cv2.imwrite(output_path, img_result)

            return pred_class, confidence, img_result

        except Exception as e:
            print(f"Error processing: {str(e)}")
            return None, None, 0

def main():
    model_weights_path = 'weights/weights.best.inc.bald.hdf5'  

    detector = BaldnessDetector(model_weights_path)    
    image_path = 'data/bald/1.png'
    output_path = 'result.jpg'
    image = cv2.imread(image_path)
    pred, conf, result_img = detector.predict_single_image(image)
    cv2.imwrite(output_path, result_img)  # Save the result image
    print(f"Prediction: {pred}, Confidence: {conf:.2f}")

if __name__ == "__main__":
    main()