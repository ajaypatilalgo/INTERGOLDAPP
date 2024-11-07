import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Load the pre-trained EfficientNetB0 model with weights trained on the ImageNet dataset.
        base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')  # Use include_top=False for feature extraction
        
        # Extract features from the last pooling layer.
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def extract(self, img):
        img = img.resize((224, 224)).convert('RGB')  # Resize and convert image to RGB
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Expand dimensions to match model input
        x = preprocess_input(x)  # Preprocess input to match ImageNet expectations
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)  # Normalize the feature vector
