import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the VOC biomarker image for the CNN model.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        arr = np.array(img, dtype='float32') / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def extract_features(image_path):
    """
    Simulate feature selection from VOC biomarkers.
    In a real scenario, this would extract specific intensities of VOC peaks.
    """
    # Return 100 mock feature values
    return np.random.rand(100)
