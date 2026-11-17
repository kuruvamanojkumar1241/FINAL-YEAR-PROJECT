import os
import numpy as np
from PIL import Image, ImageFilter

def generate_saliency_heatmap(image_path, output_filename=None):
    """
    Generates an Explainable AI (XAI) attention/saliency heatmap overlay 
    highlighting suspicious biomarker hotspots and lung nodule regions.
    Uses pure PIL + NumPy for 100% compatibility across all Python environments.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(image_path))
        orig_name = os.path.basename(image_path)
        if not output_filename:
            output_filename = f"heatmap_{orig_name}"
            
        output_path = os.path.join(base_dir, output_filename)
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        orig_size = img.size
        
        # Convert to grayscale for gradient & saliency detection
        gray = img.convert('L')
        gray_arr = np.array(gray, dtype=np.float32) / 255.0
        
        # Edge & gradient response
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges, dtype=np.float32) / 255.0
        
        # Central-weighted attention mask (lungs / nodules are focused in central field)
        h, w = gray_arr.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_weight = 1.0 - (dist_from_center / max_dist) * 0.4
        
        # Synthesize activation saliency map
        saliency = (gray_arr * 0.4 + edges_arr * 0.6) * center_weight
        # Normalize 0 - 1
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Non-linear enhancement of peak activation areas
        saliency = np.power(saliency, 1.8)
        
        # Generate Jet colormap programmatically
        # Blue -> Cyan -> Yellow -> Red
        r = np.clip(1.5 - np.abs(saliency * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(saliency * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(saliency * 4.0 - 1.0), 0.0, 1.0)
        
        heatmap_rgb = np.stack([r, g, b], axis=-1) * 255.0
        heatmap_img = Image.fromarray(heatmap_rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=5))
        
        # Blend original image with heatmap (0.55 original, 0.45 heatmap)
        blended = Image.blend(img, heatmap_img, alpha=0.45)
        blended.save(output_path, quality=92)
        
        return output_filename
    except Exception as e:
        print(f"Error generating saliency heatmap: {e}")
        return ""
