import numpy as np
from PIL import Image
import os

# Constants - MUST match training preprocessing
TARGET_SIZE = (224, 224)
PAD_COLOR = (0, 0, 0)  # Black padding

def preprocess_single_image(image_path):

    try:
        with Image.open(image_path) as img:
            # Convert to RGB (removes alpha channel if present)
            img = img.convert('RGB')
            
            # Calculate aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            
            # Determine new dimensions that preserve aspect ratio
            if aspect_ratio > 1:  
                new_width = min(width, TARGET_SIZE[0])
                new_height = int(new_width / aspect_ratio)
            else:  
                new_height = min(height, TARGET_SIZE[1])
                new_width = int(new_height * aspect_ratio)
            
            # Resize the image while preserving aspect ratio
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            padded_img = Image.new('RGB', TARGET_SIZE, PAD_COLOR)
            
            # Calculate position to paste (center the image)
            paste_x = (TARGET_SIZE[0] - new_width) // 2
            paste_y = (TARGET_SIZE[1] - new_height) // 2
            padded_img.paste(img_resized, (paste_x, paste_y))
            
            # Convert to numpy array
            img_array = np.array(padded_img, dtype=np.float32)
                        
            return img_array
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def preprocess_batch_images(image_paths):

    processed_images = []
    valid_paths = []
    
    for img_path in image_paths:
        processed_img = preprocess_single_image(img_path)
        if processed_img is not None:
            processed_images.append(processed_img)
            valid_paths.append(img_path)
    
    if len(processed_images) == 0:
        return None, []
    
    # Stack into batch
    batch_array = np.stack(processed_images, axis=0)
    
    return batch_array, valid_paths

def get_image_files_from_folder(folder_path, extensions=('.jpg', '.jpeg', '.png')):

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(extensions):
            full_path = os.path.join(folder_path, filename)
            image_files.append(full_path)
    
    return sorted(image_files)  # Sort for consistent ordering

# Example usage
if __name__ == "__main__":
    # Test preprocessing on a single image
    test_image = "test_images/image_path"
    
    if os.path.exists(test_image):
        processed = preprocess_single_image(test_image)
        if processed is not None:
            print(f"Processed image shape: {processed.shape}")
            print(f"Pixel value range: [{processed.min():.1f}, {processed.max():.1f}]")
            print(f"Data type: {processed.dtype}")
        else:
            print("Failed to process image")
    else:
        print(f"Test image not found: {test_image}")
        print("\nThis module provides preprocessing functions for inference.")
        print("Import these functions in your inference script.")