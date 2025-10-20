import os
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("standardization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Define constants
SOURCE_FOLDER = 'all screenshots'
TARGET_FOLDER = 'std_ss'
TARGET_SIZE = (224, 224)  # EfficientNetB0 input size
EXPECTED_SUBFOLDERS = ['invalid', 'fampay', 'gpay', 'misc', 'paytm', 'phonepe']
PAD_COLOR = (0, 0, 0)  # Black padding

def pad_and_resize_image(source_path, target_path):
    """
    Standardize a single image according to EfficientNetB0 requirements:
    - Resize to 224x224 while preserving aspect ratio through padding
    - Convert to RGB format
    - Ensure the entire original content is preserved
    """
    try:
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with Image.open(source_path) as img:
            # Convert to RGB (removes alpha channel if present)
            img = img.convert('RGB')
            
            # Calculate aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            
            # Determine new dimensions that preserve aspect ratio
            if aspect_ratio > 1:  # Wider than tall
                new_width = min(width, TARGET_SIZE[0])
                new_height = int(new_width / aspect_ratio)
            else:  # Taller than wide or square
                new_height = min(height, TARGET_SIZE[1])
                new_width = int(new_height * aspect_ratio)
            
            # Resize the image while preserving aspect ratio
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a new blank image with the target size
            padded_img = Image.new('RGB', TARGET_SIZE, PAD_COLOR)
            
            # Calculate position to paste (center the image)
            paste_x = (TARGET_SIZE[0] - new_width) // 2
            paste_y = (TARGET_SIZE[1] - new_height) // 2
            
            # Paste the resized image onto the padded canvas
            padded_img.paste(img_resized, (paste_x, paste_y))
            
            # Save the standardized image to the new location
            padded_img.save(target_path, quality=95)
            return True
    except Exception as e:
        logger.error(f"Error processing {source_path}: {e}")
        return False

def standardize_all_images():
    """
    Process all images in the source folder and save standardized versions to the target folder
    """
    if not os.path.exists(SOURCE_FOLDER):
        logger.error(f"Source folder '{SOURCE_FOLDER}' not found!")
        return False
    
    # Create the main target folder if it doesn't exist
    if os.path.exists(TARGET_FOLDER):
        logger.warning(f"Target folder '{TARGET_FOLDER}' already exists. Files may be overwritten.")
    else:
        os.makedirs(TARGET_FOLDER)
    
    # Collect statistics
    total_images = 0
    successful = 0
    failed = 0
    subfolder_counts = {}
    
    # Check if expected subfolders exist
    existing_subfolders = [f for f in os.listdir(SOURCE_FOLDER) 
                          if os.path.isdir(os.path.join(SOURCE_FOLDER, f))]
    
    missing_subfolders = set(EXPECTED_SUBFOLDERS) - set(existing_subfolders)
    if missing_subfolders:
        logger.warning(f"Missing expected subfolders: {missing_subfolders}")
    
    # Process all images in all subfolders
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        # Get relative path for creating the same structure in target folder
        rel_path = os.path.relpath(root, SOURCE_FOLDER)
        target_dir = os.path.join(TARGET_FOLDER, rel_path) if rel_path != '.' else TARGET_FOLDER
        
        # Create the corresponding directory in the target folder
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Get folder name for statistics
        folder_name = rel_path if rel_path != '.' else 'root'
        
        # Initialize counter for this subfolder
        if folder_name not in subfolder_counts:
            subfolder_counts[folder_name] = {'total': 0, 'success': 0, 'failed': 0}
        
        # Filter for image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            continue
            
        logger.info(f"Processing {len(image_files)} images in {folder_name}")
        
        # Process each image with progress bar
        for img_file in tqdm(image_files, desc=f"Standardizing {folder_name}"):
            source_path = os.path.join(root, img_file)
            target_path = os.path.join(target_dir, img_file)
            
            total_images += 1
            subfolder_counts[folder_name]['total'] += 1
            
            if pad_and_resize_image(source_path, target_path):
                successful += 1
                subfolder_counts[folder_name]['success'] += 1
            else:
                failed += 1
                subfolder_counts[folder_name]['failed'] += 1
    
    # Report results
    logger.info(f"\nStandardization complete!")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Successfully standardized: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Standardized images saved to: {os.path.abspath(TARGET_FOLDER)}")
    
    # Report per subfolder
    logger.info("\nResults by subfolder:")
    for subfolder, counts in subfolder_counts.items():
        logger.info(f"{subfolder}: {counts['success']}/{counts['total']} successful")
    
    return successful, failed, total_images

if __name__ == "__main__":
    print(f"Starting standardization of images for EfficientNetB0 (target size: {TARGET_SIZE})")
    print(f"Processing all images in '{SOURCE_FOLDER}' and saving to '{TARGET_FOLDER}'...")
    print(f"Method: Preserving entire content with padding (NO CROPPING)")
    successful, failed, total = standardize_all_images()
    
    if total == 0:
        print("No images were found to process!")
    else:
        success_rate = (successful / total) * 100
        print(f"\nSummary: {successful}/{total} images standardized successfully ({success_rate:.1f}%)")
        print(f"Standardized images saved to: {os.path.abspath(TARGET_FOLDER)}")
        
        if failed > 0:
            print(f"Check standardization.log for details on {failed} failed images")