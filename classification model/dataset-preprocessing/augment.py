import os
import cv2
import numpy as np
import random

# This script will process all images in the current directory
# and save augmented versions in the same directory

def rotate_image(image, angle_range=5):
    """Rotates an image by a small random angle"""
    # Choose a random angle within the range
    angle = random.uniform(-angle_range, angle_range)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    
    # Apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

def adjust_brightness(image, brightness_range=0.3):
    """Adjusts the brightness of an image"""
    # Choose a random brightness adjustment factor
    brightness_factor = random.uniform(1 - brightness_range, 1 + brightness_range)
    
    # Create a brighter/darker version
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    return adjusted_image

def zoom_crop_image(image, zoom_range=0.1):
    """Zooms in (crops) the image slightly"""
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate how much to crop
    crop_pixels_h = int(height * zoom_range)
    crop_pixels_w = int(width * zoom_range)
    
    # Random starting point for crop
    start_x = random.randint(0, crop_pixels_w)
    start_y = random.randint(0, crop_pixels_h)
    
    # Perform the crop
    cropped_image = image[start_y:height-crop_pixels_h, start_x:width-crop_pixels_w]
    
    # Resize back to original dimensions
    zoomed_image = cv2.resize(cropped_image, (width, height))
    
    return zoomed_image

def add_noise(image, noise_level=5):
    """Adds a small amount of random noise to the image"""
    # Create a noise array
    noise = np.random.randint(-noise_level, noise_level, image.shape, dtype=np.int16)
    
    # Add noise to the image and clip values to valid range
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return noisy_image

def main():
    # Get all image files in the current directory
    folder_path = r"all screenshots\absolutely no way"
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith('.jpg')]
    
    total_original = len(image_files)
    print(f"Found {total_original} original misc screenshots")
    
    # How many variations to create per image (we want about 100 total)
    variations_needed = 3  # Default to 3 variations per image
    
    if total_original < 30:  # If we have very few images, create more variations
        target_total = 100  # We want around 100 images total
        variations_needed = max(3, (target_total // total_original) - 1)
    
    print(f"Creating {variations_needed} variations of each image...")
    
    # Keep track of how many images we create
    augmented_count = 0
    
    # Process each image
    for img_file in image_files:
        # Load the image
        full_path = os.path.join(folder_path, img_file)
        image = cv2.imread(full_path)
        
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        # Generate augmented versions
        for i in range(variations_needed):
            augmented_image = image.copy()
            
            # Apply a random combination of augmentations
            # Make sure at least 2 augmentations are applied
            augmentations = [rotate_image, adjust_brightness, zoom_crop_image, add_noise]
            random.shuffle(augmentations)
            
            # Apply at least 2 random augmentations
            num_augs = random.randint(2, 4)  # Apply 2-4 augmentations
            for j in range(num_augs):
                augmented_image = augmentations[j](augmented_image)
            
            # Create a filename for the augmented image
            filename, ext = os.path.splitext(img_file)
            aug_filename = f"{filename}_aug{i+1}{ext}"
            
            # Save the augmented image
            cv2.imwrite(aug_filename, augmented_image)
            augmented_count += 1
            
            # Print progress
            if (augmented_count % 10) == 0:
                print(f"Created {augmented_count} augmented images so far...")
    
    # Final count
    total_images = total_original + augmented_count
    print(f"\nDone! Your folder now contains:")
    print(f"- {total_original} original misc screenshots")
    print(f"- {augmented_count} augmented versions")
    print(f"- {total_images} total misc images")

if __name__ == "__main__":
    main()