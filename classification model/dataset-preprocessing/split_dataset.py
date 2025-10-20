import os
import shutil
import random
import logging
from tqdm import tqdm
import math

# Configuration parameters
SOURCE_DIR = "second augment screenshots"
OUTPUT_DIR = "second-augment-split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
TEST_SPLIT = 0.2  # 20% for testing
RANDOM_SEED = 42  # For reproducibility
EXPECTED_CLASSES = ["fampay", "gpay", "invalid", "misc", "paytm", "phonepe"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("second-augment-split.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def create_directory_structure():
    """Create the necessary directory structure for train/test split"""
    # Create main output directory
    if os.path.exists(OUTPUT_DIR):
        logger.warning(f"Output directory '{OUTPUT_DIR}' already exists. Files may be overwritten.")
    else:
        os.makedirs(OUTPUT_DIR)
    
    # Create train directory
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    
    # Create test directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
    
    # Create class subdirectories in train and test
    for class_name in EXPECTED_CLASSES:
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

def split_dataset():
    """Split the dataset into training and testing sets"""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        logger.error(f"Source directory '{SOURCE_DIR}' not found!")
        return False
    
    # Create the directory structure
    create_directory_structure()
    
    # Statistics tracking
    stats = {
        "total_images": 0,
        "train_images": 0,
        "test_images": 0,
        "classes": {}
    }
    
    # Process each class directory
    source_classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    # Check for missing classes
    missing_classes = set(EXPECTED_CLASSES) - set(source_classes)
    if missing_classes:
        logger.warning(f"Missing expected classes: {missing_classes}")
    
    # Check for unexpected classes
    unexpected_classes = set(source_classes) - set(EXPECTED_CLASSES)
    if unexpected_classes:
        logger.warning(f"Found unexpected classes: {unexpected_classes}")
    
    # Process all classes
    for class_name in source_classes:
        class_dir = os.path.join(SOURCE_DIR, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Get all images in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if os.path.isfile(os.path.join(class_dir, f)) and 
                      f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Skip if no images found
        if not image_files:
            logger.warning(f"No images found in class directory: {class_name}")
            continue
        
        # Shuffle the images
        random.shuffle(image_files)
        
        # Calculate split
        num_test = math.ceil(len(image_files) * TEST_SPLIT)
        test_files = image_files[:num_test]
        train_files = image_files[num_test:]
        
        # Initialize class statistics
        stats["classes"][class_name] = {
            "total": len(image_files),
            "train": len(train_files),
            "test": len(test_files)
        }
        
        # Update total statistics
        stats["total_images"] += len(image_files)
        stats["train_images"] += len(train_files)
        stats["test_images"] += len(test_files)
        
        logger.info(f"Processing class '{class_name}': {len(image_files)} images "
                   f"({len(train_files)} train, {len(test_files)} test)")
        
        # Copy training images
        for img_file in tqdm(train_files, desc=f"Copying {class_name} training images"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(TRAIN_DIR, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Copy testing images
        for img_file in tqdm(test_files, desc=f"Copying {class_name} testing images"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(TEST_DIR, class_name, img_file)
            shutil.copy2(src_path, dst_path)
    
    return stats

def print_summary(stats):
    """Print a summary of the dataset split"""
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)
    
    print(f"\nTotal images: {stats['total_images']}")
    print(f"Training images: {stats['train_images']} ({stats['train_images']/stats['total_images']*100:.1f}%)")
    print(f"Testing images: {stats['test_images']} ({stats['test_images']/stats['total_images']*100:.1f}%)")
    
    print("\nClass distribution:")
    print("-"*50)
    print(f"{'Class':<15} {'Total':<8} {'Train':<8} {'Test':<8} {'Train %':<8} {'Test %':<8}")
    print("-"*50)
    
    for class_name, counts in stats["classes"].items():
        train_percent = counts["train"] / counts["total"] * 100
        test_percent = counts["test"] / counts["total"] * 100
        print(f"{class_name:<15} {counts['total']:<8} {counts['train']:<8} {counts['test']:<8} "
              f"{train_percent:<8.1f} {test_percent:<8.1f}")
    
    print("\nOutput directories:")
    print(f"Training data: {os.path.abspath(TRAIN_DIR)}")
    print(f"Testing data: {os.path.abspath(TEST_DIR)}")
    print("\nSplit complete! You can now use these directories for model training.")

if __name__ == "__main__":
    print(f"Starting dataset split process...")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Test split ratio: {TEST_SPLIT*100}%")
    print(f"Random seed: {RANDOM_SEED}")
    
    stats = split_dataset()
    
    if stats:
        print_summary(stats)
        logger.info("Dataset split completed successfully")
    else:
        print("Dataset split failed. Check the logs for details.")
        logger.error("Dataset split failed")