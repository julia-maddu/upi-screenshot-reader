import pandas as pd
import pytesseract
from PIL import Image
import re
import os
from tqdm import tqdm


def extract_text_from_image(image_path):

    try:
        img = Image.open(image_path)
        # Use custom config for better accuracy
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return None

def extract_utr_from_text(text):

    if not text:
        return 0
    
    # Remove all whitespace, newlines, and tabs for better matching
    cleaned_text = re.sub(r'\s+', '', text)
    
    # Pattern 1: Look for "UTR" followed by 12 digits
    pattern1 = r'UTR[:\s]*(\d{12})'
    match = re.search(pattern1, cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: Look for "UPI Transaction ID" followed by 12 digits
    pattern2 = r'UPITransactionID[:\s]*(\d{12})'
    match = re.search(pattern2, cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 3: Look for "Transaction ID" followed by 12 digits
    pattern3 = r'TransactionID[:\s]*(\d{12})'
    match = re.search(pattern3, cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # No UTR found
    return 0

def update_csv_with_utr(csv_path, image_folder, output_csv=None, verbose=False):

    # Read existing CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"\n{'='*60}")
    print("UTR Extraction from Screenshots")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} predictions from: {csv_path}")
    
    # Check if image folder exists
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    print(f"Image folder: {image_folder}")
    
    # Extract UTR for each image
    print("\nExtracting UTR numbers using OCR...")
    print("This may take a few minutes depending on the number of images.\n")
    
    utrs_found = 0
    utrs_not_found = 0
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        filename = row['filename']
        image_path = os.path.join(image_folder, filename)
        
        # Skip if image doesn't exist or prediction was ERROR
        if not os.path.exists(image_path):
            print(f"\nWarning: Image not found - {filename}")
            errors += 1
            continue
        
        if row['predicted_class'] == 'ERROR':
            continue
        
        # Extract text from image
        text = extract_text_from_image(image_path)
        
        if verbose and text:
            print(f"\n{'-'*60}")
            print(f"File: {filename}")
            print(f"Extracted text preview (first 200 chars):")
            print(text[:200])
            print(f"{'-'*60}")
        
        # Extract UTR from text
        utr = extract_utr_from_text(text)
        
        # Update DataFrame
        df.at[idx, 'utr_number'] = utr
        
        if utr != 0:
            utrs_found += 1
            if verbose:
                print(f"UTR found: {utr}")
        else:
            utrs_not_found += 1
            if verbose:
                print("UTR not found")
    
    # Print summary
    print(f"\n{'='*60}")
    print("UTR Extraction Complete!")
    print(f"{'='*60}")
    print(f"\nTotal images processed: {len(df)}")
    print(f"UTRs found: {utrs_found}")
    print(f"UTRs not found: {utrs_not_found}")
    if errors > 0:
        print(f"Errors/Missing images: {errors}")
        
    # Save updated CSV
    output_path = output_csv if output_csv else csv_path
    df.to_csv(output_path, index=False)
    print(f"\n✓ Updated results saved to: {os.path.abspath(output_path)}")
    
    return df

def main():

    CSV_PATH = "upi_predictions.csv"  
    IMAGE_FOLDER = "test_images"  
    OUTPUT_CSV = None  
    VERBOSE = False 
    
    print("UPI Screenshot UTR Extractor")
    print("=" * 60)
        
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"\nError: CSV file '{CSV_PATH}' not found!")
        print("\nPlease run 'batch_inference.py' first to generate predictions.")
        return
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"\nError: Image folder '{IMAGE_FOLDER}' not found!")
        print("\nPlease ensure the IMAGE_FOLDER variable points to your images.")
        return
    
    # Extract UTR numbers and update CSV
    try:
        df = update_csv_with_utr(
            csv_path=CSV_PATH,
            image_folder=IMAGE_FOLDER,
            output_csv=OUTPUT_CSV,
            verbose=VERBOSE
        )
        
        utrs_found_df = df[df['utr_number'] != 0]
        if len(utrs_found_df) > 0:
            print(f"\nSample of images with UTRs found:")
            print(utrs_found_df.head(10).to_string(index=False))
        
        print("\n✓ UTR extraction completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()