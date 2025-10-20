import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

# Import preprocessing functions
from inference_preprocessing import (
    preprocess_single_image,
    preprocess_batch_images,
    get_image_files_from_folder
)

class UPIClassifier:

    def __init__(self, weights_dir="upi_classifier_weights"):

        self.weights_dir = weights_dir
        self.model = None
        self.class_names = None
        self.metadata = None
        
        self._load_model_from_weights()
    
    def _reconstruct_model(self, num_classes=6, input_shape=(224, 224, 3)):

        # Create the base model from EfficientNetB0
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,  # Don't load ImageNet weights, we'll load our trained weights
            input_shape=input_shape,
            pooling="avg"
        )
        
        # Build the model with custom classification head (matches training)
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        
        return model
    
    def _load_model_from_weights(self):
        """Load the model by reconstructing architecture and loading weights."""
        # Check if directory exists
        if not os.path.exists(self.weights_dir):
            raise FileNotFoundError(f"Weights directory not found: {self.weights_dir}")
        
        # Load metadata first
        metadata_path = os.path.join(self.weights_dir, "model_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.class_names = self.metadata['class_names']
        num_classes = self.metadata['num_classes']
        input_shape = tuple(self.metadata['input_shape'])
        
        print(f"Reconstructing model architecture...")
        print(f"Classes: {self.class_names}")
        print(f"Input shape: {input_shape}")
        
        # Reconstruct the model architecture
        self.model = self._reconstruct_model(num_classes=num_classes, input_shape=input_shape)
        
        # Load weights
        weights_path = os.path.join(self.weights_dir, "model_weights.weights.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading weights from {weights_path}...")
        self.model.load_weights(weights_path)
        print("✓ Model loaded successfully with weights")
        
        # Compile model for inference
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def predict_single(self, image_path):

        # Preprocess image
        processed_img = preprocess_single_image(image_path)
        if processed_img is None:
            return {
                'filename': os.path.basename(image_path),
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'all_probabilities': None
            }
        
        # Add batch dimension
        batch_img = np.expand_dims(processed_img, axis=0)
        
        # Predict
        predictions = self.model.predict(batch_img, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        return {
            'filename': os.path.basename(image_path),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': predictions[0].tolist()
        }
    
    def predict_batch(self, image_folder, batch_size=32, save_csv=True, output_path="predictions.csv"):

        print(f"\n{'='*60}")
        print(f"Starting Batch Inference")
        print(f"{'='*60}")
        
        # Get all image files
        print(f"\nScanning folder: {image_folder}")
        image_paths = get_image_files_from_folder(image_folder)
        
        if len(image_paths) == 0:
            print("No images found in the folder!")
            return pd.DataFrame(columns=['filename', 'predicted_class', 'amount', 'utr_number'])
        
        print(f"Found {len(image_paths)} images")
        
        # Store results
        results = []
        
        # Process in batches
        print(f"\nProcessing images in batches of {batch_size}...")
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batches"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_array, valid_paths = preprocess_batch_images(batch_paths)
            
            if batch_array is None:
                # All images in batch failed to process
                for path in batch_paths:
                    results.append({
                        'filename': os.path.basename(path),
                        'predicted_class': 'ERROR'
                    })
                continue
            
            # Predict
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Process predictions
            for path, pred in zip(valid_paths, predictions):
                predicted_idx = np.argmax(pred)
                predicted_class = self.class_names[predicted_idx]
                
                results.append({
                    'filename': os.path.basename(path),
                    'predicted_class': predicted_class,
                    'amount': 0,
                    'utr_number': 0
                })
            
            # Handle failed images in batch
            failed_paths = set(batch_paths) - set(valid_paths)
            for path in failed_paths:
                results.append({
                    'filename': os.path.basename(path),
                    'predicted_class': 'ERROR',
                    'amount': 0,
                    'utr_number': 0
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        df = df[['filename', 'predicted_class', 'utr_number']]
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Inference Complete!")
        print(f"{'='*60}")
        print(f"\nTotal images processed: {len(df)}")
        print(f"\nPrediction distribution:")
        print(df['predicted_class'].value_counts())
                
        # Save to CSV
        if save_csv:
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to: {os.path.abspath(output_path)}")
        
        return df

def main():

    # Configuration
    WEIGHTS_DIR = "upi_classifier_weights" 
    IMAGE_FOLDER = "test_images"  
    OUTPUT_CSV = "upi_predictions.csv"  
    BATCH_SIZE = 32  
    
    print("UPI Screenshot Classifier - Batch Inference")
    print("=" * 60)
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"\nError: Image folder '{IMAGE_FOLDER}' not found!")
        print("\nPlease update the IMAGE_FOLDER variable in the script to point to your test images.")
        return
    
    # Initialize classifier
    try:
        classifier = UPIClassifier(weights_dir=WEIGHTS_DIR)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease ensure:")
        print(f"  1. '{WEIGHTS_DIR}' folder exists")
        print(f"  2. 'model_weights.weights.h5' is inside the folder")
        print(f"  3. 'model_metadata.json' is inside the folder")
        return
    
    # Run batch inference
    try:
        df = classifier.predict_batch(
            image_folder=IMAGE_FOLDER,
            batch_size=BATCH_SIZE,
            save_csv=True,
            output_path=OUTPUT_CSV
        )
        
        # Display first few predictions
        print("\nSample of predictions:")
        print(df.head(20).to_string(index=False))
        
        print("\n✓ Batch inference completed successfully!")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()