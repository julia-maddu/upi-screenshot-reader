# UPI Screenshot Reader

A pipeline for automated OCR on UPI transaction screenshots. It classifies screenshots by app (GPay, PhonePe, etc.), applies targeted OCR to extract UTR numbers, and outputs structured results.

-----

### ⚙️ Workflow

1.  **Upload Images**: Place your screenshots into the `input_images/` directory, create it within the root directory.
2.  **Add Model Weights**: Download the weights and place them in the `upi_classifier_weights/` directory.
3.  **Run Preprocessing**: Prepare the images for classification.
    ```bash
    python inference_preprocessing.py
    ```
4.  **Run Batch Inference**: Classify each screenshot by UPI app.
    ```bash
    python batch_inference.py
    ```
5.  **Run OCR Extraction**: Extract transaction details based on the classification.
    ```bash
    python ocr-read.py
    ```
6.  **Get Results**: Find the structured output in `upi_predictions.csv`.

-----

### 🚀 Setup

1.  **Create and activate a virtual environment:**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate the environment
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    # source venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download model weights** and place them in the `upi_classifier_weights/` folder.

-----

### 📊 Output

The final extracted data is saved in `upi_predictions.csv` with the following columns: `filename`, `apptype`, `transaction_id`

-----

### 📝 Note

  * Model files are intentionally excluded from this Git repository to keep it lightweight.
  * Please contact the owner to recieve model weights/dataset to run the code in the training notebook. 
