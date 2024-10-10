
---

# Diabetic Retinopathy Detection Using CNN

This project provides a complete pipeline for detecting diabetic retinopathy (DR) from retinal images using a Convolutional Neural Network (CNN). The project includes training, validation, and testing scripts, with all necessary details for reproducing the results.

## Diabetic Retinopathy Classes:
- **0**: No Diabetic Retinopathy
- **1**: Mild
- **2**: Moderate
- **3**: Severe
- **4**: Proliferative Diabetic Retinopathy

## Dataset
The dataset used in this project can be found on Kaggle:  
[Diabetic Retinopathy Dataset (224x224 Gaussian Filtered)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)

The dataset consists of retinal images categorized into 5 folders based on the severity of diabetic retinopathy (No_DR, Mild, Moderate, Severe, Proliferative_DR). Additionally, a CSV file `data_labels.csv` provides the image file names and their associated diagnosis.

### Example of `data_labels.csv`:
```plaintext
id_code, diagnosis
1_left, 0
1_right, 1
2_left, 2
2_right, 0
```

## Project Components

The repository contains the following main components:

1. **`simple_cnn_model_training.py`**: This script is responsible for training a CNN model on a dataset of retinal images.
2. **`simple_cnn_model_testing.ipynb`**: This Jupyter Notebook is used for validating and testing the trained model.
3. **`requirements.txt`**: A file containing the list of required libraries for running the scripts.

## Requirements
Install the required libraries using the following command in your terminal:
```bash
pip install -r requirements.txt
```

## Model Training

The script `simple_cnn_model_training.py` loads the images and labels from the dataset, processes them, and trains a CNN model. Below are the key steps:

1. **Loading images**: Images are read from the dataset folder, resized to 224x224, and normalized.
2. **Model architecture**: A simple CNN model is built using TensorFlow/Keras.
3. **Class weights**: Class weights are calculated to address the class imbalance in the dataset.
4. **Training**: The model is trained for 10 epochs using a batch size of 32.
5. **Model saving**: The trained model is saved in `.keras` format.

## Model Testing

The `simple_cnn_model_testing.ipynb` notebook allows you to load the trained model, run predictions on validation images, and test its performance. Additionally, it includes functionality for making predictions on user-provided images.

### Testing Steps:
1. **Load Preprocessed Images**: Load and preprocess the images for testing using the same process as in training.
2. **Load Trained Model**: The trained model can be loaded using TensorFlow's `load_model` function.
3. **Make Predictions**: Predictions are made using the model, and you can compare them with actual labels.
4. **Generate Report**: The script generates a classification report and confusion matrix to evaluate model performance.

## Google Drive Model Link
Since GitHub has size limitations on file uploads, the trained model (`modelx.keras`) is uploaded to Google Drive. You can download it from the following link and use it for predictions and testing:

[Download Trained Model](https://drive.google.com/file/d/1rLMV2CGg39M-JayQY3im1RQsmoacCxsV/view?usp=sharing)

## Results
At the end of testing, the classification report and confusion matrix are generated to evaluate the model's performance across all classes. 

### Model Performance:
- **Accuracy**: 96% on the given dataset.

---

