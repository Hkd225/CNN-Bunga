# CIFAR-100 Flower Classification
## Transfer Learning with MobileNetV2 & Multi-Platform Deployment
### By Muhammad Auffa Hakim Aditya

This project presents a comprehensive Deep Learning pipeline that extracts specific flower classes from the raw CIFAR-100 dataset, trains a highly accurate image classifier using Transfer Learning (MobileNetV2), and prepares the model for multi-platform deployment on Web (TensorFlow.js) and Mobile/Edge (TensorFlow Lite).

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate end-to-end Machine Learning Engineering: from low-level data extraction (parsing Python pickle files) and dynamic data augmentation, to fine-tuning a pre-trained model and generating production-ready deployment artifacts.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Low-Level Data Processing: Automatically download the CIFAR-100 dataset and decode the raw byte/pickle files.
2. Targeted Extraction: Isolate and extract specific flower classes (e.g., orchid, poppy, rose, sunflower, tulip) into a structured image directory format.
3. Pipeline Optimization: Utilize `tf.data` for efficient loading, caching, and prefetching, combined with on-the-fly Data Augmentation to prevent overfitting.
4. Transfer Learning (MobileNetV2):
   - Stage 1: Feature Extraction (Training a custom top-layer while freezing the base model).
   - Stage 2: Fine-Tuning (Unfreezing the top 100 layers of MobileNetV2 with a very low learning rate to maximize accuracy).
5. Advanced Callbacks: Implement custom Early Stopping, Learning Rate reduction, and a custom `StopAtAccuracy` callback to halt training dynamically.
6. Multi-Format Export: Automatically generate deployment artifacts for Python (`.keras`, `SavedModel`), Mobile (`.tflite`), and Web (`tfjs`).

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (fedesoriano/cifar100)
Original Format : Python Pickle (cPickle)
Target Classes  : Top 4 Flower Categories extracted dynamically based on data distribution.
Input Shape     : Resized to (224, 224, 3) to match MobileNetV2 optimal input architecture.

------------------------------------------------------------

PIPELINE ARCHITECTURE

1. Preprocessing:
   - Built-in `MobileNetV2.preprocess_input` for standard scaling.
   - Augmentation: Random Flip, Rotation, Zoom, Contrast, and Translation.

2. Model Architecture:
   - Base Model: MobileNetV2 (pre-trained on ImageNet, without top classification layer).
   - Custom Classification Head: 
     - GlobalAveragePooling2D
     - BatchNormalization
     - Dense (256 units, ReLU, L2 Regularization)
     - Dropout (0.5)
     - Output Dense (Softmax)

------------------------------------------------------------

MODEL EVALUATION & DEPLOYMENT ARTIFACTS

The script automatically generates a highly structured `submission/` directory containing everything needed to deploy the model or submit it for production review.

Evaluation Metrics Included:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Visualization

Exported Directory Structure:
submission/
    ├── tfjs_model/ (For web deployment using TensorFlow.js)
    ├── tflite/ 
    │   ├── model.tflite (For Android/iOS/IoT deployment)
    │   └── label.txt
    ├── saved_model/ (TensorFlow native format)
    ├── klasifikasi-bunga-cifar100.keras
    ├── README.md
    ├── requirements.txt
    └── notebook.ipynb

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install tensorflow kagglehub split-folders scikit-learn seaborn tensorflowjs

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/cifar100-flower-classification.git

2. Run the Python script or Notebook. The script will automatically parse the CIFAR-100 pickles, build the directories, train the two-stage model, plot the confusion matrix, and create the entire `submission/` folder containing the TFJS and TFLite models.
3. You can also upload a custom image at the end of the script to test live inference.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Deep Learning & Computer Vision
- Low-Level Data Parsing (Pickle to Image arrays)
- Transfer Learning & Fine-Tuning strategies
- TensorFlow.js (TFJS) Web Deployment
- TensorFlow Lite (TFLite) Mobile Deployment

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Image Classification
- Transfer Learning MobileNetV2
- CIFAR-100 Flowers
- TensorFlow Lite TFLite
- TensorFlow.js TFJS
- Deep Learning Portfolio
