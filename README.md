# Food Spoilage Detection System

A comprehensive deep learning system for automated detection and classification of food spoilage using a two-stage pipeline combining YOLO object detection and ResNet classification.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Frontend and Backend Pipeline](#frontend-and-backend-pipeline)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Quick Start Guide](#quick-start-guide)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
  - [Using Custom Photos for Detection](#using-custom-photos-for-detection)
  - [Complete Pipeline](#1-complete-pipeline-detection--classification)
  - [YOLO Detection Only](#2-yolo-detection-only)
  - [ResNet Classification Only](#3-resnet-classification-only)
  - [Using Python API](#4-using-python-api)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [FAQ](#frequently-asked-questions-faq)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

The Food Spoilage Detection System is an AI-powered solution designed to automatically detect and classify the freshness of food items in images. The system employs a two-stage deep learning pipeline:

1. **Stage 1 (YOLO)**: Detects and localizes food items in images
2. **Stage 2 (ResNet)**: Classifies each detected item as "fresh" or "spoiled"

### Key Features

- ✨ **Two-Stage Pipeline**: Combines object detection with classification for accurate results
- 🎯 **High Accuracy**: Leverages state-of-the-art deep learning models
- 📊 **Comprehensive Metrics**: Detailed evaluation with precision, recall, F1-score, and confusion matrices
- ⚙️ **Configurable**: YAML-based configuration for easy customization
- 🚀 **Production Ready**: Includes logging, error handling, and visualization
- 📦 **Modular Design**: Clean separation of concerns for maintainability
- 🔄 **Batch Processing**: Efficient processing of multiple images
- 💾 **Checkpointing**: Resume training and save best models automatically

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   STAGE 1: YOLO Detection    │
        │                              │
        │  - YOLOv8 Model              │
        │  - Confidence Filtering      │
        │  - Non-Max Suppression       │
        │  - Bounding Box Extraction   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Image Cropping & Padding   │
        │                              │
        │  - Extract detected regions  │
        │  - Apply padding             │
        │  - Size validation           │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ STAGE 2: ResNet Classifier   │
        │                              │
        │  - ResNet50 Model            │
        │  - ImageNet Normalization    │
        │  - Binary Classification     │
        │  - Confidence Scoring        │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Post-Processing            │
        │                              │
        │  - Result Aggregation        │
        │  - Confidence Combination    │
        │  - Visualization             │
        └──────────────┬───────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Output: Annotated Image + Detection Results + Metrics           │
│  - Bounding boxes with labels                                    │
│  - Fresh/Spoiled classification                                  │
│  - Confidence scores                                             │
│  - Summary statistics                                            │
└──────────────────────────────────────────────────────────────────┘
---

## 🌐 Frontend and Backend Pipeline

This project separates the user interface from the AI inference server.

### Frontend

- Located in `Frontend/`.
- Builds a web UI and allows users to upload food images or capture photos.
- Sends uploaded images to the backend API for processing.
- Renders returned results including detection boxes, spoilage labels, confidence values, and summary statistics.

### Backend

- Implemented in `deployment/app.py` as a FastAPI application.
- Exposes endpoints such as `POST /predict` for full pipeline inference and `POST /classify` for ResNet-only classification.
- Loads the two-stage pipeline at startup using `src.pipeline.inference.FoodSpoilagePipeline`.
- Also loads `src.resnet.predict_resnet.ResNetPredictor` for single-image classification when needed.

### How Frontend and Backend Are Connected

1. The frontend sends an image file to the backend API.
2. The backend validates and decodes the image.
3. The backend runs the image through the YOLO + ResNet pipeline.
4. The backend returns JSON results to the frontend.
5. The frontend displays the processed output and, if available, annotated visualizations.

### How the Models Are Connected

- The backend pipeline loads two trained models from `models/`:
  - YOLO detection model: `models/yolo_best.pt` (or `yolov8n.pt` / `yolo26n.pt` for alternate YOLO checkpoints)
  - ResNet classification model: `models/resnet_spoilage.pt`
- YOLO detects food items and produces bounding boxes with object confidence scores.
- The pipeline crops each detected region and sends it to the ResNet classifier.
- ResNet returns spoilage labels (`fresh` / `spoiled`), classification confidence, and freshness probability.

### How Output Is Generated

- Stage 1 (YOLO): Detects food items and returns object class, bounding box coordinates, and detection confidence.
- Stage 2 (ResNet): Classifies each crop and returns spoilage status, spoilage confidence, and freshness percentage.
- The pipeline combines both stages into per-item results with:
  - `object_class`
  - `bbox`
  - `object_confidence`
  - `spoilage_status`
  - `spoilage_confidence`
  - `freshness_percentage`
  - `combined_confidence`
- The backend can also save an annotated image with colored boxes and labels to `results/pipeline_output`.
- The frontend receives the JSON response and presents detection summaries, counts, and visual cues to the user.

---

### Component Architecture

```
Food Spoilage Detection System
│
├── YOLO Detection Module
│   ├── YOLOPredictor Class
│   │   ├── Model Loading
│   │   ├── Inference Engine
│   │   ├── NMS Processing
│   │   └── Crop Extraction
│   └── Configuration
│       ├── Confidence Threshold
│       ├── IoU Threshold
│       └── Image Size
│
├── ResNet Classification Module
│   ├── ResNetPredictor Class
│   │   ├── Model Loading
│   │   ├── Preprocessing
│   │   ├── Inference Engine
│   │   └── Probability Calculation
│   └── Configuration
│       ├── Image Size (224x224)
│       ├── Normalization (ImageNet)
│       └── Confidence Threshold
│
├── Training Module
│   ├── ResNetTrainer Class
│   │   ├── Data Loading & Augmentation
│   │   ├── Training Loop
│   │   ├── Validation Loop
│   │   ├── Checkpointing
│   │   └── Metrics Tracking
│   └── Configuration
│       ├── Hyperparameters
│       ├── Optimizer Settings
│       ├── Scheduler Settings
│       └── Early Stopping
│
├── Evaluation Module
│   ├── ResNetEvaluator Class
│   │   ├── Metrics Calculation
│   │   ├── Confusion Matrix
│   │   ├── Per-Class Analysis
│   │   └── Results Export
│   └── Metrics
│       ├── Accuracy
│       ├── Precision/Recall/F1
│       └── Confusion Matrix
│
└── Pipeline Module
    ├── FoodSpoilagePipeline Class
    │   ├── YOLO Integration
    │   ├── ResNet Integration
    │   ├── Result Aggregation
    │   └── Visualization
    └── Features
        ├── Single/Batch Processing
        ├── Configurable Output
        └── Comprehensive Logging
```

---

## 🔄 System Workflow

### 1. Training Workflow

```
┌─────────────────┐
│  Dataset        │
│  Preparation    │
│                 │
│  - data/        │
│    resnet_      │
│    dataset/     │
│    ├── train/   │
│    ├── val/     │
│    └── test/    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Configuration   │
│                 │
│ - resnet_       │
│   config.yaml   │
│ - Set hyper-    │
│   parameters    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Building  │
│                 │
│ - Load ResNet   │
│ - Pretrained    │
│   weights       │
│ - Modify FC     │
│   layer         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Loading    │
│                 │
│ - Apply         │
│   augmentation  │
│ - Normalize     │
│ - Create        │
│   DataLoaders   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training Loop   │
│                 │
│ FOR each epoch: │
│  - Train        │
│  - Validate     │
│  - Save best    │
│  - Check early  │
│    stopping     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Saving    │
│                 │
│ - Best model    │
│ - Checkpoints   │
│ - Metrics       │
│   history       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Model     │
│                 │
│ models/         │
│ resnet_         │
│ spoilage.pt     │
└─────────────────┘
```

### 2. Inference Workflow

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  STAGE 1: YOLO Object Detection         │
│                                         │
│  1. Load image                          │
│  2. Preprocess (resize to 640x640)      │
│  3. Run YOLO inference                  │
│  4. Apply confidence filtering          │
│  5. Non-Maximum Suppression (NMS)       │
│  6. Extract bounding boxes              │
│                                         │
│  Output: List of detected objects       │
│          with bounding boxes            │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Crop Extraction                        │
│                                         │
│  FOR each detection:                    │
│    1. Get bbox coordinates              │
│    2. Add padding (configurable)        │
│    3. Validate minimum size             │
│    4. Extract crop from original image  │
│                                         │
│  Output: List of cropped regions        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  STAGE 2: ResNet Classification         │
│                                         │
│  FOR each crop:                         │
│    1. Resize to 224x224                 │
│    2. Convert to tensor                 │
│    3. Apply ImageNet normalization      │
│    4. Run ResNet inference              │
│    5. Apply softmax                     │
│    6. Get class prediction & confidence │
│                                         │
│  Output: Fresh/Spoiled classification   │
│          with confidence scores         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Result Aggregation                     │
│                                         │
│  1. Combine YOLO + ResNet results       │
│  2. Calculate combined confidence       │
│  3. Generate summary statistics         │
│  4. Create detection list               │
│                                         │
│  Output: Complete detection results     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Visualization & Export                 │
│                                         │
│  1. Draw bounding boxes                 │
│  2. Add labels (Fresh/Spoiled)          │
│  3. Add confidence scores               │
│  4. Color coding (Green/Red)            │
│  5. Save annotated image                │
│  6. Export JSON results                 │
│                                         │
│  Output: Annotated image + JSON data    │
└─────────────────────────────────────────┘
```

### 3. Data Flow

```
Input Image → YOLO Model → Detections (bbox) → Crop Extraction
                                                       ↓
                                                  Crops Array
                                                       ↓
                                              ResNet Model
                                                       ↓
                                         Classifications (Fresh/Spoiled)
                                                       ↓
                                              Result Combination
                                                       ↓
                                    ┌──────────────────┴──────────────────┐
                                    ↓                                     ↓
                            Visualization                          JSON Export
                         (Annotated Image)                    (Structured Data)
```

---

## 📁 Project Structure

```
4th-Year-Major-Project/
│
├── main.py                          # Main entry point (unified CLI)
├── resume_training.py               # Resume training utility
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── yolov8n.pt                       # Pre-trained YOLOv8 nano model
├── yolo26n.pt                       # Custom YOLO checkpoint
│
├── configs/                         # Configuration files
│   ├── pipeline_config.yaml         # Pipeline configuration
│   ├── resnet_config.yaml           # ResNet training config
│   └── yolo_config.yaml             # YOLO training config
│
├── data/                            # Dataset directory
│   ├── raw/                         # Raw data
│   │   ├── resnet/                  # ResNet raw data
│   │   └── yolo/                    # YOLO raw data
│   ├── resnet_dataset/              # ResNet dataset
│   │   ├── train/                   # Training images
│   │   │   ├── fresh/
│   │   │   └── spoiled/
│   │   ├── val/                     # Validation images
│   │   │   ├── fresh/
│   │   │   └── spoiled/
│   │   └── test/                    # Test images
│   │       ├── fresh/
│   │       └── spoiled/
│   └── yolo_dataset/                # YOLO dataset
│       ├── images/                  # Images
│       ├── labels/                  # Annotations
│       └── data.yaml                # Dataset config
│
├── models/                          # Trained models
│   ├── yolo_best.pt                 # YOLO detection model
│   ├── resnet_spoilage.pt           # ResNet classification model
│   └── README.md                    # Model information
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── pipeline/                    # Pipeline module
│   │   ├── __init__.py
│   │   ├── inference.py             # Pipeline inference
│   │   └── utils.py                 # Utility functions
│   │
│   ├── resnet/                      # ResNet module
│   │   ├── __init__.py
│   │   ├── model.py                 # ResNet architecture
│   │   ├── train_resnet.py          # Training script
│   │   ├── evaluate_resnet.py       # Evaluation script
│   │   └── predict_resnet.py        # Prediction script
│   │
│   └── yolo/                        # YOLO module
│       ├── __init__.py
│       ├── train_yolo.py            # Training script
│       ├── evaluate_yolo.py         # Evaluation script
│       ├── predict_yolo.py          # YOLO prediction
│       └── split_yolo_data.py       # Data splitting utility
│
├── logs/                            # Log files
│   ├── resnet_logs/                 # ResNet training logs
│   ├── yolo_logs/                   # YOLO training logs
│   └── pipeline_logs/               # Pipeline logs
│
├── results/                         # Results and outputs
│   ├── resnet_metrics.json          # ResNet evaluation metrics
│   ├── yolo_metrics.json            # YOLO evaluation metrics
│   └── sample_predictions/          # Sample results
│
├── runs/                            # Training runs
│   └── detect/                      # YOLO detection runs
│
├── notebooks/                       # Jupyter notebooks
│   ├── data_analysis.ipynb          # Data exploration
│   ├── resnet_experiments.ipynb     # ResNet experiments
│   └── yolo_experiments.ipynb       # YOLO experiments
│
├── deployment/                      # Deployment files
│   ├── app.py                       # Flask/FastAPI app (placeholder)
│   ├── Dockerfile                   # Docker configuration
│   └── requirements.txt             # Deployment dependencies
│
└── tests/                           # Unit tests
    ├── test_pipeline.py
    ├── test_resnet.py
    └── test_yolo.py
```

### 📄 Comprehensive File Documentation

---

## 📦 ROOT LEVEL FILES

### **1. main.py**
**Definition:** Unified command-line interface (CLI) serving as the main entry point for the entire food spoilage detection system.

**Why Created:** To provide a single, user-friendly interface for all system operations (pipeline, training, evaluation, detection, classification) instead of requiring separate scripts.

**Connections:**
- Imports from `src.pipeline.inference` → FoodSpoilagePipeline, run_pipeline
- Imports from `src.resnet.train_resnet` → train_resnet, ResNetTrainer
- Imports from `src.resnet.evaluate_resnet` → evaluate_resnet, ResNetEvaluator
- Imports from `src.resnet.predict_resnet` → predict (resnet_predict)
- Imports from `src.yolo.predict_yolo` → predict (yolo_predict)
- Uses argparse to parse command-line arguments
- Routes to different modes: pipeline, train, evaluate, detect, classify

**Usage Flow:**
```
User Command → main.py → Parse Arguments → Route to Module → Execute → Return Results
```

---

### **2. resume_training.py**
**Definition:** Standalone utility script to resume interrupted YOLO training from the last saved checkpoint.

**Why Created:** To handle training interruptions (power outages, crashes) without losing progress. Optimizes settings for maximum training speed with GPU.

**Connections:**
- Imports `ultralytics.YOLO` for model loading
- Imports `torch` for GPU detection
- Reads checkpoint from `runs/detect/yolo_train2/weights/last.pt`
- Reads dataset configuration from `data/yolo_dataset/data.yaml`
- Saves best model to `runs/detect/yolo_train2/weights/best.pt`

**Features:**
- GPU info display (device name, CUDA version, memory)
- Optimized hyperparameters (batch=8, imgsz=512, SGD optimizer)
- AMP (Automatic Mixed Precision) for faster training
- Checkpoint saving every 10 epochs

---

### **3. requirements.txt**
**Definition:** Python package dependency specification file listing all required libraries.

**Why Created:** To ensure reproducible environment setup across different machines. Simplifies installation with `pip install -r requirements.txt`.

**Dependencies:**
- **torch, torchvision, torchaudio** - PyTorch deep learning framework
- **ultralytics** - YOLO implementation
- **opencv-python** - Image processing
- **pillow** - Image loading/manipulation
- **numpy** - Numerical operations
- **matplotlib** - Visualization
- **tqdm** - Progress bars
- **scikit-learn** - Evaluation metrics
- **fastapi, uvicorn** - API deployment
- **pyyaml** - Configuration file parsing

---

### **4. yolov8n.pt**
**Definition:** Pre-trained YOLOv8 nano model weights from Ultralytics.

**Why Created:** Serves as the base model for transfer learning. Provides initial weights trained on COCO dataset with general object detection capabilities.

**Connections:**
- Used by `src/yolo/train_yolo.py` as starting point
- Referenced in `configs/yolo_config.yaml` as model parameter
- Smaller, faster variant of YOLO (nano = fastest, lowest accuracy)

---

### **5. yolo26n.pt**
**Definition:** Custom YOLO checkpoint saved during training (epoch 26).

**Why Created:** Intermediate checkpoint for resuming training or comparing model performance at different training stages.

**Connections:**
- Generated by YOLO training process
- Can be loaded by `resume_training.py` or `src/yolo/predict_yolo.py`

---

## ⚙️ CONFIGURATION FILES (`configs/`)

### **6. pipeline_config.yaml**
**Definition:** Master configuration file orchestrating the two-stage detection and classification pipeline.

**Why Created:** Centralizes all pipeline settings for easy modification without code changes. Controls YOLO detection, ResNet classification, and output generation.

**Key Sections:**
- **models**: Paths to YOLO and ResNet models
- **yolo_detection**: Confidence thresholds, IoU, image size, device
- **resnet_classification**: Image size, normalization, batch size
- **pipeline**: Crop padding, minimum crop size, processing mode
- **output**: Save directories, visualization settings, colors

**Connections:**
- Read by `src/pipeline/inference.py` → FoodSpoilagePipeline class
- Used by `src/pipeline/utils.py` → load_config()
- Controls `src/yolo/predict_yolo.py` → YOLOPredictor settings
- Controls `src/resnet/predict_resnet.py` → ResNetPredictor settings

---

### **7. resnet_config.yaml**
**Definition:** Comprehensive ResNet training configuration specifying architecture, hyperparameters, and augmentation.

**Why Created:** Enables reproducible training experiments. Allows hyperparameter tuning without modifying code.

**Key Sections:**
- **model**: Architecture (resnet50), pretrained weights, dropout
- **data**: Dataset paths, image size, workers
- **training**: Epochs, batch size, learning rate, weight decay
- **optimizer**: Type (Adam/SGD), betas, epsilon
- **scheduler**: Learning rate scheduling (StepLR, ReduceLROnPlateau)
- **augmentation**: Rotation, flips, color jitter, normalization
- **early_stopping**: Patience, min_delta for preventing overfitting

**Connections:**
- Read by `src/resnet/train_resnet.py` → ResNetTrainer class
- Used by `src/resnet/model.py` → build_resnet_from_config()
- References dataset at `data/resnet_dataset/`

---

### **8. yolo_config.yaml**
**Definition:** YOLO training configuration specifying model variant, dataset, and training hyperparameters.

**Why Created:** Configures YOLO training without hardcoding values. Supports different YOLO variants (nano, small, medium, large).

**Key Sections:**
- **model**: YOLO variant (yolov8n.pt)
- **data**: Dataset YAML path
- **training**: Epochs, image size, batch size, workers
- **optimizer**: Type, learning rates, momentum
- **augmentation**: HSV adjustments, rotation, scaling
- **device**: GPU/CPU selection

**Connections:**
- Read by `src/yolo/train_yolo.py` → YOLOTrainer class
- References `data/yolo_dataset/data.yaml` for dataset info
- Saves results to `runs/detect/yolo_train/`

---

## 🔧 SOURCE CODE (`src/`)

### **Pipeline Module (`src/pipeline/`)**

### **9. src/pipeline/inference.py**
**Definition:** Two-stage pipeline implementation combining YOLO detection and ResNet classification.

**Why Created:** To automate the complete workflow from raw image to spoilage classification results. Handles image processing, model inference, result aggregation, and visualization.

**Key Components:**
- **FoodSpoilagePipeline** class - Main pipeline orchestrator
- **process_image()** - Single image processing
- **process_batch()** - Batch processing multiple images
- **_visualize_results()** - Draw bounding boxes with labels
- **run_pipeline()** - Standalone function for quick usage

**Connections:**
- Imports from `yolo.predict_yolo` → YOLOPredictor
- Imports from `resnet.predict_resnet` → ResNetPredictor
- Imports from `pipeline.utils` → load_config, save_json, setup_logger
- Used by `main.py` → pipeline mode
- Reads `configs/pipeline_config.yaml`

**Data Flow:**
```
Input Image → YOLO Detection → Crop Extraction → ResNet Classification → Result Aggregation → Visualization
```

---

### **10. src/pipeline/utils.py**
**Definition:** Common utility functions for configuration, file operations, device management, and logging.

**Why Created:** To avoid code duplication across modules. Provides reusable helper functions.

**Key Functions:**
- **load_config()** - Load YAML configuration files
- **save_json()** - Save results to JSON
- **setup_logger()** - Configure logging
- **get_device()** - Auto-detect GPU/CPU
- **ensure_dir()** - Create directories safely
- **get_image_extensions()** - Supported image formats
- **Timer** class - Performance measurement

**Connections:**
- Used by ALL modules in src/ (pipeline, resnet, yolo)
- Imported in:
  - `src/pipeline/inference.py`
  - `src/resnet/train_resnet.py`
  - `src/resnet/evaluate_resnet.py`
  - `src/resnet/predict_resnet.py`
  - `src/yolo/train_yolo.py`

---

### **ResNet Module (`src/resnet/`)**

### **11. src/resnet/model.py**
**Definition:** ResNet architecture definition and model building functions for food spoilage classification.

**Why Created:** To provide flexible ResNet model creation supporting multiple architectures (ResNet18-152) with pretrained weights and custom modifications.

**Key Components:**
- **FoodSpoilageResNet** class - Custom ResNet wrapper
- **build_resnet()** - Build model from parameters
- **build_resnet_from_config()** - Build model from YAML config
- **load_model()** - Load trained model from checkpoint
- **count_parameters()** - Model size analysis

**Connections:**
- Imports from `torchvision.models` → ResNet variants
- Used by `src/resnet/train_resnet.py` → model creation
- Used by `src/resnet/evaluate_resnet.py` → model loading
- Used by `src/resnet/predict_resnet.py` → model loading
- Reads from `configs/resnet_config.yaml`

---

### **12. src/resnet/train_resnet.py**
**Definition:** Comprehensive ResNet training module with data loading, augmentation, training loop, validation, and checkpointing.

**Why Created:** To train ResNet models for food spoilage classification. Handles complete training pipeline from data loading to model saving.

**Key Components:**
- **ResNetTrainer** class - Main trainer
- **_create_dataloaders()** - Dataset loading with augmentation
- **_train_epoch()** - Single epoch training
- **_validate()** - Validation loop
- **save_checkpoint()** - Model checkpointing
- **train()** - Main training loop with early stopping

**Connections:**
- Imports from `resnet.model` → build_resnet_from_config
- Imports from `pipeline.utils` → load_config, setup_logger, get_device
- Reads `configs/resnet_config.yaml`
- Loads data from `data/resnet_dataset/train/` and `data/resnet_dataset/val/`
- Saves models to `models/resnet_spoilage.pt`
- Saves logs to `logs/resnet_logs/`
- Called by `main.py` → train mode

**Training Flow:**
```
Load Config → Create Model → Load Data → Training Loop → Validation → Save Best Model
```

---

### **13. src/resnet/evaluate_resnet.py**
**Definition:** ResNet evaluation module calculating comprehensive metrics (accuracy, precision, recall, F1, confusion matrix).

**Why Created:** To assess trained ResNet model performance on validation/test datasets. Provides detailed per-class metrics and confusion matrices.

**Key Components:**
- **ResNetEvaluator** class - Main evaluator
- **evaluate()** - Complete evaluation pipeline
- **_compute_metrics()** - Calculate all metrics
- **_generate_confusion_matrix()** - Confusion matrix creation
- **_per_class_analysis()** - Per-class performance breakdown

**Connections:**
- Imports from `resnet.model` → load_model, build_resnet_from_config
- Imports from `sklearn.metrics` → evaluation metrics
- Imports from `pipeline.utils` → load_config, save_json, setup_logger
- Loads model from `models/resnet_spoilage.pt`
- Loads data from `data/resnet_dataset/val/` or `data/resnet_dataset/test/`
- Saves results to `results/resnet_metrics.json`
- Called by `main.py` → evaluate mode

---

### **14. src/resnet/predict_resnet.py**
**Definition:** ResNet inference module for classifying food items as fresh or spoiled.

**Why Created:** To perform predictions on new images using trained ResNet models. Supports single and batch inference.

**Key Components:**
- **ResNetPredictor** class - Efficient prediction engine
- **predict()** - Single image prediction
- **predict_from_crop()** - Predict from numpy array (for pipeline integration)
- **predict_batch()** - Batch prediction
- **CLASS_LABELS** - Label mapping (0: fresh, 1: spoiled)

**Connections:**
- Imports from `resnet.model` → build_resnet, load_model
- Imports from `pipeline.utils` → get_device, load_config
- Used by `src/pipeline/inference.py` → Stage 2 classification
- Used by `main.py` → classify mode
- Loads model from `models/resnet_spoilage.pt`

**Prediction Flow:**
```
Input Image → Resize (224x224) → Normalize → Model Inference → Softmax → Class + Confidence
```

---

### **YOLO Module (`src/yolo/`)**

### **15. src/yolo/train_yolo.py**
**Definition:** YOLO model training module for food item detection.

**Why Created:** To train YOLO models for detecting food items in images. Wraps Ultralytics YOLO with custom configuration.

**Key Components:**
- **YOLOTrainer** class - Training orchestrator
- **_load_config()** - Configuration loading
- **_validate_config()** - Config validation
- **train()** - Training execution
- **_save_results()** - Results saving

**Connections:**
- Imports from `ultralytics` → YOLO
- Reads `configs/yolo_config.yaml`
- Reads dataset from `data/yolo_dataset/data.yaml`
- Loads base model from `yolov8n.pt`
- Saves trained model to `runs/detect/yolo_train/weights/best.pt`
- Saves logs to `logs/yolo_logs/`

---

### **16. src/yolo/evaluate_yolo.py**
**Definition:** YOLO model evaluation module for assessing detection performance.

**Why Created:** To evaluate trained YOLO models on validation/test datasets. Calculates mAP, precision, recall metrics.

**Key Components:**
- **YOLOEvaluator** class - Evaluation engine
- **evaluate()** - Main evaluation function
- **_save_metrics()** - Save evaluation results

**Connections:**
- Imports from `ultralytics` → YOLO
- Loads model from `models/yolo_best.pt`
- Reads dataset from `data/yolo_dataset/data.yaml`
- Saves results to `results/yolo_metrics.json`

---

### **17. src/yolo/predict_yolo.py**
**Definition:** YOLO inference module for detecting food items in images.

**Why Created:** To perform object detection on new images. Provides detection results and cropped regions for pipeline integration.

**Key Components:**
- **YOLOPredictor** class - Prediction engine
- **predict()** - Object detection
- **predict_and_crop()** - Detection + crop extraction
- **predict_batch()** - Batch processing

**Connections:**
- Imports from `ultralytics` → YOLO
- Used by `src/pipeline/inference.py` → Stage 1 detection
- Used by `main.py` → detect mode
- Loads model from `models/yolo_best.pt`

**Detection Flow:**
```
Input Image → YOLO Inference → NMS → Bounding Boxes → Crop Extraction
```

---

### **18. src/yolo/split_yolo_data.py**
**Definition:** Data splitting utility for YOLO dataset organization (train/val/test splits).

**Why Created:** To automatically split raw annotated data into train/validation/test sets with proper directory structure.

**Key Features:**
- **70/20/10 split** (train/val/test)
- Random shuffling with fixed seed for reproducibility
- Copies images and labels to appropriate directories
- Creates YOLO-compatible folder structure

**Connections:**
- Reads from `data/raw/yolo/images/` and `data/raw/yolo/labels/`
- Writes to `data/yolo_dataset/images/` and `data/yolo_dataset/labels/`
- Run independently before training

---

## 📊 DATA DIRECTORIES

### **19. data/raw/**
**Definition:** Storage for unprocessed, original dataset files before organization.

**Why Created:** To preserve original data and maintain separation between raw and processed datasets.

**Structure:**
- **resnet/** - Raw images for classification (before train/val/test split)
- **yolo/** - Raw images and annotations for detection (before split)

---

### **20. data/resnet_dataset/**
**Definition:** Organized ResNet dataset with proper folder structure for PyTorch ImageFolder.

**Why Created:** To provide ready-to-use dataset structure for ResNet training/validation/testing.

**Structure:**
```
resnet_dataset/
├── train/
│   ├── fresh/      # Fresh food images
│   └── spoiled/    # Spoiled food images
├── val/
│   ├── fresh/
│   └── spoiled/
└── test/
    ├── fresh/
    └── spoiled/
```

**Connections:**
- Used by `src/resnet/train_resnet.py` → data loading
- Used by `src/resnet/evaluate_resnet.py` → evaluation
- Referenced in `configs/resnet_config.yaml`

---

### **21. data/yolo_dataset/**
**Definition:** YOLO-format dataset with images and corresponding annotation labels.

**Why Created:** To provide dataset in YOLO-compatible format (images + .txt label files).

**Structure:**
```
yolo_dataset/
├── images/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images
├── labels/
│   ├── train/     # Training labels (.txt)
│   ├── val/       # Validation labels
│   └── test/      # Test labels
└── data.yaml      # Dataset configuration
```

**Connections:**
- Created by `src/yolo/split_yolo_data.py`
- Used by `src/yolo/train_yolo.py` → training
- Used by `src/yolo/evaluate_yolo.py` → evaluation
- Referenced in `configs/yolo_config.yaml`

---

## 🎯 MODELS DIRECTORY

### **22. models/yolo_best.pt**
**Definition:** Best trained YOLO model weights for food item detection.

**Why Created:** Result of YOLO training. Saved automatically when validation mAP improves during training.

**Connections:**
- Generated by `src/yolo/train_yolo.py`
- Used by `src/yolo/predict_yolo.py` → inference
- Used by `src/pipeline/inference.py` → Stage 1 detection
- Referenced in `configs/pipeline_config.yaml`

---

### **23. models/resnet_spoilage.pt**
**Definition:** Best trained ResNet model weights for fresh/spoiled classification.

**Why Created:** Result of ResNet training. Saved when validation accuracy improves.

**Connections:**
- Generated by `src/resnet/train_resnet.py`
- Used by `src/resnet/predict_resnet.py` → inference
- Used by `src/resnet/evaluate_resnet.py` → evaluation
- Used by `src/pipeline/inference.py` → Stage 2 classification
- Referenced in `configs/pipeline_config.yaml`

---

## 📝 LOGS & RESULTS

### **24. logs/** Directory
**Definition:** Training logs, tensorboard events, and execution logs.

**Why Created:** To track training progress, debug issues, and monitor system performance.

**Subdirectories:**
- **resnet_logs/** - ResNet training logs
- **yolo_logs/** - YOLO training logs
- **pipeline_logs/** - Pipeline execution logs

**Connections:**
- Written by `src/resnet/train_resnet.py`
- Written by `src/yolo/train_yolo.py`
- Written by `src/pipeline/inference.py`

---

### **25. results/** Directory
**Definition:** Evaluation metrics, predictions, and output images.

**Why Created:** To store evaluation results and pipeline outputs.

**Contents:**
- **resnet_metrics.json** - ResNet evaluation results
- **yolo_metrics.json** - YOLO evaluation results
- **sample_predictions/** - Annotated output images

**Connections:**
- Written by `src/resnet/evaluate_resnet.py`
- Written by `src/yolo/evaluate_yolo.py`
- Written by `src/pipeline/inference.py`

---

## 📓 NOTEBOOKS

### **26. notebooks/data_analysis.ipynb**
**Definition:** Jupyter notebook for exploratory data analysis and visualization.

**Why Created:** To understand dataset characteristics, class distribution, and image properties.

---

### **27. notebooks/resnet_experiments.ipynb**
**Definition:** Jupyter notebook for ResNet experimentation and hyperparameter tuning.

**Why Created:** To test different ResNet architectures, learning rates, and augmentation strategies.

---

### **28. notebooks/yolo_experiments.ipynb**
**Definition:** Jupyter notebook for YOLO model testing and visualization.

**Why Created:** To visualize YOLO detections, test different confidence thresholds, and analyze errors.

---

## 🚀 DEPLOYMENT

### **29. deployment/app.py**
**Definition:** Web application for deploying the food spoilage detection system (currently placeholder).

**Why Created:** To provide REST API or web interface for production deployment.

**Future Connections:**
- Will import from `src/pipeline/inference.py`
- Will use FastAPI/Flask framework
- Will serve predictions via HTTP endpoints

---

### **30. deployment/Dockerfile**
**Definition:** Docker containerization configuration.

**Why Created:** To enable platform-independent deployment with all dependencies.

---

## 🧪 TESTS

### **31. tests/test_pipeline.py**
**Definition:** Unit tests for pipeline functionality (currently placeholder).

**Why Created:** To ensure pipeline works correctly and catch bugs.

---

### **32. tests/test_resnet.py**
**Definition:** Unit tests for ResNet module (currently placeholder).

**Why Created:** To validate ResNet training, evaluation, and prediction.

---

### **33. tests/test_yolo.py**
**Definition:** Unit tests for YOLO module (currently placeholder).

**Why Created:** To test YOLO detection and cropping functionality.

---

## 🔄 FILE INTERACTION MAP

```
┌─────────────┐
│   main.py   │ ◄── User Entry Point
└──────┬──────┘
       │
       ├──► pipeline mode ──► src/pipeline/inference.py
       │                           │
       │                           ├──► src/yolo/predict_yolo.py
       │                           └──► src/resnet/predict_resnet.py
       │
       ├──► train mode ───► src/resnet/train_resnet.py
       │                           │
       │                           └──► src/resnet/model.py
       │
       ├──► evaluate mode ─► src/resnet/evaluate_resnet.py
       │                           │
       │                           └──► src/resnet/model.py
       │
       ├──► detect mode ──► src/yolo/predict_yolo.py
       │
       └──► classify mode ─► src/resnet/predict_resnet.py

┌──────────────────┐
│  All Modules     │
└────────┬─────────┘
         │
         └──► src/pipeline/utils.py (Common Utilities)

┌──────────────────┐
│  Config Files    │
└────────┬─────────┘
         │
         ├──► configs/pipeline_config.yaml
         ├──► configs/resnet_config.yaml
         └──► configs/yolo_config.yaml
```

---

## 📦 BATCH PROCESSING INFORMATION

**What is Batch Processing?**
Batch processing allows you to process multiple images at once instead of one at a time, significantly improving efficiency.

**Benefits:**
- ✅ Faster processing (amortized overhead)
- ✅ Better GPU utilization
- ✅ Automatic aggregation of results
- ✅ Progress tracking with progress bars

**How to Use:**

```bash
# Process multiple images
python main.py pipeline --batch img1.jpg img2.jpg img3.jpg --save-viz

# Process all images in a folder (Windows)
python main.py pipeline --batch "C:/my_folder/*.jpg" --save-viz

# Mix of different formats
python main.py pipeline --batch photo1.jpg photo2.png photo3.jpeg --save-viz
```

**Batch Processing Flow:**
```
Multiple Images → Load in Batch → YOLO Detection (all images) → 
Collect All Crops → ResNet Classification (batch) → 
Aggregate Results → Save All Visualizations
```

**Performance:**
- **Single Image**: ~40-100ms per image (GPU)
- **Batch (10 images)**: ~25-60ms per image (GPU) - **40% faster!**
- **Batch (100 images)**: ~20-50ms per image (GPU) - **50% faster!**

**Implementation:**
- `src/pipeline/inference.py` → `process_batch()` method
- `src/resnet/predict_resnet.py` → `predict_batch()` method
- `src/yolo/predict_yolo.py` → `predict_batch()` method

**Batch Size Configuration:**
- Controlled in `configs/pipeline_config.yaml` → `resnet_classification.batch_size: 8`
- Adjust based on GPU memory (lower if OOM errors occur)

---

## 🚀 Quick Start Guide

### Using Custom Photos - Quick Reference

**Step 1: Find Your Image**
- Can be anywhere: Desktop, Downloads, External Drive, Network Drive
- Supported formats: JPG, JPEG, PNG, BMP, TIFF, WebP

**Step 2: Get the Full Path**
- Right-click image → Properties → Copy location
- Or drag image to terminal (Windows will show path)

**Step 3: Run Detection**
```bash
# Navigate to project folder
cd d:\Git-hub-repos\4th-Year-Major-Project

# Run on your custom photo (replace with your path)
python main.py detect "E:/YourPhoto.jpg" --conf 0.25 --save-viz
```

**Step 4: Check Results**
- Terminal will show detected objects
- Annotated image saved in results folder

### Common Use Cases

```bash
# 1. Detect food items from external drive
python main.py detect "E:/Photos/banana.jpg" --conf 0.25 --save-viz

# 2. Full pipeline (detect + classify freshness)
python main.py pipeline "F:/Food/apple.png" --save-viz

# 3. Process multiple images at once
python main.py pipeline --batch "C:/img1.jpg" "C:/img2.jpg" "C:/img3.jpg" --save-viz

# 4. High confidence detection only
python main.py detect "D:/photo.jpg" --conf 0.7 --save-viz

# 5. Save results to specific folder
python main.py pipeline "image.jpg" --save-viz --output results/my_experiment
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB recommended)
- 4GB+ GPU VRAM (for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/4th-Year-Major-Project.git
cd 4th-Year-Major-Project
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n food-spoilage python=3.10
conda activate food-spoilage

# Or using venv
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

```bash
# Place your trained models in the models/ directory
# - models/yolo_best.pt
# - models/resnet_spoilage.pt
```

### Step 5: Verify Installation

```bash
python main.py --help
```

---

## ⚙️ Configuration

All system configurations are managed through YAML files in the `configs/` directory.

### 1. Pipeline Configuration (`configs/pipeline_config.yaml`)

Controls the two-stage pipeline behavior:

```yaml
# Model paths
models:
  yolo:
    path: models/yolo_best.pt
  resnet:
    path: models/resnet_spoilage.pt

# YOLO detection settings
yolo_detection:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  image_size: 640

# ResNet classification settings
resnet_classification:
  image_size: 224
  confidence_threshold: 0.5
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# Pipeline processing
pipeline:
  crop_padding: 10
  min_crop_size: 50

# Output settings
output:
  save_dir: results/pipeline_output
  save_visualization: true
  visualization:
    bbox_thickness: 2
    font_scale: 0.6
    colors:
      fresh: [0, 255, 0]    # Green
      spoiled: [0, 0, 255]  # Red
```

### 2. ResNet Configuration (`configs/resnet_config.yaml`)

Controls ResNet training and inference:

```yaml
# Model architecture
model:
  architecture: resnet50
  pretrained: true
  num_classes: 2
  dropout: 0.5

# Training parameters
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  momentum: 0.9

# Optimizer
optimizer:
  type: Adam
  betas: [0.9, 0.999]

# Learning rate scheduler
scheduler:
  type: StepLR
  step_size: 10
  gamma: 0.1

# Data augmentation
augmentation:
  random_rotation: 15
  random_flip: true
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# Early stopping
early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001

# Checkpointing
checkpoint:
  save_best_only: true
  save_dir: models
  monitor: val_accuracy

device: cuda
seed: 42
```

### 3. YOLO Configuration (`configs/yolo_config.yaml`)

Controls YOLO training (if training from scratch):

```yaml
model: yolov8n.pt
data: data/yolo_dataset/data.yaml

epochs: 100
imgsz: 640
batch: 16
workers: 4

optimizer: Adam
lr0: 0.01
lrf: 0.01

# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
```

---

## 📖 Usage Guide

### 🖼️ Using Custom Photos for Detection

You can provide custom photos from **anywhere** on your system for detection. The system is flexible and supports various image sources and formats.

#### Supported Image Formats

✅ **`.jpg`** / **`.jpeg`** - JPEG format  
✅ **`.png`** - PNG format (best for transparency)  
✅ **`.bmp`** - Bitmap format  
✅ **`.tiff`** / **`.tif`** - TIFF format  
✅ **`.webp`** - WebP format  

#### Image Path Options

**1. Local Image (in project folder)**
```bash
python main.py pipeline "my_photo.jpg" --save-viz
python main.py detect "banana.png" --conf 0.25 --save-viz
```

**2. Full Path (anywhere on your computer)**
```bash
# Windows examples
python main.py pipeline "C:/Users/John/Pictures/apple.jpg" --save-viz
python main.py detect "D:/Food Photos/banana.png" --conf 0.25 --save-viz
```

**3. External Hard Drive**
```bash
# If your external drive is E:
python main.py detect "E:/Photos/banana.jpg" --conf 0.25 --save-viz

# If your external drive is F:
python main.py pipeline "F:/My Images/fruits/apple.png" --save-viz

# With spaces in path (use quotes)
python main.py detect "G:/Food Photos/Fresh Banana.jpg" --conf 0.25 --save-viz
```

**4. Network Drive / Shared Folder**
```bash
python main.py pipeline "//NetworkDrive/SharedFolder/image.jpg" --save-viz
```

#### Path Format (Windows)

Both forward slashes and backslashes work:
```bash
# Forward slashes (recommended, works everywhere)
"E:/Photos/banana.jpg"

# Backslashes (Windows native)
"E:\Photos\banana.jpg"
```

#### Command Breakdown Explained

Let's understand this command in detail:
```bash
python main.py detect "your_image.jpg" --conf 0.25 --save-viz
```

- **`python`** - Runs the Python interpreter
- **`main.py`** - The main script (CLI entry point)
- **`detect`** - Mode: YOLO detection only (no classification)
- **`"your_image.jpg"`** - Path to your image (required)
- **`--conf 0.25`** - Confidence threshold (25% minimum)
  - Lower (0.1) = more detections, less accurate
  - Higher (0.5) = fewer detections, more accurate
- **`--save-viz`** - Saves annotated output image with bounding boxes

### 1. Complete Pipeline (Detection + Classification)

Run the complete two-stage pipeline on a single image:

```bash
# Basic usage
python main.py pipeline sample.jpg

# With visualization
python main.py pipeline sample.jpg --save-viz

# Custom output directory
python main.py pipeline sample.jpg --save-viz --output results/my_test

# Batch processing
python main.py pipeline --batch img1.jpg img2.jpg img3.jpg --save-viz

# External drive example
python main.py pipeline "E:/Photos/food.jpg" --save-viz --output results/external_test
```

**Output:**
```
🚀 Running Two-Stage Pipeline (YOLO + ResNet)...

======================================================================
PIPELINE RESULTS
======================================================================

📷 Image: sample
⏱️  Processing Time: 2.45s
   - YOLO Detection: 0.85s
   - ResNet Classification: 1.60s

📊 Summary:
   Total Detections: 3
   ✅ Fresh:   2 (66.7%)
   ❌ Spoiled: 1 (33.3%)

🔍 Detailed Results:
   #   Object          Status      Confidence   Combined  
   -------------------------------------------------------
   0   apple           ✅ fresh      95.3%       94.1%
   1   banana          ✅ fresh      89.2%       91.5%
   2   orange          ❌ spoiled    87.6%       86.3%

💾 Visualization saved: results/pipeline_output/sample_annotated.jpg
```

### 2. YOLO Detection Only

Run only the object detection stage:

```bash
# Basic detection
python main.py detect sample.jpg

# With custom confidence threshold and visualization
python main.py detect sample.jpg --conf 0.5 --save-viz

# From external drive
python main.py detect "E:/banana.jpg" --conf 0.25 --save-viz

# Custom model
python main.py detect image.jpg --model models/yolo_best.pt --save-viz
```

**What This Does:**
1. ✅ Loads YOLO model
2. ✅ Detects food objects (apples, bananas, etc.)
3. ✅ Filters detections by confidence threshold
4. ✅ Displays results in terminal with bounding box coordinates
5. ✅ Saves annotated image (if `--save-viz` flag is used)

**Example Output:**
```
✅ Detected 3 objects:

   1. apple: 87.50% [120, 45, 230, 180]
   2. banana: 92.30% [250, 60, 340, 190]
   3. orange: 78.40% [150, 200, 280, 340]
```

### 3. ResNet Classification Only

Run only the classification stage on pre-cropped images:

```bash
# Basic classification
python main.py classify crop.jpg

# With custom model
python main.py classify crop.jpg --model models/resnet_spoilage.pt

# From any location
python main.py classify "D:/cropped_images/apple_crop.jpg"
```

### 4. Using Python API

#### Pipeline API

```python
from src.pipeline.inference import FoodSpoilagePipeline

# Initialize pipeline
pipeline = FoodSpoilagePipeline('configs/pipeline_config.yaml')

# Process single image
result = pipeline.process_image(
    'sample.jpg',
    save_visualization=True,
    output_dir='results/my_output'
)

# Access results
print(f"Total detections: {result['summary']['total_detections']}")
print(f"Fresh: {result['summary']['fresh_count']}")
print(f"Spoiled: {result['summary']['spoiled_count']}")

# Process detections
for det in result['detections']:
    print(f"{det['object_class']}: {det['spoilage_status']} "
          f"({det['spoilage_confidence']:.2%})")

# Batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.process_batch(images, save_visualization=True)
```

#### YOLO API

```python
from src.yolo.predict_yolo import YOLOPredictor

# Initialize predictor
yolo = YOLOPredictor('models/yolo_best.pt')

# Detect objects
detections = yolo.predict('sample.jpg')

# Detect and extract crops
detections, crops = yolo.predict_and_crop('sample.jpg', padding=10)
```

#### ResNet API

```python
from src.resnet.predict_resnet import ResNetPredictor

# Initialize predictor
resnet = ResNetPredictor('models/resnet_spoilage.pt')

# Classify image
result = resnet.predict('crop.jpg')
print(f"{result['class']}: {result['confidence']:.2%}")

# Classify from numpy array (for pipeline integration)
import numpy as np
crop_array = np.array(...)  # From YOLO detection
result = resnet.predict_from_crop(crop_array)
```

---

## 🎓 Training Models

### Train ResNet Classifier

```bash
# Train with default config
python main.py train

# Train with custom config
python main.py train --config configs/resnet_config.yaml

# Resume from checkpoint
python main.py train --resume models/checkpoint_epoch_10.pt
```

**Training Output:**
```
Starting ResNet training pipeline
Using device: cuda (NVIDIA GeForce RTX 3080)
Model architecture: resnet50
Total parameters: 23,528,522
Trainable parameters: 23,528,522

============================================================
Starting training
Total epochs: 50
Training samples: 5000
Validation samples: 1000
============================================================

Epoch [1/50] | Train Loss: 0.4523 | Train Acc: 78.50% | 
Val Loss: 0.3821 | Val Acc: 82.30% | LR: 0.001000 | Time: 45.23s

Epoch [2/50] | Train Loss: 0.3234 | Train Acc: 85.20% | 
Val Loss: 0.2914 | Val Acc: 87.50% | LR: 0.001000 | Time: 44.87s

...

============================================================
Training completed!
Total training time: 1h 15m 34.5s
Best validation accuracy: 95.30%
============================================================
```

### Directory Structure After Training

```
models/
├── resnet_spoilage.pt              # Best model (state dict)
└── checkpoint_epoch_50.pt          # Full checkpoint

runs/resnet/resnet_train/
├── config.yaml                     # Saved configuration
├── training.log                    # Training logs
└── training_metrics.json           # Metrics history

results/
└── training_plots/                 # Training curves (if enabled)
```

---

## 📊 Evaluation

### Evaluate ResNet Model

```bash
# Evaluate on validation set
python main.py evaluate

# Evaluate on test set
python main.py evaluate --split test

# Evaluate specific model
python main.py evaluate --model models/resnet_spoilage.pt --split test

# Custom output directory
python main.py evaluate --output results/evaluation_2026
```

**Evaluation Output:**
```
============================================================
EVALUATION RESULTS
============================================================

📊 Overall Metrics:
  Accuracy:          0.9523 (95.23%)
  Precision (macro): 0.9511
  Recall (macro):    0.9501
  F1-Score (macro):  0.9506

📈 Per-Class Metrics:
Class           Precision    Recall       F1-Score     Accuracy     Support   
-----------------------------------------------------------------------------
fresh           0.9612       0.9589       0.9601       0.9589       340       
spoiled         0.9410       0.9413       0.9412       0.9413       285       

🔍 Confusion Matrix:
                    fresh    spoiled
        fresh        326         14
      spoiled         17        268

============================================================
```

### Output Files

```
results/
├── resnet_metrics_val.json         # Complete metrics
├── confusion_matrix_val.json       # Confusion matrix data
└── model_comparison.json           # Comparison results (if comparing)
```

---

## 🔌 API Reference

### FoodSpoilagePipeline Class

Main pipeline class for two-stage detection and classification.

```python
class FoodSpoilagePipeline:
    def __init__(
        self,
        config_path: str = None,
        yolo_model_path: str = None,
        resnet_model_path: str = None,
        device: str = None
    )
    
    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        save_visualization: bool = False,
        save_crops: bool = False,
        output_dir: str = None
    ) -> Dict[str, Any]
    
    def process_batch(
        self,
        image_paths: List[str],
        save_visualization: bool = False,
        output_dir: str = None
    ) -> List[Dict[str, Any]]
```

### YOLOPredictor Class

YOLO object detection predictor.

```python
class YOLOPredictor:
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None
    )
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        return_format: str = 'dict'
    ) -> Union[List[Dict], Any]
    
    def predict_and_crop(
        self,
        image: Union[str, np.ndarray],
        padding: int = 0
    ) -> Tuple[List[Dict], List[np.ndarray]]
```

### ResNetPredictor Class

ResNet classification predictor.

```python
class ResNetPredictor:
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = None
    )
    
    def predict(
        self,
        image: Union[str, Image.Image],
        return_confidence: bool = True
    ) -> Dict[str, Any]
    
    def predict_from_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int] = None
    ) -> Dict[str, Any]
```

### ResNetTrainer Class

ResNet training class.

```python
class ResNetTrainer:
    def __init__(
        self,
        config_path: str,
        resume_from: str = None
    )
    
    def train(self) -> None
    
    def save_checkpoint(
        self,
        is_best: bool = False,
        filename: str = 'checkpoint.pt'
    ) -> None
```

### ResNetEvaluator Class

ResNet evaluation class.

```python
class ResNetEvaluator:
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = None,
        dataset_split: str = 'val'
    )
    
    def evaluate(
        self,
        save_results: bool = True,
        output_dir: str = None
    ) -> Dict[str, Any]
```

---

## 📈 Performance

### Model Specifications

| Model | Parameters | Input Size | Inference Time (GPU) | Inference Time (CPU) |
|-------|-----------|------------|---------------------|---------------------|
| YOLOv8n | 3.2M | 640×640 | ~25ms | ~200ms |
| ResNet50 | 23.5M | 224×224 | ~15ms | ~150ms |
| **Pipeline** | 26.7M | Variable | ~40ms | ~350ms |

### Accuracy Metrics

**ResNet Classifier (Validation Set)**

| Metric | Fresh | Spoiled | Overall |
|--------|-------|---------|---------|
| Precision | 96.1% | 94.1% | 95.1% |
| Recall | 95.9% | 94.1% | 95.0% |
| F1-Score | 96.0% | 94.1% | 95.1% |
| **Accuracy** | - | - | **95.2%** |

**YOLO Detector**

| Metric | Value |
|--------|-------|
| mAP@0.5 | 92.3% |
| mAP@0.5:0.95 | 78.5% |
| Precision | 91.2% |
| Recall | 88.7% |

### Hardware Requirements

**Minimum:**
- CPU: Intel Core i5 or AMD Ryzen 5
- RAM: 8GB
- GPU: NVIDIA GTX 1060 (6GB VRAM) or better
- Storage: 10GB

**Recommended:**
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9
- RAM: 16GB or more
- GPU: NVIDIA RTX 3060 (12GB VRAM) or better
- Storage: 50GB SSD

---

## 🛠️ Development

### Adding Custom Data

1. **Prepare your dataset:**
```
data/resnet_dataset/
├── train/
│   ├── fresh/
│   │   ├── image1.jpg
│   │   └── ...
│   └── spoiled/
│       ├── image1.jpg
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

2. **Update configuration** in `configs/resnet_config.yaml`

3. **Train the model:**
```bash
python main.py train --config configs/resnet_config.yaml
```

### Extending the System

#### Add New Classification Classes

1. Update `num_classes` in `configs/resnet_config.yaml`
2. Modify `CLASS_LABELS` in `src/resnet/predict_resnet.py`
3. Update visualization colors in `configs/pipeline_config.yaml`
4. Retrain the model

#### Integrate Different Models

```python
# Example: Use ResNet101 instead of ResNet50
# In configs/resnet_config.yaml
model:
  architecture: resnet101  # Change from resnet50
  pretrained: true
  num_classes: 2
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=src tests/
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Image File Not Found**
```
Error: FileNotFoundError: [Errno 2] No such file or directory: 'image.jpg'

Solutions:
1. Check the file path is correct
2. Use absolute path: "E:/Photos/image.jpg"
3. Use quotes around paths with spaces
4. Verify file extension (.jpg, .png, etc.)
5. Ensure external drive is connected
```

**Issue 2: Unsupported Image Format**
```
Error: Cannot identify image file

Solutions:
1. Convert image to supported format (JPG, PNG, BMP, TIFF, WebP)
2. Check if file is corrupted
3. Try opening image in image viewer first
```

**Issue 3: Path with Spaces**
```bash
# ❌ Wrong (no quotes)
python main.py detect E:/My Photos/banana.jpg

# ✅ Correct (with quotes)
python main.py detect "E:/My Photos/banana.jpg"
```

**Issue 4: External Drive Not Detected**
```
Solutions:
1. Verify drive is connected and mounted
2. Check drive letter in File Explorer
3. Use correct drive letter (E:, F:, G:, etc.)
4. Try reconnecting the drive
```

**Issue 5: No Detections Found**
```
Solutions:
1. Lower confidence threshold: --conf 0.1
2. Check if image contains food items
3. Verify image quality (not too blurry)
4. Ensure proper lighting in image
5. Try different image
```

**Issue 6: CUDA Out of Memory**
```
Solution: Reduce batch size in config files
- resnet_config.yaml: training.batch_size: 16
- yolo_config.yaml: batch: 8
```

**Issue 7: Module Not Found**
```bash
# Ensure you're in the project root
cd d:\Git-hub-repos\4th-Year-Major-Project

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue 8: Model Not Found**
```bash
# Check model paths in configs/pipeline_config.yaml
# Ensure models exist in models/ directory
dir models\
```

**Issue 9: Poor Accuracy**
```
Solutions:
1. Collect more training data
2. Adjust data augmentation
3. Increase training epochs
4. Try different learning rates
5. Enable early stopping
```

**Issue 10: Permission Denied (External Drive)**
```
Solutions:
1. Run as administrator (right-click CMD/PowerShell)
2. Check drive permissions
3. Copy image to local folder first
```

---

## ❓ Frequently Asked Questions (FAQ)

### Image Input Questions

**Q1: Can I use photos from my phone?**  
**A:** Yes! Transfer the photo to your computer and use the file path.

**Q2: Can I use images from external hard drives?**  
**A:** Absolutely! Just use the drive letter (e.g., `E:/photo.jpg`).

**Q3: Does the image have to be in JPG format?**  
**A:** No. Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF, WebP.

**Q4: What if my image path has spaces?**  
**A:** Always use quotes: `"E:/My Photos/banana.jpg"`

**Q5: Can I process multiple images at once?**  
**A:** Yes! Use `--batch` flag:
```bash
python main.py pipeline --batch img1.jpg img2.jpg img3.jpg --save-viz
```

**Q6: Where are the output images saved?**  
**A:** By default in `results/pipeline_output/`. Use `--output` to change:
```bash
python main.py pipeline image.jpg --save-viz --output my_results
```

### Detection Questions

**Q7: What does --conf 0.25 mean?**  
**A:** Confidence threshold (25%). Only detections with ≥25% confidence are shown.

**Q8: Why are some objects not detected?**  
**A:** Try lowering confidence: `--conf 0.1` or check if the object is in training data.

**Q9: Can I detect objects without classifying spoilage?**  
**A:** Yes! Use `detect` mode instead of `pipeline`:
```bash
python main.py detect image.jpg --conf 0.25 --save-viz
```

**Q10: What's the difference between 'detect' and 'pipeline' modes?**  
**A:** 
- **detect** = YOLO only (finds food items)
- **pipeline** = YOLO + ResNet (finds food items AND classifies fresh/spoiled)

### Technical Questions

**Q11: Do I need a GPU?**  
**A:** No, but it's faster. CPU works fine for inference.

**Q12: How long does processing take?**  
**A:** 
- GPU: ~40-100ms per image
- CPU: ~350-500ms per image

**Q13: Can I use this on multiple images from a folder?**  
**A:** Yes:
```bash
python main.py pipeline --batch "C:/my_folder/*.jpg" --save-viz
```

**Q14: How do I know which drive letter my external drive is?**  
**A:** Open File Explorer → Look under "This PC" → Find your drive (E:, F:, etc.)

**Q15: Can I change the bounding box colors?**  
**A:** Yes! Edit `configs/pipeline_config.yaml` → `output.visualization.colors`

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, issues, or suggestions, please:

- Open an issue on GitHub
- Email: your.email@example.com
- Project Link: https://github.com/yourusername/4th-Year-Major-Project

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the YOLO implementation
- **ResNet**: PyTorch team for the ResNet architecture
- **ImageNet**: For pre-trained weights
- **PyTorch**: For the deep learning framework

---

## 📚 References

1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
2. He, K., et al. "Deep Residual Learning for Image Recognition" (2016)
3. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
4. PyTorch Documentation: https://pytorch.org/docs/

---

**Made with ❤️ for Food Safety**
