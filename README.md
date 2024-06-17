# SSD-object-detection-PyTorch

## Description

This project implements an SSD (Single Shot Multibox Detector) object detection model using PyTorch and the Pascal VOC datasets. The model is capable of identifying and localizing multiple objects in an image with bounding boxes. The implementation follows best practices and includes advanced data augmentation, multi-year dataset integration, and customizable training routines.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#Usage)
- [Google Colab Setup](#google-colab-setup)
- [Theory Behind the Model](#theory-behind-the-model)
- [Current State](#current-state)
- [Future Steps](#future-steps)
- [Contribution Instructions](#contribution-instructions)
- [License](#licence)

## Installation
To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, use the following command:

```bash
python scripts/main.py
```
This will download the Pascal VOC datasets as well as building the model.

### Running Inference

To run inference on a video, use the scripts/inference_video.py script:

```bash
python scripts/inference_video.py
```
Specify the paths to your trained model and input video in the script.

## Google Colab Setup
The project includes a Google Colab notebook for training and inference. You can find the notebook in the notebooks/ directory.

(I recomend using this unless you have a very good GPU)

### Training on Google Colab

1. Open the ssd_model_compiler.ipynb notebook in Google Colab.
2. Follow the instructions in the notebook to set up the environment, upload the datasets, and train the model.
3. Save the trained model to your Google Drive.

### Running Inference
1. Open the ssd_model_inference.ipynb in Google Colab.
2. Follow the instructions in the notebook to set up the environment, upload the trained model, input video, and supporting files.
3. View the output video.

## Theory Behind the Model
SSD is a popular object detection model known for its balance between speed and accuracy. It detects objects in images in a single forward pass through the network, making it efficient for real-time applications. The model uses default boxes of different aspect ratios and scales for detecting objects and applies non-maximum suppression to produce final detections.

### Architecture
- Base Convolutions: Derived from the VGG-16 architecture, pretrained on ImageNet.
- Auxiliary Convolutions: Added on top of the base network to provide higher-level feature maps.
- Prediction Convolutions: Used for locating and identifying objects in these feature maps.

### Loss Functions
- Localization Loss: Smooth L1 loss used for bounding box regression.
- Confidence Loss: Cross-entropy loss used for object classification.

## Current State
The current implementation identifies objects in videos but requires improvement in detection accuracy and robustness.

## Future Steps
1. Hyperparameter Tuning: Adjust learning rate, batch size, and number of epochs to improve model performance.
2. Data Augmentation: Incorporate more advanced augmentation techniques.
3. Fine-Tuning: Fine-tune the model on a larger dataset like COCO.
4. Advanced Architectures: Experiment with architectures like Faster R-CNN, YOLO, or EfficientDet.

## Contribution Instructions
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

