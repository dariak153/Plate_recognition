# Character Recognition System

## Overview
This project develops a character recognition system using a Random Forest Classifier, it classifies characters from various images, crucial for applications such as automatic license plate recognition.
## Result
| Filename    | Identified License Plate |
|-------------|--------------------------|
| CIN20356.jpg | CIN20356                 |
| CMG21FG.jpg | CMG21FG                  |
| FSD23429.jpg | FSD23429                 |
| PCT15PY.jpg | PCT15PY                  |
| PGN141GR.jpg | PGN141GR                 |
| PGN756EC.jpg | PGN756EC                 |
| PKL8C63.jpg | PKL8C63                  |
| PKRR788.jpg | PKRR788L                 |
| PKS30W3.jpg | PKS30W3                  |
| PO033AX.jpg | POO33AX                  |
| PO096NT.jpg | PO096NT                  |
| PO155KU.jpg | PO155KU                  |
| PO2J735.jpg | PO2J735                  |
| PO2W494.jpg | PO2W494                  |
| PO522WJ.jpg | PO522WJ                  |
| PO5T224.jpg | PO5T224                  |
| PO6K534.jpg | PO6K534                  |
| PO778SS.jpg | PO778SS                  |
| POBTC81.jpg | POBTC81                  |
| POZS221.jpg | POZS221                  |
| PSE22800.jpg | PSE22800                 |
| PSZ47620.jpg | PSZ47620                 |
| PZ0460J.jpg | PZ0460                   |
| PZ492AK.jpg | PZ492AK                  |
| WSCUP62.jpg | WSC7UG62                 |
| ZSL17729.jpg | ZSL17729                 |

## Methods Used
- **Image Processing**: The images are initially processed to improve the classifier's accuracy. This involves loading images from a dataset, converting them to grayscale to reduce complexity, applying a binary threshold to separate characters from the background, and dilating to emphasize features.
- **Data Preparation**: Each processed image is resized to a uniform dimension (120x120 pixels) and flattened into a one-dimensional vector. This vectorization transforms the image data into a format suitable for machine learning models.
- **Model Training**: Utilizing the RandomForestClassifier for training on processed image data.
- **Model Evaluation**: After training, the model's effectiveness is evaluated through metrics such as precision, recall, and F1-score across a test dataset. These metrics help quantify the model’s ability to classify each character correctly.
- **Model Serialization**: The fully trained model is serialized using Python’s `pickle` module, enabling it to be saved and reloaded for future use without retraining.

## Files Overview
- `characters_dataset/`: Contains labeled images of characters. Each label directory houses images representing a specific character, aiding in supervised learning.
- `characters_recognizer_rf.pkl`: This file is a serialized version of the trained Random Forest model, ready for deployment in character recognition tasks.
-  `templates/`: directory contains a comprehensive template featuring all the characters used on license plates. This template is designed with a font style closely resembling that found on vehicle registration plates, making it ideal for training the character recognition system. 

## Data Preparation and Processing
1. **Image Loading and Processing**: This crucial step involves several image manipulations to prepare data for training:
   - **Grayscale Conversion**: Converts the RGB image to grayscale to reduce computational load.
   - **Thresholding**: Applies a binary threshold to make the image binary, which is useful for isolating characters from backgrounds.
   - **Dilation**: Enhances features of the character in the binary image.
   - **Resizing and Flattening**: Standardizes the size of images and flattens them into arrays for machine learning processing.
2. **Dataset Compilation**: Compiles the feature vectors (`X`) and labels (`y`) from the processed images, ensuring accurate correspondence between features and targets.

## Model Training and Evaluation
1. **Training Data Split**: The dataset is split into 80% for training and 20% for testing, maintaining a balance between learning and validation capabilities to prevent overfitting.
2. **Random Forest Classifier**: Trains on the flattened image data, learning to recognize and classify each character based on its features.
3. **Performance Evaluation**: The model’s performance is evaluated using classification metrics such as precision, recall, and F1-score, providing insights into its accuracy and effectiveness.


# Classification Metrics Summary for License Plate Characters

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 1.00      | 1.00   | 1.00     | 38      |
| 1         | 1.00      | 1.00   | 1.00     | 31      |
| 2         | 1.00      | 1.00   | 1.00     | 29      |
| 3         | 1.00      | 1.00   | 1.00     | 35      |
| 4         | 1.00      | 1.00   | 1.00     | 35      |
| 5         | 1.00      | 1.00   | 1.00     | 33      |
| 6         | 1.00      | 1.00   | 1.00     | 29      |
| 7         | 1.00      | 1.00   | 1.00     | 30      |
| 8         | 1.00      | 1.00   | 1.00     | 29      |
| 9         | 1.00      | 1.00   | 1.00     | 36      |
| A         | 1.00      | 1.00   | 1.00     | 36      |
| B         | 1.00      | 1.00   | 1.00     | 34      |
| C         | 1.00      | 1.00   | 1.00     | 26      |
| D         | 1.00      | 1.00   | 1.00     | 22      |
| E         | 1.00      | 1.00   | 1.00     | 25      |
| F         | 1.00      | 1.00   | 1.00     | 34      |
| G         | 1.00      | 1.00   | 1.00     | 28      |
| H         | 1.00      | 1.00   | 1.00     | 32      |
| I         | 1.00      | 1.00   | 1.00     | 23      |
| J         | 1.00      | 1.00   | 1.00     | 26      |
| K         | 1.00      | 1.00   | 1.00     | 27      |
| L         | 1.00      | 1.00   | 1.00     | 25      |
| M         | 1.00      | 1.00   | 1.00     | 25      |
| N         | 1.00      | 1.00   | 1.00     | 31      |
| O         | 1.00      | 1.00   | 1.00     | 31      |
| P         | 1.00      | 1.00   | 1.00     | 25      |
| R         | 1.00      | 1.00   | 1.00     | 29      |
| S         | 1.00      | 1.00   | 1.00     | 30      |
| T         | 1.00      | 1.00   | 1.00     | 42      |
| U         | 1.00      | 1.00   | 1.00     | 30      |
| V         | 1.00      | 1.00   | 1.00     | 38      |
| W         | 1.00      | 1.00   | 1.00     | 30      |
| X         | 1.00      | 1.00   | 1.00     | 31      |
| Y         | 1.00      | 1.00   | 1.00     | 25      |
| Z         | 1.00      | 1.00   | 1.00     | 28      |

|    | precision | recall | f1-score | support |
|----|-----------|--------|----------|---------|
| accuracy |           |        | 1.00     | 1058    |
| macro avg | 1.00      | 1.00   | 1.00     | 1058    |
| weighted avg | 1.00      | 1.00   | 1.00     | 1058    |

