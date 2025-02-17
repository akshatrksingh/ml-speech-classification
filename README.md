# ML Speech Classification

## Overview
A speech classification pipeline using ML and deep learning models.
Classifies different speaking styles from the Expresso dataset using extracted MFCC and spectral features.

## Dataset: Expresso
Source: https://huggingface.co/datasets/ylacombe/expresso

Speaking Styles:
- Confused
- Enunciated
- Happy
- Laughing
- Default
- Sad
- Whisper

## Data Processing & Features
Extracted Features:
- MFCCs (Mel Frequency Cepstral Coefficients)
- Zero Crossing Rate
- Spectral Centroid
- Spectral Rolloff
- RMS Energy

Features stored in: data/processed_features.csv

## Project Structure
```
ml-speech-classification/
│── data/                     # Dataset and processed features
│   ├── processed_features.csv # Extracted features from audio
│   ├── train.csv             # Training split
│   ├── test.csv              # Testing split
│── models/                   # Trained models & scalers
│   ├── rf_model.pkl          # Random Forest model
│   ├── nn_model.pkl          # Neural Network model
│   ├── scaler.pkl            # Feature scaler
│   ├── label_encoder.pkl     # Label encoder
│── output/                   # Evaluation results
│   ├── conf_rf.png           # RF Confusion Matrix
│   ├── conf_nn.png           # NN Confusion Matrix
│── src/                      # Code files
│   ├── prep_features.py      # Feature extraction script
│   ├── data_splitter.py      # Train-test split
│   ├── model_rf.py           # Train Random Forest
│   ├── model_nn.py           # Train Neural Network
│── notebooks/                # Jupyter Notebooks
│── requirements.txt          # Dependencies
│── .gitignore               # Ignore unnecessary files
│── README.md                # Project documentation
```

## Model Training Steps

1. Extract Features:
```bash
python src/prep_features.py
```

2. Split Data:
```bash
python src/data_splitter.py
```

3. Train Random Forest:
```bash
python src/model_rf.py
```

4. Train Neural Network:
```bash
python src/model_nn.py
```

## Model Performance

Random Forest Results:
- Accuracy: 0.75
- Precision: 0.74
- Recall: 0.75
- F1 Score: 0.75
- Confusion Matrix: output/conf_rf.png

Neural Network Results:
- Accuracy: 0.87
- Precision: 0.87
- Recall: 0.87
- F1 Score: 0.87
- Confusion Matrix: output/conf_nn.png

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/akshatrksingh/ml-speech-classification.git
cd ml-speech-classification

# Install dependencies
pip install -r requirements.txt
```
