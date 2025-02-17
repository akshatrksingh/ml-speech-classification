import numpy as np
import pandas as pd
import librosa
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the main classes we want to keep
main_classes = ['confused', 'enunciated', 'happy', 'laughing', 'default', 'sad', 'whisper']

# Load dataset
ds = load_dataset("ylacombe/expresso")["train"]

# Filter dataset
filtered_ds = ds.filter(lambda x: x['style'] in main_classes)

# Feature extraction function
def extract_features(audio, sr):
    features = {}
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_var'] = np.var(mfccs[i])
    
    # Other features
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    features['rms'] = np.mean(librosa.feature.rms(y=audio))
    
    return features

# Prepare dataset
def prepare_dataset(filtered_ds, test_size=0.2, random_state=42):
    features_list = []
    labels = []
    
    for sample in tqdm(filtered_ds, desc="Extracting features"):
        try:
            features = extract_features(sample['audio']['array'], sample['audio']['sampling_rate'])
            features_list.append(features)
            labels.append(sample['style'])
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    feature_df = pd.DataFrame(features_list)
    
    # Save processed features to CSV
    feature_df["style"] = labels  # Add the labels to the dataframe
    feature_df.to_csv("data/processed_features.csv", index=False)
    print("Processed features saved to 'data/processed_features.csv'")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df.drop(columns=["style"]), labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_dataset(filtered_ds)
    print("Feature extraction and dataset preparation complete!")