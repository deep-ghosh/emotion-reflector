"""
Setup script for Face Emotion Recognition Model
This script will help you set up the required files and dataset
"""

import os
import urllib.request
import zipfile
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def create_model_architecture():
    """Create the emotion recognition model architecture"""
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(7, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model_files():
    """Save model architecture to JSON file"""
    try:
        model = create_model_architecture()
        
        # Save model architecture
        model_json = model.to_json()
        with open("facialemotionmodel.json", "w") as json_file:
            json_file.write(model_json)
        
        # Initialize with random weights and save
        model.save_weights("facialemotionmodel.h5")
        
        print("✅ Model files created successfully!")
        print("📁 Created: facialemotionmodel.json")
        print("📁 Created: facialemotionmodel.h5")
        print("⚠️  Note: These are untrained models. For accurate results, you need to train the model.")
        
    except Exception as e:
        print(f"❌ Error creating model files: {e}")

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        "images",
        "images/train",
        "images/test",
        "images/train/angry",
        "images/train/disgust", 
        "images/train/fear",
        "images/train/happy",
        "images/train/neutral",
        "images/train/sad",
        "images/train/surprise",
        "images/test/angry",
        "images/test/disgust",
        "images/test/fear", 
        "images/test/happy",
        "images/test/neutral",
        "images/test/sad",
        "images/test/surprise"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created!")

def download_sample_data():
    """Instructions for downloading the FER2013 dataset"""
    print("\n📥 To get the FER2013 dataset:")
    print("1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Download the dataset")
    print("3. Extract it to the 'images' folder")
    print("4. The structure should be:")
    print("   images/")
    print("   ├── train/")
    print("   │   ├── angry/")
    print("   │   ├── disgust/")
    print("   │   ├── fear/")
    print("   │   ├── happy/")
    print("   │   ├── neutral/")
    print("   │   ├── sad/")
    print("   │   └── surprise/")
    print("   └── test/")
    print("       ├── angry/")
    print("       ├── disgust/")
    print("       ├── fear/")
    print("       ├── happy/")
    print("       ├── neutral/")
    print("       ├── sad/")
    print("       └── surprise/")

def main():
    print("🎭 Face Emotion Recognition Setup")
    print("=" * 40)
    
    # Create directory structure
    create_directory_structure()
    
    # Create model files
    save_model_files()
    
    # Show dataset download instructions
    download_sample_data()
    
    print("\n🚀 Next Steps:")
    print("1. Download the FER2013 dataset (see instructions above)")
    print("2. Run the training notebook to train the model")
    print("3. Then run realtimedetection.py for real-time emotion detection")
    print("\nFor immediate testing, run: python simple_emotion_detector.py")

if __name__ == "__main__":
    main()
